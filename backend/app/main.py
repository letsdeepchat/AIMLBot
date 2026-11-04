"""HTTP API. Thin by design: validation and storage here, all ML in quest_core."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel

from quest_core import advisor as adv
from quest_core import endgame as eg
from quest_core import experiments as ex
from quest_core.pipeline import DATA_SUFFIXES, find_file, load_inputs, load_table
from quest_core.recipes import STEP_CATALOG, FeatureRecipe, validate_recipe
from quest_core.sandbox import run_features
from quest_core.spec import METRICS, ChallengeSpec, infer_task, resolve_metric

from .config import load_settings
from .db import Store
from .jobs import JobRejected, challenge_dir, enqueue

settings = load_settings()
store = Store(settings.db_path)
CHUNK = 1024 * 1024
_MODEL_KEY = re.compile(r"^[a-z0-9_]+$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    (settings.data_dir / "challenges").mkdir(parents=True, exist_ok=True)
    if settings.job_backend == "inline":
        store.fail_interrupted_jobs()
    yield


app = FastAPI(title="Quest Copilot", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers


def _blank_to_none(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _require_challenge(challenge_id: str) -> dict:
    challenge = store.get_challenge(challenge_id)
    if not challenge:
        raise HTTPException(404, "No challenge with that id.")
    return challenge


def _with_status(challenge: dict) -> dict:
    root = challenge_dir(settings, challenge["id"])
    jobs = store.jobs_for(challenge["id"])
    return {
        **challenge,
        "reports": {name: (root / f"{name}.json").exists() for name in ("analysis", "baselines")},
        "jobs": jobs,
        "active_job": next((j for j in jobs if j["status"] in ("queued", "running")), None),
    }


async def _save_upload(upload: UploadFile, dest_stem: Path) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in DATA_SUFFIXES:
        raise HTTPException(400, f"'{upload.filename}' isn't a CSV, TSV or Parquet file.")
    dest = dest_stem.with_suffix(suffix)
    limit, written = settings.max_upload_mb * CHUNK, 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"'{upload.filename}' is larger than {settings.max_upload_mb} MB.")
            fh.write(chunk)
    return dest


def _validate_columns(train_path: Path, test_path: Path, fields: dict) -> dict:
    train, test = load_table(train_path), load_table(test_path)
    target = fields["target"]
    if target not in train.columns:
        raise ValueError(f"Target '{target}' isn't a column in the train file.")
    for key, frame, where in (("id_col", test, "test"), ("group_col", train, "train"), ("time_col", train, "train")):
        col = fields.get(key)
        if col and col not in frame.columns:
            raise ValueError(f"'{col}' isn't a column in the {where} file.")
        if col == target:
            raise ValueError(f"'{col}' is the target, so it can't also be the {key.replace('_col', '')} column.")
    y = train[target].dropna()
    if len(y) < 20:
        raise ValueError("The train file has fewer than 20 labelled rows.")
    task = fields.get("task") or infer_task(y)
    metric = resolve_metric(task, fields.get("metric"))
    return {"task": task, "metric": metric.name, "n_train": len(train), "n_test": len(test)}


# ------------------------------------------------------------------- routes


@app.get("/api/health")
def health():
    return {"ok": True, "job_backend": settings.job_backend}


@app.get("/api/metrics")
def metrics():
    return [{"name": m.name, "label": m.label, "tasks": list(m.tasks), "greater_is_better": m.greater_is_better}
            for m in METRICS.values()]


@app.get("/api/challenges")
def list_challenges():
    return [_with_status(c) for c in store.list_challenges()]


@app.post("/api/challenges", status_code=201)
async def create_challenge(
    name: str = Form(...),
    target: str = Form(...),
    description: str = Form(""),
    metric: str | None = Form(None),
    task: str | None = Form(None),
    id_col: str | None = Form(None),
    group_col: str | None = Form(None),
    time_col: str | None = Form(None),
    submission_limit: int | None = Form(None),
    deadline: str | None = Form(None),
    train: UploadFile = File(...),
    test: UploadFile = File(...),
):
    fields = {
        "target": target.strip(),
        "metric": _blank_to_none(metric),
        "task": _blank_to_none(task),
        "id_col": _blank_to_none(id_col),
        "group_col": _blank_to_none(group_col),
        "time_col": _blank_to_none(time_col),
    }
    if fields["task"] and fields["task"] not in ("binary", "multiclass", "regression"):
        raise HTTPException(400, "Task must be binary, multiclass or regression.")

    challenge_id = uuid.uuid4().hex[:12]
    root = challenge_dir(settings, challenge_id)
    root.mkdir(parents=True)
    try:
        train_path = await _save_upload(train, root / "train")
        test_path = await _save_upload(test, root / "test")
        checked = await asyncio.to_thread(_validate_columns, train_path, test_path, fields)
    except HTTPException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except Exception as exc:  # unreadable file, bad column, metric/task mismatch
        shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc

    spec = {**fields, "task": checked["task"], "metric": checked["metric"], "drop_cols": [], "seed": 42}
    challenge = store.create_challenge(
        id=challenge_id, name=name.strip() or "Untitled challenge", description=description, spec=spec,
        task=checked["task"], submission_limit=submission_limit, deadline=_blank_to_none(deadline),
        train_file=train_path.name, test_file=test_path.name,
        n_train=checked["n_train"], n_test=checked["n_test"],
    )
    enqueue(settings, store, challenge_id, "analysis")  # analysis is cheap; start it right away
    return _with_status(challenge)


@app.get("/api/challenges/{challenge_id}")
def get_challenge(challenge_id: str):
    return _with_status(_require_challenge(challenge_id))


@app.delete("/api/challenges/{challenge_id}", status_code=204)
def delete_challenge(challenge_id: str):
    _require_challenge(challenge_id)
    store.delete_challenge(challenge_id)
    shutil.rmtree(challenge_dir(settings, challenge_id), ignore_errors=True)


class JobRequest(BaseModel):
    kind: str


class SpecUpdate(BaseModel):
    id_col: str | None = None
    group_col: str | None = None
    time_col: str | None = None
    drop_cols: list[str] = []


class RecipeBody(BaseModel):
    steps: list[dict]
    code: str | None = None


class ExperimentBody(BaseModel):
    name: str
    model: str
    steps: list[dict]
    hypothesis: str = ""
    parent: str | None = None
    code: str | None = None


class AdviceBody(BaseModel):
    focus: str = ""
    n: int = 5


class AcceptBody(BaseModel):
    code_reviewed: bool = False


class EndgameSettings(BaseModel):
    public_share: float | None = None
    n_picks: int = 2


class SubmissionBody(BaseModel):
    entry_id: str
    public_score: float
    note: str = ""


class BlendBody(BaseModel):
    name: str = ""
    members: list[str]
    method: str = "greedy"


def _columns_of(path: Path) -> list[str]:
    if path.suffix.lower() in (".csv", ".tsv"):
        return list(pd.read_csv(path, nrows=0, sep="\t" if path.suffix.lower() == ".tsv" else ",").columns)
    return list(load_table(path).columns)


def _analysis(challenge_id: str) -> dict:
    path = challenge_dir(settings, challenge_id) / "analysis.json"
    if not path.exists():
        raise HTTPException(409, "Run the analysis first.")
    return json.loads(path.read_text())


def _spec(challenge_id: str) -> ChallengeSpec:
    return ChallengeSpec(**_require_challenge(challenge_id)["spec"])


def _check_recipe(challenge_id: str, steps: list[dict], code: str | None = None) -> dict:
    """Validate steps against the columns available at each step, including feature-code outputs.
    Code problems are reported as step -1."""
    analysis = _analysis(challenge_id)
    kinds, code_columns = ex.feature_kinds(analysis), {}
    if code and code.strip():
        code_columns, problem = ex.code_kinds(challenge_dir(settings, challenge_id), _spec(challenge_id), code)
        if problem:
            return {"errors": [{"step": -1, "message": problem}], "columns_before": [], "output": [], "code_columns": []}
    check = validate_recipe(steps, {**kinds, **code_columns}, analysis["task"], ex.n_classes_of(analysis))
    return {**check, "code_columns": [{"name": c, "kind": k} for c, k in code_columns.items()]}


def _dry_run(challenge_id: str, steps: list[dict], code: str | None, task: str) -> None:
    """Fit the recipe (and code) on a slice of real rows so runtime errors surface before the job starts."""
    spec = _spec(challenge_id)
    inp = load_inputs(challenge_dir(settings, challenge_id), spec)
    n = min(len(inp.X), 300)
    X = inp.X.iloc[:n]
    if code and code.strip():
        train_raw, _ = ex.code_input(inp, spec)
        produced = run_features(code, train_raw.iloc[:n], timeout=60)
        X = pd.concat([X, produced.train.set_index(X.index)], axis=1)
    FeatureRecipe(steps, task).fit_transform(X, inp.y[:n])


@app.post("/api/challenges/{challenge_id}/jobs", status_code=202)
def start_job(challenge_id: str, body: JobRequest):
    _require_challenge(challenge_id)
    try:
        return enqueue(settings, store, challenge_id, body.kind)
    except JobRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/recipe-steps")
def recipe_steps():
    return STEP_CATALOG


@app.get("/api/challenges/{challenge_id}/columns")
def columns(challenge_id: str):
    challenge = _require_challenge(challenge_id)
    root = challenge_dir(settings, challenge_id)
    analysis_path = root / "analysis.json"
    features = []
    if analysis_path.exists():
        features = [{"name": c, "kind": k} for c, k in ex.feature_kinds(json.loads(analysis_path.read_text())).items()]
    return {
        "train": _columns_of(find_file(root, "train")),
        "test": _columns_of(find_file(root, "test")),
        "features": features,
        "spec": challenge["spec"],
    }


@app.patch("/api/challenges/{challenge_id}/spec")
def update_spec(challenge_id: str, body: SpecUpdate):
    challenge = _require_challenge(challenge_id)
    if store.active_job(challenge_id):
        raise HTTPException(409, "Wait for the running job to finish before changing columns.")
    root = challenge_dir(settings, challenge_id)
    train_cols, test_cols = set(_columns_of(find_file(root, "train"))), set(_columns_of(find_file(root, "test")))
    target = challenge["spec"]["target"]
    fields = {"id_col": _blank_to_none(body.id_col), "group_col": _blank_to_none(body.group_col),
              "time_col": _blank_to_none(body.time_col)}
    for key, col in fields.items():
        where = test_cols if key == "id_col" else train_cols
        if col and (col not in where or col == target):
            raise HTTPException(400, f"'{col}' can't be the {key.replace('_col', '')} column.")
    unknown = [c for c in body.drop_cols if c not in train_cols or c == target]
    if unknown:
        raise HTTPException(400, f"Can't drop: {', '.join(unknown)}.")
    spec = {**challenge["spec"], **fields, "drop_cols": sorted(set(body.drop_cols))}
    store.update_spec(challenge_id, spec)
    enqueue(settings, store, challenge_id, "analysis")  # new settings mean a new plan; baselines become stale
    return _with_status(store.get_challenge(challenge_id))


@app.post("/api/challenges/{challenge_id}/recipes/validate")
def validate(challenge_id: str, body: RecipeBody):
    _require_challenge(challenge_id)
    return _check_recipe(challenge_id, body.steps, body.code)


def _experiment_view(experiment: dict, jobs: dict, fingerprint: str | None) -> dict:
    job = jobs.get(experiment["id"])
    result = experiment.get("result") or {}
    return {
        **experiment,
        "result": {k: v for k, v in result.items() if k != "fold_scores"} if result else None,
        "job": job,
        "stale": bool(result) and experiment.get("fingerprint") != fingerprint,
    }


@app.get("/api/challenges/{challenge_id}/experiments")
def list_experiments(challenge_id: str):
    _require_challenge(challenge_id)
    root = challenge_dir(settings, challenge_id)
    fingerprint = json.loads((root / "analysis.json").read_text()).get("fingerprint") if (root / "analysis.json").exists() else None
    jobs = store.latest_job_by_ref(challenge_id)
    return [_experiment_view(e, jobs, fingerprint) for e in ex.list_experiments(root)]


@app.post("/api/challenges/{challenge_id}/experiments", status_code=201)
def create_experiment(challenge_id: str, body: ExperimentBody):
    return _create_experiment(challenge_id, body)


def _create_experiment(challenge_id: str, body: ExperimentBody, source: dict | None = None) -> dict:
    _require_challenge(challenge_id)
    root = challenge_dir(settings, challenge_id)
    if not (root / "baselines.json").exists():
        raise HTTPException(409, "Run the baselines first. Each experiment is compared against its baseline model.")
    analysis = _analysis(challenge_id)
    if not body.steps and not (body.code or "").strip():
        raise HTTPException(400, "Add at least one feature step or feature code.")
    check = _check_recipe(challenge_id, body.steps, body.code)
    if check["errors"]:
        first = check["errors"][0]
        where = "Feature code" if first["step"] < 0 else f"Step {first['step'] + 1}"
        raise HTTPException(400, f"{where}: {first['message']}")
    available = {m["key"] for m in json.loads((root / "baselines.json").read_text())["models"] if m["status"] == "ok"}
    if body.model not in available:
        raise HTTPException(400, f"'{body.model}' has no baseline to compare against. Pick one of: {', '.join(sorted(available))}.")

    try:
        _dry_run(challenge_id, body.steps, body.code, analysis["task"])
    except Exception as exc:
        raise HTTPException(400, f"The recipe fails on real rows: {exc}") from exc

    experiment = ex.create_experiment(root, body.name, body.model, body.steps, body.hypothesis, body.parent, body.code, source)
    try:
        job = enqueue(settings, store, challenge_id, "experiment", experiment["id"])
    except JobRejected as exc:
        raise HTTPException(409, f"Saved, but it can't start yet: {exc}") from exc
    return {**experiment, "job": job, "stale": False}


@app.post("/api/challenges/{challenge_id}/experiments/{exp_id}/run", status_code=202)
def rerun_experiment(challenge_id: str, exp_id: str):
    _require_challenge(challenge_id)
    try:
        ex.load_experiment(challenge_dir(settings, challenge_id), exp_id)
        return enqueue(settings, store, challenge_id, "experiment", exp_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except JobRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@app.delete("/api/challenges/{challenge_id}/experiments/{exp_id}", status_code=204)
def delete_experiment(challenge_id: str, exp_id: str):
    _require_challenge(challenge_id)
    if store.active_job(challenge_id, "experiment", exp_id):
        raise HTTPException(409, "That experiment is running. Delete it when it finishes.")
    try:
        ex.delete_experiment(challenge_dir(settings, challenge_id), exp_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/challenges/{challenge_id}/experiments/{exp_id}/submission")
def experiment_submission(challenge_id: str, exp_id: str):
    challenge = _require_challenge(challenge_id)
    try:
        experiment = ex.load_experiment(challenge_dir(settings, challenge_id), exp_id)
        path = ex.submission_path(challenge_dir(settings, challenge_id), exp_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "No experiment with that id.") from exc
    if not path.exists():
        raise HTTPException(404, "This experiment hasn't finished yet.")
    slug = re.sub(r"[^a-z0-9]+", "-", f"{challenge['name']} {experiment['name']}".lower()).strip("-")
    return FileResponse(path, media_type="text/csv", filename=f"{slug}.csv")


@app.get("/api/advisor")
def advisor_status():
    return {"configured": adv.is_configured(), "model": adv.advisor_model()}


@app.get("/api/challenges/{challenge_id}/advice")
def list_advice(challenge_id: str):
    _require_challenge(challenge_id)
    jobs = store.latest_job_by_ref(challenge_id)
    return [{**{k: v for k, v in s.items() if k != "description"}, "job": jobs.get(s["id"])}
            for s in adv.list_sessions(challenge_dir(settings, challenge_id))]


@app.post("/api/challenges/{challenge_id}/advice", status_code=202)
def request_advice(challenge_id: str, body: AdviceBody):
    challenge = _require_challenge(challenge_id)
    if not adv.is_configured():
        raise HTTPException(409, "The advisor needs ANTHROPIC_API_KEY set for the API and the worker.")
    if store.active_job(challenge_id, "advice"):
        raise HTTPException(409, "The advisor is already working on this challenge.")
    root = challenge_dir(settings, challenge_id)
    if not (root / "baselines.json").exists():
        raise HTTPException(409, "Run the baselines first. The advisor measures ideas against them.")
    session = adv.create_session(root, body.focus, max(1, min(body.n, 8)), challenge["description"], challenge["name"])
    try:
        job = enqueue(settings, store, challenge_id, "advice", session["id"])
    except JobRejected as exc:
        raise HTTPException(409, str(exc)) from exc
    return {**session, "job": job}


@app.post("/api/challenges/{challenge_id}/advice/{session_id}/proposals/{proposal_id}/accept", status_code=201)
def accept_proposal(challenge_id: str, session_id: str, proposal_id: str, body: AcceptBody):
    _require_challenge(challenge_id)
    root = challenge_dir(settings, challenge_id)
    try:
        proposal = adv.find_proposal(root, session_id, proposal_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    if proposal["status"] == "accepted" and proposal["experiment_id"]:
        return ex.load_experiment(root, proposal["experiment_id"])
    if proposal["errors"]:
        raise HTTPException(400, "This proposal didn't validate. Open it in the builder and fix it first.")
    if proposal["code"] and not body.code_reviewed:
        raise HTTPException(400, "Read the feature code and confirm before running it.")
    created = _create_experiment(challenge_id, ExperimentBody(
        name=proposal["title"], model=proposal["model"], steps=proposal["steps"],
        hypothesis=proposal["hypothesis"], code=proposal["code"]), source={"advice": session_id, "proposal": proposal_id})
    adv.set_proposal_status(root, session_id, proposal_id, "accepted", created["id"])
    return created


@app.post("/api/challenges/{challenge_id}/advice/{session_id}/proposals/{proposal_id}/dismiss")
def dismiss_proposal(challenge_id: str, session_id: str, proposal_id: str):
    _require_challenge(challenge_id)
    try:
        return adv.set_proposal_status(challenge_dir(settings, challenge_id), session_id, proposal_id, "dismissed")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/challenges/{challenge_id}/endgame")
def endgame(challenge_id: str):
    challenge = _require_challenge(challenge_id)
    summary = eg.endgame_summary(challenge_dir(settings, challenge_id), _spec(challenge_id))
    jobs = store.latest_job_by_ref(challenge_id)
    for blend in summary.get("blends", []):
        blend["job"] = jobs.get(blend["id"])
    summary["submission_limit"] = challenge["submission_limit"]
    return summary


@app.put("/api/challenges/{challenge_id}/endgame/settings")
def endgame_settings(challenge_id: str, body: EndgameSettings):
    _require_challenge(challenge_id)
    try:
        return eg.save_settings(challenge_dir(settings, challenge_id), body.public_share, body.n_picks)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/challenges/{challenge_id}/endgame/submissions", status_code=201)
def add_submission(challenge_id: str, body: SubmissionBody):
    _require_challenge(challenge_id)
    root = challenge_dir(settings, challenge_id)
    summary = eg.endgame_summary(root, _spec(challenge_id))
    entry = next((c for c in summary.get("candidates", []) if c["id"] == body.entry_id), None)
    if not entry:
        raise HTTPException(400, "Pick a current baseline, experiment or blend.")
    return eg.add_submission(root, entry["id"], entry["label"], body.public_score, body.note, entry["cv_mean"])


@app.delete("/api/challenges/{challenge_id}/endgame/submissions/{submission_id}", status_code=204)
def delete_submission(challenge_id: str, submission_id: str):
    _require_challenge(challenge_id)
    eg.delete_submission(challenge_dir(settings, challenge_id), submission_id)


@app.post("/api/challenges/{challenge_id}/endgame/blends", status_code=201)
def create_blend(challenge_id: str, body: BlendBody):
    _require_challenge(challenge_id)
    root = challenge_dir(settings, challenge_id)
    try:
        blend = eg.create_blend(root, body.name, body.members, body.method)
        job = enqueue(settings, store, challenge_id, "blend", blend["id"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**blend, "job": job}


@app.delete("/api/challenges/{challenge_id}/endgame/blends/{blend_id}", status_code=204)
def delete_blend(challenge_id: str, blend_id: str):
    _require_challenge(challenge_id)
    if store.active_job(challenge_id, "blend", blend_id):
        raise HTTPException(409, "That blend is running. Delete it when it finishes.")
    try:
        eg.delete_blend(challenge_dir(settings, challenge_id), blend_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/challenges/{challenge_id}/endgame/blends/{blend_id}/submission")
def blend_submission(challenge_id: str, blend_id: str):
    challenge = _require_challenge(challenge_id)
    try:
        path = eg.blend_submission_path(challenge_dir(settings, challenge_id), blend_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404, "This blend hasn't finished yet.")
    slug = re.sub(r"[^a-z0-9]+", "-", f"{challenge['name']} blend {blend_id}".lower()).strip("-")
    return FileResponse(path, media_type="text/csv", filename=f"{slug}.csv")


@app.get("/api/challenges/{challenge_id}/leaderboard")
def leaderboard(challenge_id: str):
    _require_challenge(challenge_id)
    return ex.build_leaderboard(challenge_dir(settings, challenge_id))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "No job with that id.")
    return job


@app.get("/api/challenges/{challenge_id}/reports/{name}")
def get_report(challenge_id: str, name: str):
    _require_challenge(challenge_id)
    if name not in ("analysis", "baselines"):
        raise HTTPException(404, "Reports are 'analysis' or 'baselines'.")
    path = challenge_dir(settings, challenge_id) / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"The {name} hasn't been run yet.")
    return FileResponse(path, media_type="application/json")


@app.get("/api/challenges/{challenge_id}/submissions/{model}")
def download_submission(challenge_id: str, model: str):
    challenge = _require_challenge(challenge_id)
    if not _MODEL_KEY.match(model):
        raise HTTPException(404, "Unknown model.")
    path = challenge_dir(settings, challenge_id) / "submissions" / f"{model}.csv"
    if not path.exists():
        raise HTTPException(404, "No submission file for that model. Run the baselines first.")
    slug = re.sub(r"[^a-z0-9]+", "-", challenge["name"].lower()).strip("-") or "challenge"
    return FileResponse(path, media_type="text/csv", filename=f"{slug}-{model}.csv")


if settings.static_dir and settings.static_dir.exists():
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="ui")
