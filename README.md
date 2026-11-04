# Quest Copilot — weeks 1 to 4

A human-in-the-loop workbench for small-data ML challenges. You choose a challenge and upload its files;
the app tells you whether the data and validation can be trusted, then runs baselines and shows which models
are genuinely ahead rather than lucky. You review everything and submit yourself.

## What week 1 does

| Stage | What you get |
|---|---|
| Data health | Column kinds, target profile, and leakage checks: single-column target proxies, ID-like columns that predict the target, duplicates and conflicting labels, train/test row overlap, row-order signal, likely group columns. |
| Shift | Adversarial validation AUC, per-column PSI, KS and single-column separation, and each train row's out-of-fold test-likeness. |
| Validation plan | Time-based, grouped or repeated stratified CV chosen from the data, with more repeats on small datasets. Every choice has a written reason. Under moderate or severe shift, a 20% test-like holdout is added. |
| Baselines | Prior, linear, k-NN, random forest and sklearn HistGBM always; LightGBM and CatBoost (native categoricals) when installed. Identical splits for every model, gaps to the leader with Nadeau–Bengio corrected standard errors, holdout scores, and a starting submission CSV per model. |

## What week 2 adds

| Feature | What you get |
|---|---|
| Feature recipes | Ordered, declarative steps: formula, log transform, clip outliers, missing indicators, row statistics, date parts, frequency encoding, target encoding, group statistics, combined categories, drop. |
| Fold safety | The recipe is the first stage of the model pipeline, so everything learned from data (encodings, group stats, frequencies, clip thresholds) is learned inside each training fold. Target encoding also cross-fits inside the fold. |
| Safe formulas | Expressions like `abs(log(a / b))` are parsed into an AST and checked against a whitelist of numpy functions. No `eval`, no attribute access, no imports. |
| Live validation | The builder asks the server which columns exist before each step, so later steps can use columns earlier steps create, and errors appear on the step that caused them. A dry run on real rows catches the rest before the job starts. |
| Experiments | A recipe plus a baseline model, run on the identical splits. Each gets a verdict against the same model without the recipe (helps, hurts or no clear change, by corrected paired test), its holdout change, and single-column strength for the new columns. |
| Combined leaderboard | Baselines and experiments on one forest plot, each paired with the leader. |
| Staleness guard | Every result carries a fingerprint of the settings behind it (rows, folds, holdout, ID/group/time/dropped columns, metric). Results from other settings are marked stale instead of being compared with incompatible folds. |
| Column settings | Set the ID, group and time columns and exclude columns (flagged ones are outlined). Saving re-runs the analysis. |

Each analysis, baseline and experiment run is logged to MLflow (or to `runs.jsonl` when MLflow isn't configured), and
out-of-fold and test predictions are saved under `preds/` for blending in week 4.

## What week 3 adds: the Claude advisor

| Feature | What you get |
|---|---|
| Ranked proposals | Claude reads a compact summary of the analysis, baselines and every experiment so far (with verdicts) and proposes experiments, each with its hypothesis, the specific numbers behind it, the expected effect and the risk. |
| Validated before you see them | Every proposal is checked by the same validator and sandbox the experiment runner uses. Invalid ones go back to Claude once with the exact errors; anything still invalid is shown with its errors and can't be run. |
| Feature code | For ideas the steps can't express (text parsing, multi-branch rules), a proposal can include a `features(df)` function. Its output columns can feed recipe steps. You can also write code yourself in the builder. |
| Human approval | Run a proposal as-is, open it in the builder to edit first, or dismiss it. Code proposals stay locked until you open the code and confirm you've read it. |

### Turning it on

Set `ANTHROPIC_API_KEY` in `.env` (the API and the worker both read it) and restart. The default model is
`claude-opus-5-5`; set `QUEST_ADVISOR_MODEL` to change it. A typical request uses 5–15k tokens, and twice
that when a repair round is needed.

### What leaves your machine

Only summaries: column names, kinds and profile statistics (including up to three example category values per
column), findings, shift scores, baseline and experiment scores, recipes, and the problem description you
pasted. Raw rows are never sent. The description is wrapped and marked as untrusted, because text pasted from a
website can contain instructions aimed at the model.

### How feature code is contained

- **Static check:** imports limited to numpy, pandas, math and re; no file, process, network or introspection
  builtins; no underscore attributes; no pandas or numpy I/O helpers.
- **Separate process:** isolated-mode Python with an empty environment (your API key never reaches it), a
  temporary working directory, and CPU, memory, file-size and open-file limits it sets on itself before
  reading any input, plus a wall-clock timeout.
- **Safe results:** output comes back as JSON and `.npy` read with `allow_pickle=False`, so nothing the code
  writes is ever unpickled.
- **Fold safety, enforced:** the code runs on a sample in one batch and again in shuffled chunks. If any
  output differs, it depends on other rows (a column mean, a rank, a shift) and would leak across CV folds,
  so it's rejected with a pointer to the fold-safe recipe step for the job.
- **Never the target:** code receives the feature columns (text included) and the time column, never the
  target.

This is defence in depth for code you have read, not a boundary for hostile code. The sandbox runs inside the
worker, which can reach the network. For untrusted code, run feature-code jobs in a separate container with
networking disabled.

## What week 4 adds: the endgame

| Feature | What you get |
|---|---|
| Consistent scoring | Every candidate (baseline, experiment, blend) is scored the same way here: from its saved out-of-fold predictions on each fold of the shared plan. Fold-model scores and averaged OOF scores differ slightly under repeated CV, so mixing them would quietly favour blends. |
| Public LB noise | Enter the public share of the test set. Bootstrapping your OOF predictions at that size shows how far a public and a private score move on luck alone, and the smallest public gap between your two best models that means anything. |
| Submission log | Log each submission's public score. You see CV next to public, the remaining budget, Spearman agreement between the two orderings, and a flag on any submission whose public score is out of line with the usual CV–public gap by more than the noise ("lucky" or "unlucky" public split). |
| Honest blends | Equal average, rank average (ranking metrics only) or greedy ensemble selection. Greedy weights are fitted on the other folds and scored on the held-out fold, so a blend's score isn't inflated by choosing and judging weights on the same rows. Each blend gets a verdict against its best member on identical folds. |
| Correlations | Out-of-fold prediction correlations among the top candidates, because blends gain most from strong entries that disagree. |
| Final picks | Pick 1 is the candidate with the best score after subtracting its fold uncertainty. Under train/test shift, a statistically tied candidate with a better test-like holdout takes its place. Further picks are the least-correlated candidates statistically tied with pick 1, as a hedge. Each pick comes with its reasons, and the panel says when your best public score is probably luck. |

The picks are deterministic rules, not model output, so they're reproducible and explainable. They can't know
things you know about the competition, so treat them as a strong default rather than a verdict.

## Upgrading from week 1

Copy the new code over and restart. The database gains a `jobs.ref` column automatically. Then, for each
challenge, re-run the analysis and the baselines once so their reports carry a settings fingerprint.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: the backend test suite (including
the API tests), a production build of the UI, and a Docker image build.

## Run it with Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

- App: http://localhost:8000
- MLflow: http://localhost:5000

Try it with `examples/churn_train.csv` and `examples/churn_test.csv` (target `target`, ID `row_id`).
That dataset has planted covariate shift and a repeating `customer_id`, so most checks have something to say.

## Run it without Docker

```bash
# API with jobs running in-process (no Redis needed)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
QUEST_JOB_BACKEND=inline uvicorn app.main:app --reload --port 8000

# UI, in a second terminal
cd frontend
npm install
npm run dev        # http://localhost:5173, forwards /api to :8000
```

Set `MLFLOW_TRACKING_URI` to log to an MLflow server; leave it unset to log to `data/challenges/<id>/runs.jsonl`.

## Use it from your phone

Run it on your GPU box and reach it over Tailscale. The service worker (offline shell, installable app) needs
HTTPS on anything other than localhost, so expose it with:

```bash
tailscale serve --bg 8000
```

then open the `https://<machine>.<tailnet>.ts.net` address on your phone and use **Add to Home Screen**.

## Tests

```bash
cd backend && pytest
```

`tests/test_core.py` plants leaks, shift, groups and time order in synthetic data and checks they are found.
`tests/test_endgame.py` shows that greedily blending 12 pure-noise predictors looks like a gain in-sample but
scores about 0.5 when weights are fitted on other folds, and checks noise estimates, lucky-split flags, rank-blend
refusal for log loss, and picks when a blend leads under shift. `tests/test_advisor.py` drives the advisor with a scripted stand-in for Claude: the forced tool call, the wrapped
description, a repair round that returns only the failing proposals' errors, and an approved code proposal
running as an experiment. `tests/test_experiments.py` checks formula safety, step-by-step validation, that target encoding of pure noise
stays near AUC 0.5, that a genuinely useful feature is judged "better", and that changed settings make results stale.
`tests/test_jobs.py` runs analysis and baselines end to end through the job runner. `tests/test_api.py`
exercises upload, validation, jobs and downloads over HTTP.

## Layout

```
backend/
  quest_core/      ML logic, no web dependencies; usable from notebooks
    encoding.py    column kinds and feature preparation
    spec.py        challenge spec, task inference, metrics
    profiling.py   dataset and target profile
    univariate.py  single-column strength (tree, or out-of-fold target encoding for categoricals)
    leakage.py     integrity and leakage findings
    shift.py       adversarial validation and per-column drift
    cv_designer.py validation plan and splits
    baselines.py   model zoo, shared evaluate(), corrected comparison, submissions
    recipes.py     feature steps, safe formulas, validation, fold-safe transformer
    experiments.py experiment runs, control verdicts, combined leaderboard
    sandbox.py     static checks and process isolation for feature code
    _sandbox_runner.py  the restricted child process
    advisor.py     context building, Claude call, validation and repair
    endgame.py     consistent scoring, public LB noise, submission log, honest blends, final picks
    tracking.py    MLflow or local JSON
    pipeline.py    analysis and baseline stages, on-disk layout
  app/             FastAPI, SQLite store, job runner, RQ worker
frontend/          React PWA (Vite)
examples/          sample challenge data
```

## Before you rely on it

- Requirements are lower bounds. Lock exact versions on first install (`uv pip compile`, and commit
  `package-lock.json`), and check `npm outdated` since the UI pins were chosen without registry access.
- Newer MLflow 3.x releases may reject requests whose Host header isn't localhost. If the worker can't log runs,
  start the server with `--allowed-hosts "mlflow,mlflow:5000,localhost,localhost:5000"`.
- Group statistics include the row itself in its group's statistics on the training fold. That's standard
  and harmless for feature columns, but don't point group statistics at anything derived from the target.
- Text columns are profiled but excluded from baselines. High-cardinality categoricals go to tree models as
  ordinal codes (CatBoost handles them natively).
- Submission files are a starting point. Check the format each challenge asks for before submitting.
- Check each platform's rules on AI assistance before using this on it.

## Where to take it next

- Stacking: a level-2 model on out-of-fold predictions, fitted with the same honest per-fold scheme as blends.
- Seed averaging: re-run a chosen entry with several seeds to shrink its variance before the final submission.
- Advisor in the endgame: let Claude read the endgame summary and argue for or against the deterministic picks.
