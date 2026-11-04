import { useCallback, useEffect, useState } from "react";
import { api, fmt, signed } from "../api.js";
import ExperimentBuilder from "./ExperimentBuilder.jsx";
import ForestPlot from "./ForestPlot.jsx";

const VERDICT = {
  better: { text: "Helps", tone: "better" },
  worse: { text: "Hurts", tone: "worse" },
  no_clear_change: { text: "No clear change", tone: "neutral" },
};

function stepSummary(steps, catalog) {
  return steps.map((s, i) => <span key={i} className="tag">{catalog[s.type]?.label ?? s.type}</span>);
}

function ExperimentItem({ exp, challengeId, catalog, modelLabels, onCopy, onChanged }) {
  const r = exp.result;
  const job = exp.job;
  const running = job && ["queued", "running"].includes(job.status);
  const failed = job?.status === "failed" && !running;
  const vc = r?.vs_control;
  const verdict = vc ? VERDICT[vc.verdict] : null;

  async function act(fn) {
    try {
      await fn();
      onChanged();
    } catch (e) {
      window.alert(e.message);
    }
  }

  return (
    <li className={`experiment${exp.stale ? " is-stale" : ""}`}>
      <div className="experiment__head">
        <h3>{exp.name}</h3>
        {verdict && !exp.stale && !running ? <span className={`verdict verdict--${verdict.tone}`}>{verdict.text}</span> : null}
      </div>
      <p className="small muted">
        {r?.model_label ?? modelLabels[exp.model] ?? exp.model} with {exp.steps.length} step{exp.steps.length === 1 ? "" : "s"}
        {exp.code ? " and feature code" : ""}
      </p>
      {exp.hypothesis ? <p className="experiment__hypothesis">{exp.hypothesis}</p> : null}
      <p className="tags">
        {exp.source?.advice ? <span className="tag tag--advisor">From the advisor</span> : null}
        {exp.code ? <span className="tag tag--code">Feature code</span> : null}
        {stepSummary(exp.steps, catalog)}
      </p>

      {running ? (
        <p className="job-status" role="status"><span className="pulse" aria-hidden="true" />{job.stage || "Queued"}</p>
      ) : null}
      {failed ? <p className="notice notice--risk small">This run failed: {job.error}</p> : null}
      {exp.stale && !running ? (
        <p className="notice notice--caution small">
          Ran with earlier column settings, so it isn't compared with current results. Run it again to update it.
        </p>
      ) : null}

      {r && !running ? (
        <dl className="facts facts--compact">
          <div><dt>CV</dt><dd>{fmt(r.cv_mean)} ± {fmt(r.cv_std)}</dd></div>
          {vc ? (
            <div>
              <dt>Against its baseline</dt>
              <dd>{signed(vc.mean_diff)}{vc.corrected_se ? ` ± ${fmt(2 * vc.corrected_se)}` : ""}</dd>
            </div>
          ) : null}
          {r.holdout_score !== null && r.holdout_score !== undefined ? (
            <div>
              <dt>Test-like holdout</dt>
              <dd>{fmt(r.holdout_score)}{vc?.holdout_diff !== null && vc?.holdout_diff !== undefined ? ` (${signed(vc.holdout_diff)})` : ""}</dd>
            </div>
          ) : null}
          <div><dt>Time</dt><dd>{r.seconds < 60 ? `${r.seconds.toFixed(1)}s` : `${(r.seconds / 60).toFixed(1)}m`}</dd></div>
        </dl>
      ) : null}
      {r?.new_columns?.length && !running ? (
        <p className="small">
          Strongest new columns on their own:{" "}
          {r.new_columns.slice(0, 3).map((c, i) => (
            <span key={c.column}>{i ? ", " : ""}<code>{c.column}</code> {fmt(c.strength, 3)}</span>
          ))}
        </p>
      ) : null}

      <div className="actions actions--tight">
        {r && !running ? <a className="button button--small" href={api.experimentSubmissionUrl(challengeId, exp.id)} download>Download CSV</a> : null}
        <button className="button button--small" type="button" onClick={() => onCopy(exp)}>Copy into new</button>
        {(exp.stale || failed) && !running ? (
          <button className="button button--small" type="button" onClick={() => act(() => api.rerunExperiment(challengeId, exp.id))}>
            Run again
          </button>
        ) : null}
        {!running ? (
          <button className="button button--small button--quiet" type="button"
                  onClick={() => window.confirm(`Delete "${exp.name}"?`) && act(() => api.deleteExperiment(challengeId, exp.id))}>
            Delete
          </button>
        ) : null}
      </div>
    </li>
  );
}

export default function ExperimentsPanel({ challengeId, baselines, refreshKey, busy, onChanged, externalDraft, onDraftTaken }) {
  const [experiments, setExperiments] = useState([]);
  const [board, setBoard] = useState(null);
  const [catalog, setCatalog] = useState({});
  const [draft, setDraft] = useState(null); // null = closed; {} = blank; {...} = copy
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [exps, lb] = await Promise.all([api.experiments(challengeId), api.leaderboard(challengeId)]);
      setExperiments(exps);
      setBoard(lb);
    } catch (e) {
      setError(e.message);
    }
  }, [challengeId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  // A proposal opened from the advisor arrives here as a draft.
  useEffect(() => {
    if (externalDraft) {
      setDraft(externalDraft);
      onDraftTaken();
    }
  }, [externalDraft, onDraftTaken]);

  useEffect(() => {
    api.recipeSteps().then((c) => setCatalog(Object.fromEntries(c.map((m) => [m.type, m])))).catch(() => {});
  }, []);

  const models = (baselines?.models ?? []).filter((m) => m.status === "ok" && m.key !== "dummy");
  const rows = (board?.entries ?? [])
    .filter((e) => e.key !== "dummy")
    .map((e) => ({ ...e, sublabel: e.kind === "experiment" ? e.model_label : "baseline" }));
  const leader = board?.entries?.[0];
  const holdoutLeader = board?.entries?.find((e) => e.id === board.holdout_leader);

  return (
    <div className="panel">
      {board?.baselines_stale ? (
        <p className="notice notice--caution">
          Column settings changed since the baselines ran. Re-run the baselines, then run each experiment again to
          compare them on the new folds.
        </p>
      ) : leader ? (
        <p>
          <strong>{leader.label}</strong>{leader.kind === "experiment" ? ` (${leader.model_label})` : ""} leads at{" "}
          {fmt(leader.cv_mean)} {board.metric_label}.
          {holdoutLeader && holdoutLeader.id !== leader.id
            ? ` On the test-like holdout, ${holdoutLeader.label} leads instead.`
            : ""}
        </p>
      ) : null}

      <ForestPlot
        rows={rows}
        caption="Baselines and experiments on one scale, each paired with the leader over the same folds. Bars are ±2 corrected standard errors."
      />

      {draft ? (
        <ExperimentBuilder
          challengeId={challengeId}
          models={models}
          initial={draft.id ? { ...draft, name: `${draft.name} (copy)`, parent: draft.id } : draft.name ? draft : null}
          key={draft.id ?? draft.name ?? "blank"}
          onCancel={() => setDraft(null)}
          onCreated={() => {
            setDraft(null);
            onChanged();
            load();
          }}
        />
      ) : (
        <div className="actions">
          <button className="button button--primary" type="button" onClick={() => setDraft({})}
                  disabled={busy || board?.baselines_stale || !models.length}>
            New experiment
          </button>
        </div>
      )}

      {error ? <p className="notice notice--risk">{error}</p> : null}
      {experiments.length ? (
        <ul className="experiments">
          {experiments.map((exp) => (
            <ExperimentItem key={exp.id} exp={exp} challengeId={challengeId} catalog={catalog}
                            modelLabels={Object.fromEntries((baselines?.models ?? []).map((m) => [m.key, m.label]))}
                            onCopy={(e) => setDraft(e)}
                            onChanged={() => { onChanged(); load(); }} />
          ))}
        </ul>
      ) : !draft ? (
        <p className="muted">
          No experiments yet. Start with the strongest hint from the analysis: a ratio between related columns, an
          encoding for a high-cardinality category, or group statistics for a repeating entity.
        </p>
      ) : null}
    </div>
  );
}
