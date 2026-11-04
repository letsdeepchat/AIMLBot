import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { go } from "../router.js";
import AdvisorPanel from "./AdvisorPanel.jsx";
import BaselinesPanel from "./BaselinesPanel.jsx";
import ColumnSettings from "./ColumnSettings.jsx";
import EndgamePanel from "./EndgamePanel.jsx";
import ExperimentsPanel from "./ExperimentsPanel.jsx";
import DataHealth from "./DataHealth.jsx";
import ShiftPanel from "./ShiftPanel.jsx";
import ValidationPanel from "./ValidationPanel.jsx";

const TASK_LABEL = { binary: "Binary", multiclass: "Multiclass", regression: "Regression" };
const JOB_LABEL = { analysis: "Analysis", baselines: "Baselines", experiment: "Experiment", advice: "Advisor", blend: "Blend" };

function lastDone(jobs, kind) {
  return jobs.find((j) => j.kind === kind && j.status === "done")?.id ?? null;
}

function Stage({ id, title, state, children }) {
  return (
    <section className={`stage stage--${state}`} aria-labelledby={`${id}-title`}>
      <div className="stage__mark" aria-hidden="true" />
      <div className="stage__body">
        <h2 id={`${id}-title`}>{title}</h2>
        {children}
      </div>
    </section>
  );
}

export default function ChallengeView({ id, onChange }) {
  const [challenge, setChallenge] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [baselines, setBaselines] = useState(null);
  const [error, setError] = useState(null);
  const [draft, setDraft] = useState(null); // a proposal handed from the advisor to the experiment builder
  const takeDraft = useCallback(() => setDraft(null), []);

  const load = useCallback(async () => {
    try {
      setChallenge(await api.challenge(id));
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while a job is running; the list in the sidebar refreshes too.
  const active = challenge?.active_job;
  useEffect(() => {
    if (!active) return;
    const t = setTimeout(() => {
      load();
      onChange();
    }, 1500);
    return () => clearTimeout(t);
  }, [challenge, active, load, onChange]);

  const analysisJob = challenge ? lastDone(challenge.jobs, "analysis") : null;
  const baselinesJob = challenge ? lastDone(challenge.jobs, "baselines") : null;
  const hasAnalysis = challenge?.reports.analysis;
  const hasBaselines = challenge?.reports.baselines;

  useEffect(() => {
    if (hasAnalysis) api.report(id, "analysis").then(setAnalysis).catch((e) => setError(e.message));
  }, [id, hasAnalysis, analysisJob]);

  useEffect(() => {
    if (hasBaselines) api.report(id, "baselines").then(setBaselines).catch((e) => setError(e.message));
  }, [id, hasBaselines, baselinesJob]);

  async function start(kind) {
    try {
      await api.startJob(id, kind);
      await load();
      onChange();
    } catch (e) {
      setError(e.message);
    }
  }

  async function remove() {
    if (!window.confirm(`Delete "${challenge.name}" and all its files and results?`)) return;
    await api.deleteChallenge(id);
    onChange();
    go("/");
  }

  if (!challenge) {
    return error ? <p className="notice notice--risk">{error}</p> : <p className="muted">Loading…</p>;
  }

  // Experiment failures are shown on the experiment itself; the header reports pipeline failures.
  const lastPipelineJob = challenge.jobs.find((j) => j.kind === "analysis" || j.kind === "baselines");
  const lastFailed = lastPipelineJob?.status === "failed" ? lastPipelineJob : null;
  const jobsKey = challenge.jobs.map((j) => `${j.id}:${j.status}`).join(",");
  const hasExperimentResults = challenge.jobs.some((j) => j.kind === "experiment" && j.status === "done");
  const hasAdvice = challenge.jobs.some((j) => j.kind === "advice" && j.status === "done");
  const pipelineBusy = Boolean(active) && (active.kind === "analysis" || active.kind === "baselines");
  const running = (kind) => active?.kind === kind;
  const stateOf = (kind, ready) => (running(kind) ? "running" : ready ? "done" : "pending");
  const spec = challenge.spec;

  return (
    <article className="challenge">
      <a className="back-link" href="#/">All challenges</a>
      <header className="challenge__header">
        <h1>{challenge.name}</h1>
        <dl className="facts">
          <div><dt>Task</dt><dd>{TASK_LABEL[challenge.task]}</dd></div>
          <div><dt>Metric</dt><dd>{analysis?.metric.label ?? spec.metric}</dd></div>
          <div><dt>Rows</dt><dd>{challenge.n_train.toLocaleString()} train, {challenge.n_test.toLocaleString()} test</dd></div>
          {challenge.submission_limit ? <div><dt>Submissions</dt><dd>{challenge.submission_limit}</dd></div> : null}
          {challenge.deadline ? <div><dt>Deadline</dt><dd>{challenge.deadline}</dd></div> : null}
        </dl>
        <div className="actions">
          <button className="button button--primary" onClick={() => start("baselines")}
                  disabled={!hasAnalysis || Boolean(active)}>
            {hasBaselines ? "Re-run baselines" : "Run baselines"}
          </button>
          <button className="button" onClick={() => start("analysis")} disabled={Boolean(active)}>
            Re-run analysis
          </button>
          <button className="button button--quiet" onClick={remove}>Delete</button>
        </div>
        <ColumnSettings challengeId={id} analysis={analysis} disabled={Boolean(active)}
                        onSaved={() => { load(); onChange(); }} />
        {active ? (
          <p className="job-status" role="status">
            <span className="pulse" aria-hidden="true" />
            {JOB_LABEL[active.kind]}: {active.stage || "queued"}
          </p>
        ) : null}
        {lastFailed && !active ? (
          <p className="notice notice--risk" role="alert">
            The {lastFailed.kind} failed: {lastFailed.error}
          </p>
        ) : null}
        {error ? <p className="notice notice--risk">{error}</p> : null}
      </header>

      {challenge.description ? (
        <details className="description">
          <summary>Problem description</summary>
          <p>{challenge.description}</p>
        </details>
      ) : null}

      <Stage id="health" title="Data health" state={stateOf("analysis", analysis)}>
        {analysis ? <DataHealth analysis={analysis} /> : <p className="muted">Appears when the analysis finishes.</p>}
      </Stage>
      <Stage id="shift" title="Train and test shift" state={stateOf("analysis", analysis)}>
        {analysis ? <ShiftPanel shift={analysis.shift} /> : <p className="muted">Appears when the analysis finishes.</p>}
      </Stage>
      <Stage id="validation" title="Validation plan" state={stateOf("analysis", analysis)}>
        {analysis ? <ValidationPanel plan={analysis.cv_plan} /> : <p className="muted">Appears when the analysis finishes.</p>}
      </Stage>
      <Stage id="baselines" title="Baselines" state={stateOf("baselines", baselines)}>
        {baselines ? (
          <BaselinesPanel report={baselines} challengeId={id} />
        ) : (
          <p className="muted">
            {hasAnalysis
              ? "Run the baselines to train every model on the validation plan above."
              : "Available once the analysis has produced a validation plan."}
          </p>
        )}
      </Stage>
      <Stage id="advisor" title="Advisor" state={stateOf("advice", hasAdvice)}>
        {baselines ? (
          <AdvisorPanel challengeId={id} baselines={baselines} refreshKey={jobsKey} busy={pipelineBusy}
                        onEdit={(p) => setDraft({ name: p.title, hypothesis: p.hypothesis, model: p.model, steps: p.steps, code: p.code })}
                        onChanged={() => { load(); onChange(); }} />
        ) : (
          <p className="muted">Available once the baselines have run.</p>
        )}
      </Stage>
      <Stage id="experiments" title="Experiments" state={stateOf("experiment", hasExperimentResults)}>
        {baselines ? (
          <ExperimentsPanel challengeId={id} baselines={baselines} refreshKey={jobsKey} busy={pipelineBusy}
                            externalDraft={draft} onDraftTaken={takeDraft} onChanged={() => { load(); onChange(); }} />
        ) : (
          <p className="muted">Available once the baselines have run. Each experiment is compared against one of them.</p>
        )}
      </Stage>
      <Stage id="endgame" title="Endgame" state={stateOf("blend", hasBaselines)}>
        {baselines ? (
          <EndgamePanel challengeId={id} refreshKey={jobsKey} onChanged={() => { load(); onChange(); }} />
        ) : (
          <p className="muted">Available once the baselines have run.</p>
        )}
      </Stage>
    </article>
  );
}
