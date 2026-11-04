import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";

function Proposal({ proposal, challengeId, sessionId, catalog, modelLabels, onEdit, onChanged }) {
  const [reviewed, setReviewed] = useState(false);
  const [codeOpen, setCodeOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const p = proposal;
  const invalid = p.errors.length > 0;
  const done = p.status !== "proposed";

  async function run() {
    setBusy(true);
    setError(null);
    try {
      await api.acceptProposal(challengeId, sessionId, p.id, reviewed);
      onChanged();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function dismiss() {
    await api.dismissProposal(challengeId, sessionId, p.id);
    onChanged();
  }

  return (
    <li className={`proposal proposal--${p.status}${invalid ? " is-invalid" : ""}`}>
      <div className="proposal__head">
        <span className="proposal__rank" aria-label={`Priority ${p.priority}`}>{p.priority}</span>
        <h3>{p.title}</h3>
        {p.status === "accepted" ? <span className="verdict verdict--better">Running as an experiment</span> : null}
        {p.status === "dismissed" ? <span className="verdict verdict--neutral">Dismissed</span> : null}
      </div>
      <p>{p.hypothesis}</p>
      <dl className="proposal__facts">
        <div><dt>Evidence</dt><dd>{p.evidence}</dd></div>
        {p.expected_effect ? <div><dt>Expected</dt><dd>{p.expected_effect}</dd></div> : null}
        {p.risk ? <div><dt>Risk</dt><dd>{p.risk}</dd></div> : null}
      </dl>
      <p className="tags">
        <span className="tag tag--model">{modelLabels[p.model] ?? p.model}</span>
        {p.code ? <span className="tag tag--code">Feature code</span> : null}
        {p.steps.map((s, i) => <span key={i} className="tag">{catalog[s.type]?.label ?? s.type}</span>)}
      </p>

      {p.code ? (
        <details className="code-review" open={codeOpen} onToggle={(e) => setCodeOpen(e.currentTarget.open)}>
          <summary>Read the feature code ({p.code.split("\n").length} lines)</summary>
          <pre><code>{p.code}</code></pre>
        </details>
      ) : null}

      {invalid ? (
        <div className="notice notice--risk small">
          <p>This proposal didn't pass validation, even after Claude's repair attempt:</p>
          <ul>{p.errors.map((e) => <li key={e}>{e}</li>)}</ul>
        </div>
      ) : null}
      {error ? <p className="notice notice--risk small" role="alert">{error}</p> : null}

      {!done ? (
        <div className="actions actions--tight">
          {p.code && !invalid ? (
            <label className="confirm">
              <input type="checkbox" checked={reviewed} disabled={!codeOpen && !reviewed}
                     onChange={(e) => setReviewed(e.target.checked)} />
              I've read the code
            </label>
          ) : null}
          {!invalid ? (
            <button className="button button--primary button--small" type="button" onClick={run}
                    disabled={busy || (p.code && !reviewed)}>
              {busy ? "Starting…" : "Run as experiment"}
            </button>
          ) : null}
          <button className="button button--small" type="button" onClick={() => onEdit(p)}>Edit in builder</button>
          <button className="button button--small button--quiet" type="button" onClick={dismiss}>Dismiss</button>
        </div>
      ) : null}
      {p.code && !codeOpen && !done && !invalid ? <p className="hint">Open the code to confirm you've read it.</p> : null}
    </li>
  );
}

export default function AdvisorPanel({ challengeId, baselines, refreshKey, busy, onEdit, onChanged }) {
  const [status, setStatus] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [catalog, setCatalog] = useState({});
  const [focus, setFocus] = useState("");
  const [count, setCount] = useState(4);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setSessions(await api.advice(challengeId));
    } catch (e) {
      setError(e.message);
    }
  }, [challengeId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  useEffect(() => {
    api.advisor().then(setStatus).catch(() => setStatus({ configured: false }));
    api.recipeSteps().then((c) => setCatalog(Object.fromEntries(c.map((m) => [m.type, m])))).catch(() => {});
  }, []);

  async function ask(e) {
    e.preventDefault();
    setError(null);
    try {
      await api.requestAdvice(challengeId, { focus, n: count });
      setFocus("");
      onChanged();
      load();
    } catch (err) {
      setError(err.message);
    }
  }

  const modelLabels = Object.fromEntries((baselines?.models ?? []).map((m) => [m.key, m.label]));
  const latest = sessions[0];
  const working = latest?.job && ["queued", "running"].includes(latest.job.status);
  const older = sessions.slice(1);

  if (status && !status.configured) {
    return (
      <div className="panel">
        <p>
          The advisor reads this analysis and your experiments, then proposes ranked experiments for you to review. To turn it
          on, set <code>ANTHROPIC_API_KEY</code> in <code>.env</code> and restart the API and worker.
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <p className="small muted">
        Claude sees summaries only (column profiles, findings, scores and your experiment history), never raw rows. Nothing
        runs until you approve it.{status?.model ? ` Model: ${status.model}.` : ""}
      </p>
      <form className="ask" onSubmit={ask}>
        <div className="field">
          <label htmlFor="advice-focus">Anything to focus on?</label>
          <textarea id="advice-focus" rows={2} value={focus} onChange={(e) => setFocus(e.target.value)}
                    placeholder="Optional, e.g. the shift in x1, or ideas for the text column" />
        </div>
        <div className="ask__row">
          <div className="field">
            <label htmlFor="advice-count">Proposals</label>
            <select id="advice-count" value={count} onChange={(e) => setCount(Number(e.target.value))}>
              {[2, 3, 4, 5, 6].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <button className="button button--primary" type="submit" disabled={busy || working}>
            {working ? "Claude is thinking…" : sessions.length ? "Ask for more proposals" : "Ask for proposals"}
          </button>
        </div>
      </form>
      {error ? <p className="notice notice--risk" role="alert">{error}</p> : null}

      {working ? (
        <p className="job-status" role="status"><span className="pulse" aria-hidden="true" />{latest.job.stage || "Queued"}</p>
      ) : null}
      {latest?.job?.status === "failed" ? <p className="notice notice--risk">The advisor failed: {latest.job.error}</p> : null}

      {latest?.reading ? (
        <>
          <blockquote className="reading">
            {latest.focus ? <p className="small muted">You asked about: {latest.focus}</p> : null}
            <p>{latest.reading}</p>
          </blockquote>
          <ol className="proposals">
            {latest.proposals.map((p) => (
              <Proposal key={p.id} proposal={p} challengeId={challengeId} sessionId={latest.id} catalog={catalog}
                        modelLabels={modelLabels} onEdit={onEdit} onChanged={() => { onChanged(); load(); }} />
            ))}
          </ol>
          {latest.usage ? (
            <p className="small muted">
              {latest.usage.calls === 2 ? "Some proposals failed validation, so Claude had one round to revise them. " : ""}
              {(latest.usage.input_tokens + latest.usage.output_tokens).toLocaleString()} tokens.
            </p>
          ) : null}
        </>
      ) : null}

      {older.length ? (
        <details>
          <summary>Earlier advice ({older.length})</summary>
          <ul className="history">
            {older.map((s) => (
              <li key={s.id}>
                {new Date(s.created_at * 1000).toLocaleString()}: {s.proposals.length} proposals,{" "}
                {s.proposals.filter((p) => p.status === "accepted").length} run
                {s.reading ? <p className="small muted">{s.reading}</p> : null}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}
