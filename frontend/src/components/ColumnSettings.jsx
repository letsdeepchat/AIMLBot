import { useEffect, useState } from "react";
import { api } from "../api.js";

/** Edit the ID, group, time and dropped columns. Saving re-runs the analysis with the new settings. */
export default function ColumnSettings({ challengeId, analysis, disabled, onSaved }) {
  const [open, setOpen] = useState(false);
  const [cols, setCols] = useState(null);
  const [form, setForm] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!open || cols) return;
    api.columns(challengeId).then((c) => {
      setCols(c);
      setForm({ id_col: c.spec.id_col ?? "", group_col: c.spec.group_col ?? "", time_col: c.spec.time_col ?? "",
                drop_cols: c.spec.drop_cols ?? [] });
    }).catch((e) => setError(e.message));
  }, [open, cols, challengeId]);

  // Columns the analysis flagged, so leaks and group candidates are easy to act on.
  const flagged = {};
  for (const f of analysis?.leakage.findings ?? []) {
    for (const c of f.columns) flagged[c] = flagged[c] || f;
  }

  async function save(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.updateSpec(challengeId, form);
      setOpen(false);
      setCols(null);
      onSaved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const target = cols?.spec.target;
  const candidates = (cols?.train ?? []).filter((c) => c !== target);
  const toggleDrop = (c) =>
    setForm((f) => ({ ...f, drop_cols: f.drop_cols.includes(c) ? f.drop_cols.filter((x) => x !== c) : [...f.drop_cols, c] }));

  return (
    <details className="settings" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Column settings</summary>
      {!form ? (
        error ? <p className="notice notice--risk">{error}</p> : <p className="muted small">Loading columns…</p>
      ) : (
        <form onSubmit={save}>
          <div className="field-row">
            {[["id_col", "ID column", cols.test], ["group_col", "Group column", candidates], ["time_col", "Time column", candidates]].map(
              ([key, label, options]) => (
                <div className="field" key={key}>
                  <label htmlFor={`set-${key}`}>{label}</label>
                  <select id={`set-${key}`} value={form[key]} onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}>
                    <option value="">None</option>
                    {options.filter((c) => c !== target).map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )
            )}
          </div>
          <fieldset className="drop-list">
            <legend>Exclude from features</legend>
            <div className="chips">
              {candidates.filter((c) => ![form.id_col, form.group_col, form.time_col].includes(c)).map((c) => (
                <label key={c} className={`chip${form.drop_cols.includes(c) ? " is-on" : ""}${flagged[c] ? ` chip--${flagged[c].severity}` : ""}`}
                       title={flagged[c]?.title}>
                  <input type="checkbox" checked={form.drop_cols.includes(c)} onChange={() => toggleDrop(c)} />
                  {c}
                </label>
              ))}
            </div>
            <p className="hint">Outlined columns were flagged in Data health; hover to see why.</p>
          </fieldset>
          <p className="small muted">
            Saving re-runs the analysis. Baselines and experiments from the old settings are kept but marked stale until
            you run them again.
          </p>
          {error ? <p className="notice notice--risk" role="alert">{error}</p> : null}
          <div className="actions">
            <button className="button button--primary" type="submit" disabled={busy || disabled}>
              {busy ? "Saving…" : "Save and re-run analysis"}
            </button>
          </div>
          {disabled ? <p className="hint">You can save once the running job finishes.</p> : null}
        </form>
      )}
    </details>
  );
}
