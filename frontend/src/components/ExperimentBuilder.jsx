import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";

function defaultsFor(meta) {
  const params = {};
  for (const p of meta.params) {
    if (p.default !== undefined) params[p.name] = p.default;
    else if (p.kind === "columns" || p.kind === "options") params[p.name] = [];
    else params[p.name] = "";
  }
  return params;
}

function ColumnChips({ name, options, value, onChange }) {
  const selected = new Set(value || []);
  const toggle = (c) => {
    const next = new Set(selected);
    next.has(c) ? next.delete(c) : next.add(c);
    onChange(options.filter((o) => next.has(o)));
  };
  if (!options.length) return <p className="hint">No columns of the right kind exist at this step.</p>;
  return (
    <div className="chips" role="group" aria-label={name}>
      {options.map((c) => (
        <label key={c} className={`chip${selected.has(c) ? " is-on" : ""}`}>
          <input type="checkbox" checked={selected.has(c)} onChange={() => toggle(c)} />
          {c}
        </label>
      ))}
      {options.length > 3 ? (
        <button type="button" className="chip chip--action"
                onClick={() => onChange(selected.size === options.length ? [] : options)}>
          {selected.size === options.length ? "Clear" : "Select all"}
        </button>
      ) : null}
    </div>
  );
}

function StepEditor({ index, step, meta, columns, errors, onChange, onMove, onRemove, isFirst, isLast }) {
  const set = (name, value) => onChange({ ...step, params: { ...step.params, [name]: value } });
  const options = (accepts) => columns.filter((c) => accepts.includes(c.kind)).map((c) => c.name);
  const numericCols = options(["numeric", "datetime"]);

  return (
    <li className={`step${errors.length ? " has-error" : ""}`}>
      <div className="step__head">
        <h4><span className="step__index">{index + 1}</span>{meta.label}</h4>
        <div className="step__tools">
          <button type="button" className="icon-button" onClick={() => onMove(-1)} disabled={isFirst} aria-label="Move step up">↑</button>
          <button type="button" className="icon-button" onClick={() => onMove(1)} disabled={isLast} aria-label="Move step down">↓</button>
          <button type="button" className="icon-button" onClick={onRemove} aria-label={`Remove step ${index + 1}`}>×</button>
        </div>
      </div>
      <p className="hint">{meta.help}</p>
      {meta.params.map((p) => {
        const id = `step-${index}-${p.name}`;
        const value = step.params[p.name];
        return (
          <div key={p.name} className="field">
            <label htmlFor={id}>{p.label}</label>
            {p.kind === "text" ? (
              <input id={id} value={value ?? ""} onChange={(e) => set(p.name, e.target.value)} />
            ) : p.kind === "expression" ? (
              <>
                <input id={id} className="code-input" value={value ?? ""} spellCheck="false" autoComplete="off"
                       placeholder="log1p(a) / (b + 1)" onChange={(e) => set(p.name, e.target.value)} />
                <p className="hint">Numeric columns here: {numericCols.slice(0, 12).join(", ")}{numericCols.length > 12 ? "…" : ""}</p>
              </>
            ) : p.kind === "number" ? (
              <input id={id} type="number" step="any" min={p.min} max={p.max} value={value ?? ""}
                     onChange={(e) => set(p.name, e.target.value === "" ? "" : Number(e.target.value))} />
            ) : p.kind === "column" ? (
              <select id={id} value={value ?? ""} onChange={(e) => set(p.name, e.target.value)}>
                <option value="">Choose a column</option>
                {options(p.accepts).map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : p.kind === "columns" ? (
              <ColumnChips name={p.label} options={options(p.accepts)} value={value} onChange={(v) => set(p.name, v)} />
            ) : p.kind === "options" ? (
              <ColumnChips name={p.label} options={p.options} value={value} onChange={(v) => set(p.name, v)} />
            ) : null}
          </div>
        );
      })}
      {errors.map((e) => <p key={e} className="step__error" role="alert">{e}</p>)}
    </li>
  );
}

export default function ExperimentBuilder({ challengeId, models, initial, onCreated, onCancel }) {
  const [catalog, setCatalog] = useState([]);
  const [name, setName] = useState(initial?.name ?? "");
  const [hypothesis, setHypothesis] = useState(initial?.hypothesis ?? "");
  const [model, setModel] = useState(initial?.model ?? models[0]?.key ?? "");
  const [steps, setSteps] = useState(initial?.steps ?? []);
  const [code, setCode] = useState(initial?.code ?? "");
  const [check, setCheck] = useState({ errors: [], columns_before: [], output: [], code_columns: [] });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const seq = useRef(0);
  const formRef = useRef(null);

  useEffect(() => {
    formRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, []);

  useEffect(() => {
    api.recipeSteps().then(setCatalog).catch((e) => setError(e.message));
  }, []);

  // Validate on the server after edits settle; it knows which columns exist at each step.
  // Code runs in the sandbox on a sample, so give typing a longer pause before checking it.
  useEffect(() => {
    const ticket = ++seq.current;
    const t = setTimeout(() => {
      api.validateRecipe(challengeId, steps, code)
        .then((res) => ticket === seq.current && setCheck(res))
        .catch((e) => ticket === seq.current && setError(e.message));
    }, code.trim() ? 900 : 350);
    return () => clearTimeout(t);
  }, [challengeId, steps, code]);

  const metaOf = useMemo(() => Object.fromEntries(catalog.map((m) => [m.type, m])), [catalog]);
  const errorsAt = (i) => check.errors.filter((e) => e.step === i).map((e) => e.message);
  const columnsAt = (i) => check.columns_before[i] ?? check.output;

  function addStep(type) {
    if (!type) return;
    setSteps((s) => [...s, { type, params: defaultsFor(metaOf[type]) }]);
  }

  function move(i, delta) {
    setSteps((s) => {
      const next = [...s];
      [next[i], next[i + delta]] = [next[i + delta], next[i]];
      return next;
    });
  }

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onCreated(await api.createExperiment(challengeId, { name, hypothesis, model, steps, code: code.trim() || null, parent: initial?.parent ?? null }));
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const codeErrors = check.errors.filter((e) => e.step < 0).map((e) => e.message);
  const blocked = busy || (!steps.length && !code.trim()) || check.errors.length > 0 || !name.trim();
  const newCols = check.output.length;

  return (
    <form className="builder" onSubmit={submit} ref={formRef}>
      <h3>{initial?.parent ? "New experiment from a copy" : initial?.name ? "New experiment from a proposal" : "New experiment"}</h3>
      <div className="field-row">
        <div className="field">
          <label htmlFor="exp-name">Name</label>
          <input id="exp-name" value={name} onChange={(e) => setName(e.target.value)} required />
        </div>
        <div className="field">
          <label htmlFor="exp-model">Model</label>
          <select id="exp-model" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
          <p className="hint">Compared against this model's baseline, fold by fold.</p>
        </div>
      </div>
      <div className="field">
        <label htmlFor="exp-hypothesis">Hypothesis</label>
        <textarea id="exp-hypothesis" rows={2} value={hypothesis} onChange={(e) => setHypothesis(e.target.value)}
                  placeholder="What you expect these features to capture, and why." />
      </div>

      <details className="code-section" open={Boolean(code)}>
        <summary>Feature code{code.trim() ? "" : " (optional)"}</summary>
        <p className="hint">
          For ideas the steps can't express, such as parsing text. Define <code>features(df)</code> returning a DataFrame of new
          columns with df's index. Work row by row: the sandbox rejects code whose output changes when rows are batched
          differently. Steps below can use the new columns.
        </p>
        <textarea className="code-input" rows={Math.min(18, Math.max(8, code.split("\n").length + 1))} value={code}
                  spellCheck="false" aria-label="Feature code" onChange={(e) => setCode(e.target.value)}
                  placeholder={"def features(df):\n    out = pd.DataFrame(index=df.index)\n    out[\"words\"] = df[\"note\"].str.split().str.len()\n    return out"} />
        {codeErrors.map((e) => <pre key={e} className="step__error code-error">{e}</pre>)}
        {check.code_columns?.length ? (
          <p className="small">Creates: {check.code_columns.map((c) => <code key={c.name}>{c.name}</code>)}</p>
        ) : null}
      </details>

      <ol className="steps">
        {steps.map((step, i) =>
          metaOf[step.type] ? (
            <StepEditor key={i} index={i} step={step} meta={metaOf[step.type]} columns={columnsAt(i)}
                        errors={errorsAt(i)} isFirst={i === 0} isLast={i === steps.length - 1}
                        onChange={(next) => setSteps((s) => s.map((x, j) => (j === i ? next : x)))}
                        onMove={(d) => move(i, d)} onRemove={() => setSteps((s) => s.filter((_, j) => j !== i))} />
          ) : null
        )}
      </ol>

      <div className="field add-step">
        <label htmlFor="add-step">Add a step</label>
        <select id="add-step" value="" onChange={(e) => addStep(e.target.value)}>
          <option value="">Choose a feature step</option>
          {catalog.map((m) => <option key={m.type} value={m.type}>{m.label}</option>)}
        </select>
      </div>

      {steps.length ? (
        <p className="small muted">
          The model will see {newCols} columns. Steps that learn from data learn inside each training fold only.
        </p>
      ) : null}
      {error ? <p className="notice notice--risk" role="alert">{error}</p> : null}
      <div className="actions">
        <button className="button button--primary" type="submit" disabled={blocked}>
          {busy ? "Starting…" : "Run experiment"}
        </button>
        <button className="button" type="button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}
