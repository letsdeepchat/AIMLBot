import { useEffect, useMemo, useState } from "react";
import { api, readHeader } from "../api.js";

const TASKS = [
  { value: "", label: "Detect from target" },
  { value: "binary", label: "Binary classification" },
  { value: "multiclass", label: "Multiclass classification" },
  { value: "regression", label: "Regression" },
];

function ColumnInput({ id, label, hint, value, onChange, options, required }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} list={`${id}-options`} value={value} onChange={(e) => onChange(e.target.value)}
             required={required} autoComplete="off" spellCheck="false" />
      <datalist id={`${id}-options`}>
        {options.map((c) => <option key={c} value={c} />)}
      </datalist>
      {hint ? <p className="hint">{hint}</p> : null}
    </div>
  );
}

export default function NewChallenge({ onCreated }) {
  const [metrics, setMetrics] = useState([]);
  const [files, setFiles] = useState({ train: null, test: null });
  const [headers, setHeaders] = useState({ train: [], test: [] });
  const [form, setForm] = useState({
    name: "", description: "", target: "", id_col: "", group_col: "", time_col: "",
    task: "", metric: "", submission_limit: "", deadline: "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const set = (key) => (value) => setForm((f) => ({ ...f, [key]: value }));

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => setMetrics([]));
  }, []);

  async function pickFile(kind, file) {
    setFiles((f) => ({ ...f, [kind]: file }));
    const cols = await readHeader(file);
    setHeaders((h) => ({ ...h, [kind]: cols }));
  }

  // Suggest target (in train, not in test) and id (shared, id-like name) once both headers are known.
  useEffect(() => {
    const { train, test } = headers;
    if (!train.length || !test.length) return;
    const testSet = new Set(test);
    const trainOnly = train.filter((c) => !testSet.has(c));
    const idGuess = train.find((c) => testSet.has(c) && /(^id$|_id$|^id_)/i.test(c));
    setForm((f) => ({
      ...f,
      target: f.target || (trainOnly.length === 1 ? trainOnly[0] : ""),
      id_col: f.id_col || idGuess || "",
    }));
  }, [headers]);

  const metricOptions = useMemo(
    () => metrics.filter((m) => !form.task || m.tasks.includes(form.task)),
    [metrics, form.task]
  );

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const body = new FormData();
    Object.entries(form).forEach(([k, v]) => {
      if (String(v).trim() !== "") body.append(k, String(v).trim());
    });
    body.append("train", files.train);
    body.append("test", files.test);
    try {
      onCreated(await api.createChallenge(body));
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const sharedCols = headers.train.filter((c) => headers.test.includes(c));

  return (
    <form className="new-challenge" onSubmit={submit}>
      <h1>New challenge</h1>
      <p className="lede">
        Bring in a challenge you've chosen. The analysis starts as soon as the files upload.
      </p>

      <fieldset>
        <legend>Challenge</legend>
        <div className="field">
          <label htmlFor="name">Name</label>
          <input id="name" value={form.name} onChange={(e) => set("name")(e.target.value)} required />
        </div>
        <div className="field">
          <label htmlFor="description">Problem description</label>
          <textarea id="description" rows={6} value={form.description}
                    onChange={(e) => set("description")(e.target.value)}
                    placeholder="Paste the challenge text, including the metric and any data notes." />
          <p className="hint">Saved with the challenge. The week-3 advisor will read it.</p>
        </div>
        <div className="field-row">
          <div className="field">
            <label htmlFor="submission_limit">Submission limit</label>
            <input id="submission_limit" type="number" min="1" inputMode="numeric" value={form.submission_limit}
                   onChange={(e) => set("submission_limit")(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="deadline">Deadline</label>
            <input id="deadline" type="date" value={form.deadline} onChange={(e) => set("deadline")(e.target.value)} />
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Data</legend>
        <div className="field-row">
          <div className="field">
            <label htmlFor="train">Train file</label>
            <input id="train" type="file" accept=".csv,.tsv,.parquet,.pq" required
                   onChange={(e) => pickFile("train", e.target.files[0])} />
          </div>
          <div className="field">
            <label htmlFor="test">Test file</label>
            <input id="test" type="file" accept=".csv,.tsv,.parquet,.pq" required
                   onChange={(e) => pickFile("test", e.target.files[0])} />
          </div>
        </div>
        <p className="hint">CSV, TSV or Parquet. Column names below are filled from CSV and TSV headers.</p>

        <ColumnInput id="target" label="Target column" value={form.target} onChange={set("target")}
                     options={headers.train} required />
        <ColumnInput id="id_col" label="ID column" hint="Used for the submission file. Excluded from features."
                     value={form.id_col} onChange={set("id_col")} options={sharedCols} />
        <div className="field-row">
          <ColumnInput id="group_col" label="Group column"
                       hint="An entity that repeats across rows, such as a customer or patient."
                       value={form.group_col} onChange={set("group_col")} options={headers.train} />
          <ColumnInput id="time_col" label="Time column"
                       hint="Set it when test comes from a later period."
                       value={form.time_col} onChange={set("time_col")} options={headers.train} />
        </div>
      </fieldset>

      <fieldset>
        <legend>Scoring</legend>
        <div className="field-row">
          <div className="field">
            <label htmlFor="task">Task</label>
            <select id="task" value={form.task} onChange={(e) => setForm((f) => ({ ...f, task: e.target.value, metric: "" }))}>
              {TASKS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="metric">Metric</label>
            <select id="metric" value={form.metric} onChange={(e) => set("metric")(e.target.value)}>
              <option value="">Default for the task</option>
              {metricOptions.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}
            </select>
          </div>
        </div>
      </fieldset>

      {error ? <p className="notice notice--risk" role="alert">{error}</p> : null}
      <div className="actions">
        <button className="button button--primary" type="submit" disabled={busy || !files.train || !files.test}>
          {busy ? "Uploading…" : "Upload and analyse"}
        </button>
        <a className="button" href="#/">Cancel</a>
      </div>
    </form>
  );
}
