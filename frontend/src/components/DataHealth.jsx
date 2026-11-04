import { fmt, pct } from "../api.js";

const SEVERITY_LABEL = { high: "Fix first", medium: "Check", info: "Note" };
const KIND_LABEL = { numeric: "numeric", categorical: "categorical", datetime: "date", text: "text (excluded)" };

function Findings({ findings }) {
  if (!findings.length) {
    return <p className="all-clear">No leakage or integrity problems found.</p>;
  }
  return (
    <ul className="findings">
      {findings.map((f) => (
        <li key={f.code} className={`finding finding--${f.severity}`}>
          <span className="finding__severity">{SEVERITY_LABEL[f.severity]}</span>
          <div>
            <h3>{f.title}</h3>
            <p>{f.detail}</p>
            {f.columns.length ? (
              <p className="columns">{f.columns.map((c) => <code key={c}>{c}</code>)}</p>
            ) : null}
          </div>
        </li>
      ))}
    </ul>
  );
}

function Target({ target }) {
  if (target.classes) {
    return (
      <div>
        <h3>Target</h3>
        <ul className="bars">
          {target.classes.slice(0, 8).map((c) => (
            <li key={c.label}>
              <span className="bars__label">{c.label}</span>
              <span className="bars__track"><span className="bars__fill" style={{ width: pct(c.share) }} /></span>
              <span className="bars__value">{pct(c.share)}</span>
            </li>
          ))}
        </ul>
        <p className="small muted">
          Largest class is {fmt(target.imbalance_ratio, 1)}× the smallest; the rarest has {target.min_class_count} rows.
        </p>
      </div>
    );
  }
  return (
    <div>
      <h3>Target</h3>
      <dl className="facts facts--compact">
        <div><dt>Median</dt><dd>{fmt(target.quantiles["0.5"])}</dd></div>
        <div><dt>Mean</dt><dd>{fmt(target.mean)}</dd></div>
        <div><dt>1st–99th pct</dt><dd>{fmt(target.quantiles["0.01"])} to {fmt(target.quantiles["0.99"])}</dd></div>
        <div><dt>Skew</dt><dd>{fmt(target.skew, 2)}</dd></div>
      </dl>
      {target.suggest_log_target ? (
        <p className="small">The target is right-skewed and non-negative. Training on log1p(target) is worth a try.</p>
      ) : null}
    </div>
  );
}

function Strength({ rows, scale }) {
  if (!rows.length) return null;
  const isAuc = scale.startsWith("AUC");
  const width = (s) => `${Math.max(0, isAuc ? (s - 0.5) / 0.5 : s) * 100}%`;
  return (
    <div>
      <h3>Strongest single columns</h3>
      <p className="small muted">Each column on its own, out of fold. Scale: {scale}.</p>
      <ul className="bars">
        {rows.slice(0, 10).map((r) => (
          <li key={r.column}>
            <span className="bars__label"><code>{r.column}</code></span>
            <span className="bars__track"><span className="bars__fill" style={{ width: width(r.strength) }} /></span>
            <span className="bars__value">{fmt(r.strength, 3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function DataHealth({ analysis }) {
  const { profile, leakage } = analysis;
  const kinds = Object.entries(profile.kind_counts).map(([k, n]) => `${n} ${KIND_LABEL[k] ?? k}`).join(", ");
  return (
    <div className="panel">
      <p>
        {profile.n_features} feature columns: {kinds}.
        {profile.high_missing.length ? ` ${profile.high_missing.length} are more than half empty.` : ""}
      </p>
      <Findings findings={leakage.findings} />
      <div className="split">
        <Target target={profile.target} />
        <Strength rows={leakage.univariate_strength} scale={leakage.strength_scale} />
      </div>
      <details>
        <summary>All columns</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Column</th><th scope="col">Kind</th>
                <th scope="col" className="num">Missing train</th><th scope="col" className="num">Missing test</th>
                <th scope="col" className="num">Unique</th><th scope="col" className="num">Unseen in test</th>
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr key={c.name}>
                  <th scope="row"><code>{c.name}</code></th>
                  <td>{KIND_LABEL[c.kind] ?? c.kind}</td>
                  <td className="num">{pct(c.missing_train)}</td>
                  <td className="num">{pct(c.missing_test)}</td>
                  <td className="num">{c.n_unique.toLocaleString()}</td>
                  <td className="num">{c.unseen_in_test === undefined ? "–" : pct(c.unseen_in_test)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
