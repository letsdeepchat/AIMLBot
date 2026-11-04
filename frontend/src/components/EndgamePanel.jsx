import { useCallback, useEffect, useState } from "react";
import { api, fmt, signed } from "../api.js";

const VERDICT = { better: ["Helps", "better"], worse: ["Hurts", "worse"], no_clear_change: ["No clear change", "neutral"] };

function Settings({ challengeId, settings, onSaved }) {
  const [share, setShare] = useState(settings.public_share ? Math.round(settings.public_share * 100) : "");
  const [picks, setPicks] = useState(settings.n_picks);
  const [error, setError] = useState(null);
  async function save(e) {
    e.preventDefault();
    try {
      await api.endgameSettings(challengeId, { public_share: share === "" ? null : Number(share) / 100, n_picks: Number(picks) });
      setError(null);
      onSaved();
    } catch (err) {
      setError(err.message);
    }
  }
  return (
    <form className="endgame-settings" onSubmit={save}>
      <div className="field">
        <label htmlFor="eg-share">Public share of test</label>
        <div className="input-suffix">
          <input id="eg-share" type="number" min="1" max="99" step="1" value={share} onChange={(e) => setShare(e.target.value)} />
          <span>%</span>
        </div>
      </div>
      <div className="field">
        <label htmlFor="eg-picks">Final picks</label>
        <select id="eg-picks" value={picks} onChange={(e) => setPicks(e.target.value)}>
          {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </div>
      <button className="button" type="submit">Save</button>
      {error ? <p className="notice notice--risk small">{error}</p> : null}
    </form>
  );
}

function Picks({ picks, challengeId }) {
  if (!picks.picks.length) return <p className="muted">{picks.notes[0]}</p>;
  return (
    <div>
      <ol className="picks">
        {picks.picks.map((p, i) => (
          <li key={p.id} className="pick">
            <span className="pick__n">{i + 1}</span>
            <div>
              <h3>{p.label}{p.sublabel ? <span className="pick__sub">{p.sublabel}</span> : null}</h3>
              <p className="pick__score">
                {fmt(p.cv_mean)} ± {fmt(2 * (p.cv_se ?? 0))}
                {p.holdout !== null && p.holdout !== undefined ? <span className="muted">, holdout {fmt(p.holdout)}</span> : null}
              </p>
              {p.reasons.map((r) => <p key={r}>{r}</p>)}
              <a className="button button--small" href={`/api/challenges/${challengeId}${p.submission}`} download>Download CSV</a>
            </div>
          </li>
        ))}
      </ol>
      {picks.notes.map((n) => <p key={n} className="notice notice--caution">{n}</p>)}
    </div>
  );
}

function Noise({ noise, metricLabel }) {
  if (!noise) return <p className="muted">Enter the public share of the test set above to see how much of the public board is luck.</p>;
  return (
    <div>
      <p>
        The public board scores about <strong>{noise.n_public.toLocaleString()}</strong> rows, the private board about{" "}
        {noise.n_private.toLocaleString()}. From luck alone, a public {metricLabel} moves by about ±{fmt(2 * noise.public_sd)} and
        a private one by ±{fmt(2 * noise.private_sd)}.
      </p>
      {noise.paired_public_sd ? (
        <p>
          Between your two best ({noise.pair.join(" and ")}), a public gap smaller than <strong>±{fmt(2 * noise.paired_public_sd)}</strong>{" "}
          is indistinguishable from noise.
        </p>
      ) : null}
      <p className="small muted">Bootstrapped from out-of-fold predictions on {noise.rows_in_oof.toLocaleString()} train rows; ranges are ±2 SD.</p>
    </div>
  );
}

function Scatter({ rows, sd }) {
  const pts = rows.filter((r) => r.cv !== null && r.cv !== undefined);
  if (pts.length < 2) return null;
  const xs = pts.map((p) => p.cv), ys = pts.map((p) => p.public_score);
  const pad = (a, b) => (b - a || Math.abs(a) * 0.01 || 0.01) * 0.15;
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys) - 2 * (sd ?? 0), Math.max(...ys) + 2 * (sd ?? 0)];
  const X = (v) => 40 + ((v - (x0 - pad(x0, x1))) / (x1 - x0 + 2 * pad(x0, x1))) * 340;
  const Y = (v) => 190 - ((v - (y0 - pad(y0, y1))) / (y1 - y0 + 2 * pad(y0, y1))) * 170;
  return (
    <figure className="scatter">
      <svg viewBox="-40 0 440 220" role="img" aria-label="Public score against CV score for each submission">
        <line x1="40" y1="190" x2="390" y2="190" className="scatter__axis" />
        <line x1="40" y1="10" x2="40" y2="190" className="scatter__axis" />
        <text x="215" y="214" className="scatter__label" textAnchor="middle">CV</text>
        <text x={X(x0)} y="204" className="scatter__tick" textAnchor="middle">{fmt(x0)}</text>
        <text x={X(x1)} y="204" className="scatter__tick" textAnchor="middle">{fmt(x1)}</text>
        <text x="36" y={Y(Math.min(...ys)) + 3} className="scatter__tick" textAnchor="end">{fmt(Math.min(...ys))}</text>
        <text x="36" y={Y(Math.max(...ys)) + 3} className="scatter__tick" textAnchor="end">{fmt(Math.max(...ys))}</text>
        <text x="-28" y="100" className="scatter__label" textAnchor="middle" transform="rotate(-90 -28 100)">Public</text>
        {pts.map((p) => (
          <g key={p.id} className={`scatter__pt${p.flag ? ` is-${p.flag}` : ""}`}>
            {sd ? <line x1={X(p.cv)} x2={X(p.cv)} y1={Y(p.public_score - 2 * sd)} y2={Y(p.public_score + 2 * sd)} /> : null}
            <circle cx={X(p.cv)} cy={Y(p.public_score)} r="4.5"><title>{`${p.label}: CV ${fmt(p.cv)}, public ${fmt(p.public_score)}`}</title></circle>
          </g>
        ))}
      </svg>
      <figcaption className="small muted">Each submission's public score against its CV. Bars are ±2 SD of public-split luck.</figcaption>
    </figure>
  );
}

function Submissions({ data, challengeId, onChanged }) {
  const [entry, setEntry] = useState(data.candidates[0]?.id ?? "");
  const [score, setScore] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState(null);
  const t = data.tracking;
  const used = t.rows.length;

  async function add(e) {
    e.preventDefault();
    try {
      await api.addSubmission(challengeId, { entry_id: entry, public_score: Number(score), note });
      setScore("");
      setNote("");
      setError(null);
      onChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div>
      <p>
        {data.submission_limit ? <><strong>{used} of {data.submission_limit}</strong> submissions used. </> : `${used} logged. `}
        {t.spearman !== null && t.spearman !== undefined ? `CV and public agree on ordering with Spearman ${fmt(t.spearman, 2)}.` : ""}
      </p>
      {data.submission_limit && used >= data.submission_limit ? (
        <p className="notice notice--caution">
          The submission budget is used up. Choose your final picks from what's already scored; the picks above are based on CV,
          which no further public score could overrule.
        </p>
      ) : null}
      <form className="sub-form" onSubmit={add}>
        <div className="field">
          <label htmlFor="sub-entry">Submitted</label>
          <select id="sub-entry" value={entry} onChange={(e) => setEntry(e.target.value)}>
            {data.candidates.map((c) => <option key={c.id} value={c.id}>{c.label}{c.sublabel ? ` (${c.sublabel})` : ""}</option>)}
          </select>
        </div>
        <div className="field field--narrow">
          <label htmlFor="sub-score">Public score</label>
          <input id="sub-score" type="number" step="any" required value={score} onChange={(e) => setScore(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="sub-note">Note</label>
          <input id="sub-note" value={note} onChange={(e) => setNote(e.target.value)} />
        </div>
        <button className="button" type="submit">Log submission</button>
      </form>
      {error ? <p className="notice notice--risk small">{error}</p> : null}
      {t.rows.length ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th scope="col">Entry</th><th scope="col" className="num">CV</th><th scope="col" className="num">Public</th>
              <th scope="col" className="num">Public − CV</th><th scope="col">Signal</th><th scope="col">Note</th><th scope="col"><span className="visually-hidden">Remove</span></th></tr></thead>
            <tbody>
              {t.rows.map((r) => (
                <tr key={r.id}>
                  <th scope="row">{r.label}{r.current ? "" : <span className="muted"> (stale)</span>}</th>
                  <td className="num">{fmt(r.cv)}</td>
                  <td className="num">{fmt(r.public_score)}</td>
                  <td className="num">{signed(r.gap)}</td>
                  <td>{r.flag === "lucky" ? <span className="flag flag--lucky">Lucky public split</span> : r.flag === "unlucky" ? <span className="flag flag--unlucky">Unlucky public split</span> : "–"}</td>
                  <td>{r.note || "–"}</td>
                  <td><button className="button button--small button--quiet" type="button"
                              onClick={() => api.deleteSubmission(challengeId, r.id).then(onChanged)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <Scatter rows={t.rows} sd={data.noise?.public_sd} />
    </div>
  );
}

function Blends({ data, challengeId, onChanged }) {
  const singles = data.candidates.filter((c) => c.kind !== "blend");
  const [members, setMembers] = useState(singles.slice(0, 2).map((c) => c.id));
  const [method, setMethod] = useState("greedy");
  const [name, setName] = useState("");
  const [error, setError] = useState(null);
  const toggle = (id) => setMembers((m) => (m.includes(id) ? m.filter((x) => x !== id) : [...m, id]));

  async function run(e) {
    e.preventDefault();
    try {
      await api.createBlend(challengeId, { name: name || `Blend of ${members.length}`, members, method });
      setError(null);
      onChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div>
      <p className="small muted">
        Weights are fitted on the other folds and scored on each held-out fold, then compared with the blend's best member
        on identical folds.
      </p>
      <form onSubmit={run}>
        <div className="chips" role="group" aria-label="Entries to blend">
          {singles.map((c) => (
            <label key={c.id} className={`chip${members.includes(c.id) ? " is-on" : ""}`}>
              <input type="checkbox" checked={members.includes(c.id)} onChange={() => toggle(c.id)} />
              {c.label}
            </label>
          ))}
        </div>
        <div className="sub-form">
          <div className="field">
            <label htmlFor="blend-method">Method</label>
            <select id="blend-method" value={method} onChange={(e) => setMethod(e.target.value)}>
              {Object.entries(data.blend_methods).map(([k, v]) => (
                <option key={k} value={k} disabled={k === "rank_mean" && !data.rank_ok}>{v}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="blend-name">Name</label>
            <input id="blend-name" value={name} placeholder={`Blend of ${members.length}`} onChange={(e) => setName(e.target.value)} />
          </div>
          <button className="button button--primary" type="submit" disabled={members.length < 2}>Run blend</button>
        </div>
      </form>
      {error ? <p className="notice notice--risk small">{error}</p> : null}
      <ul className="blends">
        {data.blends.map((b) => {
          const r = b.result;
          const running = b.job && ["queued", "running"].includes(b.job.status);
          const v = r?.vs_best_member;
          return (
            <li key={b.id} className="blend">
              <div className="experiment__head">
                <h3>{b.name}</h3>
                {v && !running ? <span className={`verdict verdict--${VERDICT[v.verdict][1]}`}>{VERDICT[v.verdict][0]}</span> : null}
              </div>
              {running ? <p className="job-status"><span className="pulse" aria-hidden="true" />{b.job.stage || "Queued"}</p> : null}
              {b.job?.status === "failed" ? <p className="notice notice--risk small">This blend failed: {b.job.error}</p> : null}
              {r && !running ? (
                <>
                  <p className="small">
                    {fmt(r.cv_mean)} ± {fmt(2 * (r.cv_se ?? 0))} against its best member, {r.best_member.label} ({fmt(r.best_member.cv_mean)}):{" "}
                    {signed(v.mean_diff)}{v.corrected_se ? ` ± ${fmt(2 * v.corrected_se)}` : ""}.
                  </p>
                  <ul className="bars">
                    {r.weights.map((w) => (
                      <li key={w.id}>
                        <span className="bars__label">{w.label}</span>
                        <span className="bars__track"><span className="bars__fill" style={{ width: `${w.weight * 100}%` }} /></span>
                        <span className="bars__value">{fmt(w.weight, 2)}</span>
                      </li>
                    ))}
                  </ul>
                  {r.note ? <p className="small muted">{r.note}</p> : null}
                </>
              ) : null}
              <div className="actions actions--tight">
                {r && !running ? <a className="button button--small" href={`/api/challenges/${challengeId}/endgame/blends/${b.id}/submission`} download>Download CSV</a> : null}
                {!running ? <button className="button button--small button--quiet" type="button"
                                    onClick={() => api.deleteBlend(challengeId, b.id).then(onChanged)}>Delete</button> : null}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Correlations({ corr }) {
  if (corr.labels.length < 2) return null;
  return (
    <div className="table-wrap">
      <table className="corr">
        <thead><tr><th scope="col"><span className="visually-hidden">Entry</span></th>
          {corr.labels.map((l, j) => <th key={j} scope="col" className="num" title={l}>{j + 1}</th>)}</tr></thead>
        <tbody>
          {corr.matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row">{i + 1}. {corr.labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="num corr__cell" style={{ "--c": i === j ? 0 : Math.max(0, (v - 0.5) / 0.5) }}>{i === j ? "–" : fmt(v, 2)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">Correlation of out-of-fold predictions. Blends gain most from strong entries that disagree.</p>
    </div>
  );
}

export default function EndgamePanel({ challengeId, refreshKey, onChanged }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(async () => {
    try {
      setData(await api.endgame(challengeId));
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [challengeId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  const changed = () => {
    load();
    onChanged();
  };

  if (error) return <p className="notice notice--risk">{error}</p>;
  if (!data) return <p className="muted">Scoring every candidate on the same folds…</p>;
  if (!data.ready) {
    return <p className="muted">{data.stale ? "Column settings changed. Re-run the baselines to use the endgame tools." : "Available once the baselines have run."}</p>;
  }
  return (
    <div className="panel endgame">
      <Settings challengeId={challengeId} settings={data.settings} onSaved={changed} key={JSON.stringify(data.settings)} />
      <h3 className="endgame__h">Final picks</h3>
      <Picks picks={data.picks} challengeId={challengeId} />
      <h3 className="endgame__h">How noisy the public board is</h3>
      <Noise noise={data.noise} metricLabel={data.metric_label} />
      <h3 className="endgame__h">Submissions</h3>
      <Submissions data={data} challengeId={challengeId} onChanged={changed} />
      <h3 className="endgame__h">Blends</h3>
      <Blends data={data} challengeId={challengeId} onChanged={changed} />
      <h3 className="endgame__h">How similar the candidates are</h3>
      <Correlations corr={data.correlations} />
    </div>
  );
}
