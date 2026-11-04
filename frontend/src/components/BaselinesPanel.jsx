import { api, fmt } from "../api.js";
import ForestPlot from "./ForestPlot.jsx";

export default function BaselinesPanel({ report, challengeId }) {
  const ok = report.models.filter((m) => m.status === "ok");
  const failed = report.models.filter((m) => m.status !== "ok");
  const leader = ok[0];
  const tied = ok.filter((m) => m.rank > 1 && m.vs_leader?.within_noise && m.key !== "dummy");
  const hasHoldout = ok.some((m) => m.holdout_score !== null && m.holdout_score !== undefined);
  const better = (a, b) => (report.greater_is_better ? a > b : a < b);
  const holdoutLeader = hasHoldout
    ? ok.filter((m) => m.holdout_score !== null).reduce((best, m) => (better(m.holdout_score, best.holdout_score) ? m : best))
    : null;

  return (
    <div className="panel">
      {leader ? (
        <p>
          <strong>{leader.label}</strong> leads on {report.metric_label} at {fmt(leader.cv_mean)} ± {fmt(leader.cv_std)} over{" "}
          {report.n_folds} folds
          {tied.length
            ? `. ${tied.map((m) => m.label).join(", ")} ${tied.length === 1 ? "is" : "are"} statistically tied with it.`
            : ", clearly ahead of the rest."}
        </p>
      ) : null}
      {holdoutLeader && leader && holdoutLeader.key !== leader.key ? (
        <p className="notice notice--caution">
          On the test-like holdout, {holdoutLeader.label} leads instead ({fmt(holdoutLeader.holdout_score)}). With shifted
          test data, that ranking is the better guide for the private leaderboard.
        </p>
      ) : null}

      <ForestPlot rows={report.models.filter((m) => m.status === "ok" && m.key !== "dummy")} />

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col" className="num">#</th>
              <th scope="col">Model</th>
              <th scope="col" className="num">CV mean</th>
              <th scope="col" className="num">± std</th>
              <th scope="col" className="num">Out-of-fold</th>
              {hasHoldout ? <th scope="col" className="num">Test-like holdout</th> : null}
              <th scope="col" className="num">Time</th>
              <th scope="col"><span className="visually-hidden">Submission</span></th>
            </tr>
          </thead>
          <tbody>
            {ok.map((m) => (
              <tr key={m.key} className={m.rank === 1 ? "is-leader" : undefined}>
                <td className="num">{m.rank}</td>
                <th scope="row">{m.label}</th>
                <td className="num">{fmt(m.cv_mean)}</td>
                <td className="num">{fmt(m.cv_std)}</td>
                <td className="num">{fmt(m.oof_score)}</td>
                {hasHoldout ? <td className="num">{fmt(m.holdout_score)}</td> : null}
                <td className="num">{m.seconds < 60 ? `${m.seconds.toFixed(1)}s` : `${(m.seconds / 60).toFixed(1)}m`}</td>
                <td>
                  {report.submissions.includes(m.key) ? (
                    <a href={api.submissionUrl(challengeId, m.key)} download>Download CSV</a>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {failed.map((m) => (
        <p key={m.key} className="notice notice--risk small">{m.label} failed: {m.error}</p>
      ))}
      <p className="small muted">
        {report.positive_class !== null && report.positive_class !== undefined
          ? `Binary submissions hold the probability of class “${report.positive_class}”. `
          : ""}
        Submission files are a starting point: check the column names and format the challenge asks for before you submit.
      </p>
    </div>
  );
}
