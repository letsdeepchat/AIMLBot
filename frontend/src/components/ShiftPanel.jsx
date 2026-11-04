import { fmt, pct } from "../api.js";

const BAND_TEXT = {
  none: "Train and test look alike. Random CV should track the private leaderboard.",
  mild: "A small difference. Worth watching, rarely decisive.",
  moderate: "Train and test differ noticeably. Trust the test-like holdout alongside CV.",
  severe: "A model can easily tell train from test. CV alone will mislead you about the private leaderboard.",
};

function Gauge({ auc, band }) {
  // 0.5 to 1.0 mapped onto the track; zones mark the band thresholds.
  const pos = (v) => `${((Math.min(Math.max(v, 0.5), 1) - 0.5) / 0.5) * 100}%`;
  return (
    <div className={`gauge gauge--${band}`} role="img" aria-label={`Adversarial AUC ${fmt(auc, 3)}, ${band} shift`}>
      <div className="gauge__track">
        <span className="gauge__zone" style={{ left: 0, width: pos(0.55) }} />
        <span className="gauge__zone" style={{ left: pos(0.55), width: "20%" }} />
        <span className="gauge__zone" style={{ left: pos(0.65), width: "30%" }} />
        <span className="gauge__zone" style={{ left: pos(0.8), width: "40%" }} />
        <span className="gauge__marker" style={{ left: pos(auc) }} />
      </div>
      <div className="gauge__scale" aria-hidden="true">
        {[0.5, 0.65, 0.8, 1.0].map((v) => (
          <span key={v} style={{ left: pos(v) }}>{v.toFixed(v === 0.5 || v === 1 ? 1 : 2)}</span>
        ))}
      </div>
    </div>
  );
}

export default function ShiftPanel({ shift }) {
  const adv = shift.adversarial;
  const top = shift.features.slice(0, 10);
  return (
    <div className="panel">
      <div className="headline-metric">
        <span className="headline-metric__value">{fmt(adv.auc, 3)}</span>
        <span className="headline-metric__label">adversarial AUC, {adv.band === "none" ? "no shift" : `${adv.band} shift`}</span>
      </div>
      <Gauge auc={adv.auc} band={adv.band} />
      <p>{BAND_TEXT[adv.band]}</p>
      {adv.train_rows_more_test_like_than_not ? (
        <p className="small muted">
          {adv.train_rows_more_test_like_than_not.toLocaleString()} train rows look more like test than train.
        </p>
      ) : null}
      {shift.time ? (
        <p className="small">
          {shift.time.test_after_train
            ? "Test starts after train ends: a forward-in-time split."
            : `${pct(shift.time.overlap_share, 0)} of test rows fall inside the train period.`}
        </p>
      ) : null}

      <h3>Columns that differ most</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Column</th>
              <th scope="col" className="num" title="How well this column alone separates train from test">Separates</th>
              <th scope="col" className="num" title="Population stability index; above 0.25 is a large shift">PSI</th>
              <th scope="col" className="num" title="Kolmogorov–Smirnov statistic, numeric columns">KS</th>
              <th scope="col" className="num">Unseen levels</th>
            </tr>
          </thead>
          <tbody>
            {top.map((r) => (
              <tr key={r.column} className={shift.drifting_features.includes(r.column) ? "is-flagged" : undefined}>
                <th scope="row"><code>{r.column}</code></th>
                <td className="num">{fmt(r.adversarial_auc, 3)}</td>
                <td className="num">{fmt(r.psi, 3)}</td>
                <td className="num">{r.ks_stat === undefined ? "–" : fmt(r.ks_stat, 3)}</td>
                <td className="num">{r.unseen_in_test === undefined ? "–" : pct(r.unseen_in_test)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Highlighted rows have PSI above 0.25 or separate train from test with AUC above 0.7.</p>
    </div>
  );
}
