import { signed } from "../api.js";

/**
 * Paired difference from the leader with ±2 corrected standard errors.
 * A bar that reaches the leader line is within fold-to-fold noise.
 * rows: [{ id|key, label, sublabel?, rank, vs_leader: { mean_diff, corrected_se, within_noise } }]
 */
export default function ForestPlot({ rows, caption }) {
  const usable = rows.filter((m) => m.vs_leader);
  if (usable.length < 2) return null;
  const half = (m) => 2 * (m.vs_leader.corrected_se ?? 0);
  const lo = Math.min(...usable.map((m) => m.vs_leader.mean_diff - half(m)));
  const hi = Math.max(0, ...usable.map((m) => m.vs_leader.mean_diff + half(m)));
  const span = hi - lo || 1e-9;
  const pad = span * 0.08;
  const x = (v) => `${((v - (lo - pad)) / (span + 2 * pad)) * 100}%`;

  return (
    <figure className="forest">
      <figcaption>
        {caption ??
          "Each model against the leader, paired over the same folds. Bars are ±2 corrected standard errors; a bar that reaches the leader line is within fold-to-fold noise."}
      </figcaption>
      <div className="forest__rows">
        {usable.map((m) => {
          const { mean_diff: d, within_noise: tied } = m.vs_leader;
          const isLeader = m.rank === 1;
          const h = half(m);
          const state = isLeader ? "leader" : tied ? "tied" : "behind";
          return (
            <div key={m.id ?? m.key} className={`forest__row forest__row--${state}`}>
              <span className="forest__label">
                {m.label}
                {m.sublabel ? <span className="forest__sublabel">{m.sublabel}</span> : null}
              </span>
              <span className="forest__track">
                <span className="forest__zero" style={{ left: x(0) }} aria-hidden="true" />
                {!isLeader && h > 0 ? (
                  <span className="forest__bar" style={{ left: x(d - h), width: `calc(${x(d + h)} - ${x(d - h)})` }} />
                ) : null}
                <span className="forest__point" style={{ left: x(d) }} />
              </span>
              <span className="forest__value">{isLeader ? "Leader" : `${signed(d)}${tied ? ", tied" : ""}`}</span>
            </div>
          );
        })}
      </div>
    </figure>
  );
}
