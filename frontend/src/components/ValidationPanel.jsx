const STRATEGY_LABEL = {
  repeated_stratified_kfold: "Repeated stratified K‑fold",
  repeated_kfold: "Repeated K‑fold",
  stratified_group_kfold: "Stratified group K‑fold",
  group_kfold: "Group K‑fold",
  time_series: "Forward-in-time splits",
};

export default function ValidationPanel({ plan }) {
  const fits = plan.n_splits * plan.n_repeats;
  return (
    <div className="panel">
      <div className="headline-metric">
        <span className="headline-metric__value headline-metric__value--text">{STRATEGY_LABEL[plan.strategy] ?? plan.strategy}</span>
        <span className="headline-metric__label">
          {plan.n_splits} folds{plan.n_repeats > 1 ? ` × ${plan.n_repeats} repeats` : ""}, {fits} fits per model
          {plan.adversarial_holdout ? `, plus a ${Math.round(plan.holdout_share * 100)}% test-like holdout` : ""}
        </span>
      </div>
      {plan.reasons.map((r) => <p key={r}>{r}</p>)}
      {plan.warnings.map((w) => <p key={w} className="notice notice--caution">{w}</p>)}
    </div>
  );
}
