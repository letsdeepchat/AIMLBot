function statusOf(c) {
  if (c.active_job) return { tone: "running", text: c.active_job.stage || "Queued" };
  const last = c.jobs[0];
  if (last?.status === "failed") return { tone: "risk", text: `${last.kind === "analysis" ? "Analysis" : "Baselines"} failed` };
  if (c.reports.baselines) return { tone: "done", text: "Baselines ready" };
  if (c.reports.analysis) return { tone: "done", text: "Analysis ready" };
  return { tone: "idle", text: "Not analysed" };
}

export default function ChallengeList({ challenges, activeId }) {
  if (!challenges.length) return <p className="muted small">No challenges yet.</p>;
  return (
    <nav className="challenge-list" aria-label="Challenges">
      {challenges.map((c) => {
        const status = statusOf(c);
        return (
          <a key={c.id} href={`#/c/${c.id}`} className={`challenge-link${c.id === activeId ? " is-active" : ""}`}
             aria-current={c.id === activeId ? "page" : undefined}>
            <span className="challenge-link__name">{c.name}</span>
            <span className={`status status--${status.tone}`}>{status.text}</span>
          </a>
        );
      })}
    </nav>
  );
}
