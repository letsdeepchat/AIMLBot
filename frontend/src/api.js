async function request(path, options = {}) {
  const res = await fetch(path, options);
  if (res.status === 204) return null;
  const body = res.headers.get("content-type")?.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = typeof body === "object" ? body.detail : body;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  metrics: () => request("/api/metrics"),
  challenges: () => request("/api/challenges"),
  challenge: (id) => request(`/api/challenges/${id}`),
  createChallenge: (form) => request("/api/challenges", { method: "POST", body: form }),
  deleteChallenge: (id) => request(`/api/challenges/${id}`, { method: "DELETE" }),
  startJob: (id, kind) =>
    request(`/api/challenges/${id}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind }),
    }),
  report: (id, name) => request(`/api/challenges/${id}/reports/${name}`),
  submissionUrl: (id, model) => `/api/challenges/${id}/submissions/${model}`,
  recipeSteps: () => request("/api/recipe-steps"),
  columns: (id) => request(`/api/challenges/${id}/columns`),
  updateSpec: (id, body) => json(`/api/challenges/${id}/spec`, "PATCH", body),
  validateRecipe: (id, steps, code) => json(`/api/challenges/${id}/recipes/validate`, "POST", { steps, code: code || null }),
  experiments: (id) => request(`/api/challenges/${id}/experiments`),
  createExperiment: (id, body) => json(`/api/challenges/${id}/experiments`, "POST", body),
  rerunExperiment: (id, expId) => request(`/api/challenges/${id}/experiments/${expId}/run`, { method: "POST" }),
  deleteExperiment: (id, expId) => request(`/api/challenges/${id}/experiments/${expId}`, { method: "DELETE" }),
  experimentSubmissionUrl: (id, expId) => `/api/challenges/${id}/experiments/${expId}/submission`,
  leaderboard: (id) => request(`/api/challenges/${id}/leaderboard`),
  advisor: () => request("/api/advisor"),
  advice: (id) => request(`/api/challenges/${id}/advice`),
  requestAdvice: (id, body) => json(`/api/challenges/${id}/advice`, "POST", body),
  acceptProposal: (id, sid, pid, codeReviewed) =>
    json(`/api/challenges/${id}/advice/${sid}/proposals/${pid}/accept`, "POST", { code_reviewed: codeReviewed }),
  dismissProposal: (id, sid, pid) => json(`/api/challenges/${id}/advice/${sid}/proposals/${pid}/dismiss`, "POST", {}),
  endgame: (id) => request(`/api/challenges/${id}/endgame`),
  endgameSettings: (id, body) => json(`/api/challenges/${id}/endgame/settings`, "PUT", body),
  addSubmission: (id, body) => json(`/api/challenges/${id}/endgame/submissions`, "POST", body),
  deleteSubmission: (id, sid) => request(`/api/challenges/${id}/endgame/submissions/${sid}`, { method: "DELETE" }),
  createBlend: (id, body) => json(`/api/challenges/${id}/endgame/blends`, "POST", body),
  deleteBlend: (id, bid) => request(`/api/challenges/${id}/endgame/blends/${bid}`, { method: "DELETE" }),
};

function json(path, method, body) {
  return request(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export function signed(v, digits = 4) {
  if (v === null || v === undefined) return "–";
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v), digits)}`;
}

export function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString();
  return Number(value).toFixed(digits);
}

export function pct(value, digits = 1) {
  if (value === null || value === undefined) return "–";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Read the header row of a CSV/TSV in the browser so column pickers can be filled before upload. */
export async function readHeader(file) {
  if (!file || !/\.(csv|tsv)$/i.test(file.name)) return [];
  const text = await file.slice(0, 65536).text();
  const line = text.split(/\r?\n/, 1)[0] ?? "";
  const sep = /\.tsv$/i.test(file.name) ? "\t" : ",";
  const cols = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else quoted = !quoted;
    } else if (ch === sep && !quoted) {
      cols.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  cols.push(cur.trim());
  return cols.filter(Boolean);
}
