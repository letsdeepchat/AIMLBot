import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import ChallengeList from "./components/ChallengeList.jsx";
import ChallengeView from "./components/ChallengeView.jsx";
import NewChallenge from "./components/NewChallenge.jsx";
import { go } from "./router.js";

// Hash routes: #/ (list), #/new, #/c/<id>
function parseRoute() {
  const hash = window.location.hash.replace(/^#/, "") || "/";
  const match = hash.match(/^\/c\/([a-z0-9]+)/);
  if (match) return { page: "challenge", id: match[1] };
  if (hash.startsWith("/new")) return { page: "new" };
  return { page: "home" };
}

export default function App() {
  const [route, setRoute] = useState(parseRoute);
  const [challenges, setChallenges] = useState([]);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setChallenges(await api.challenges());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    const onHash = () => setRoute(parseRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, route]);

  return (
    <div className={`shell shell--${route.page}`}>
      <aside className="sidebar">
        <a className="brand" href="#/">Quest Copilot</a>
        <a className="button button--primary" href="#/new">New challenge</a>
        {error ? <p className="notice notice--risk">Can't reach the API: {error}</p> : null}
        <ChallengeList challenges={challenges} activeId={route.id} />
      </aside>
      <main className="main">
        {route.page === "new" ? (
          <NewChallenge
            onCreated={(c) => {
              refresh();
              go(`/c/${c.id}`);
            }}
          />
        ) : route.page === "challenge" ? (
          <ChallengeView key={route.id} id={route.id} onChange={refresh} />
        ) : (
          <div className="empty">
            <h1>Start with a challenge</h1>
            <p>
              Upload a challenge's train and test files. The analysis runs straight away: data health, leakage, train/test
              shift and a validation plan. Baselines run when you're ready.
            </p>
            <a className="button button--primary" href="#/new">New challenge</a>
          </div>
        )}
      </main>
    </div>
  );
}
