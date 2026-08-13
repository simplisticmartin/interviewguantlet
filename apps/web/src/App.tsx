import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { api, clearSession, getIdentity, getToken } from "./lib/api";
import Companies from "./pages/Companies";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Interview from "./pages/Interview";
import Login from "./pages/Login";
import NewInterview from "./pages/NewInterview";
import Questions from "./pages/Questions";
import Report from "./pages/Report";
import Skills from "./pages/Skills";
import StudyPlanPage from "./pages/StudyPlan";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/practice", label: "Practice" },
  { to: "/history", label: "History" },
  { to: "/skills", label: "Skills" },
  { to: "/study-plan", label: "Study plan" },
  { to: "/questions", label: "Questions" },
  { to: "/companies", label: "Companies" },
];

function Shell({ children }: { children: React.ReactNode }) {
  const identity = getIdentity();
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    staleTime: 120_000,
    retry: false,
  });

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">G</span>
          Gauntlet
        </div>

        <nav className="nav">
          <div className="nav-section">Practice</div>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div style={{ marginBottom: 8 }}>{identity?.displayName ?? "Signed in"}</div>
          {health?.llm_degraded && (
            // The offline engine produces heuristic scores. Saying so on every screen
            // is the difference between a demo and a misleading product.
            <div className="small" style={{ color: "var(--warn)", marginBottom: 8 }}>
              Offline engine — heuristic scoring
            </div>
          )}
          <button
            className="btn-ghost"
            style={{ padding: "4px 8px" }}
            onClick={() => {
              clearSession();
              window.location.href = "/login";
            }}
          >
            Sign out
          </button>
        </div>
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}

function RequireAuth({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  if (!getToken()) {
    return <Navigate to="/login" state={{ from: location.pathname }} replace />;
  }
  return <>{children}</>;
}

export default function App() {
  // Bumped after login so the shell re-reads identity from storage.
  const [, setAuthTick] = useState(0);

  return (
    <Routes>
      <Route path="/login" element={<Login onAuthenticated={() => setAuthTick((n) => n + 1)} />} />

      {/* The live interview is full-bleed: no sidebar, no score, no distractions. */}
      <Route
        path="/interview/:sessionId"
        element={
          <RequireAuth>
            <Interview />
          </RequireAuth>
        }
      />

      <Route
        path="*"
        element={
          <RequireAuth>
            <Shell>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/practice" element={<NewInterview />} />
                <Route path="/history" element={<History />} />
                <Route path="/report/:sessionId" element={<Report />} />
                <Route path="/skills" element={<Skills />} />
                <Route path="/study-plan" element={<StudyPlanPage />} />
                <Route path="/questions" element={<Questions />} />
                <Route path="/companies" element={<Companies />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Shell>
          </RequireAuth>
        }
      />
    </Routes>
  );
}
