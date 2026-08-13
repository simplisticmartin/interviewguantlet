import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api, setSession } from "../lib/api";

export default function Login({ onAuthenticated }: { onAuthenticated: () => void }) {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const result =
        mode === "login"
          ? await api.login(email, password)
          : await api.register(email, password, displayName || email.split("@")[0]!);
      setSession(result.access_token, {
        candidateId: result.candidate_id,
        displayName: result.display_name,
      });
      onAuthenticated();
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-shell">
      <form className="auth-card" onSubmit={submit}>
        <div className="brand" style={{ padding: 0, marginBottom: 6 }}>
          <span className="brand-mark">G</span>
          Gauntlet
        </div>
        <p className="muted small" style={{ marginBottom: 22 }}>
          Adaptive technical interview practice. Upload a resume, paste a job description,
          and get evidence-backed feedback.
        </p>

        {mode === "register" && (
          <div className="field">
            <label htmlFor="name">Name</label>
            <input
              id="name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Alex Morgan"
              autoComplete="name"
            />
          </div>
        )}

        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
        </div>

        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            required
            minLength={mode === "register" ? 10 : 1}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "register" ? "new-password" : "current-password"}
          />
          {mode === "register" && <div className="field-hint">At least 10 characters.</div>}
        </div>

        <button type="submit" className="btn-primary" style={{ width: "100%" }} disabled={busy}>
          {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
        </button>

        {error && <div className="error-text">{error}</div>}

        <p className="small muted" style={{ marginTop: 16, marginBottom: 0 }}>
          {mode === "login" ? "No account yet?" : "Already have an account?"}{" "}
          <button
            type="button"
            className="btn-ghost"
            style={{ padding: "2px 6px" }}
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError(null);
            }}
          >
            {mode === "login" ? "Create one" : "Sign in"}
          </button>
        </p>
      </form>
    </div>
  );
}
