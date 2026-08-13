/**
 * The live interview screen.
 *
 * Deliberately hostile to distraction: no sidebar, no running score, no rubric. In Real
 * Interview Mode the candidate must not be able to infer how they are doing, because
 * that changes how they answer the next question (spec section 34).
 */
import Editor from "@monaco-editor/react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { formatDuration } from "../components/ui";
import { ApiError, answerStreaming, api, type QuestionView, type TurnResponse } from "../lib/api";

interface Turn {
  role: "interviewer" | "candidate" | "clarification" | "coaching";
  text: string;
  ordinal?: number | null;
  isFollowup?: boolean;
}

const LANGUAGES = ["java", "python", "javascript", "typescript"];

export default function Interview() {
  const { sessionId = "" } = useParams();
  const navigate = useNavigate();

  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState<QuestionView | null>(null);
  const [draft, setDraft] = useState("");
  const [code, setCode] = useState("");
  const [language, setLanguage] = useState("java");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [remaining, setRemaining] = useState(0);
  const [stage, setStage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);
  const [showEditor, setShowEditor] = useState(false);

  const transcriptRef = useRef<HTMLDivElement>(null);
  const loadedRef = useRef(false);

  // Resume an in-flight interview (refresh, closed tab, new device).
  useEffect(() => {
    if (!sessionId || loadedRef.current) return;
    loadedRef.current = true;

    api
      .getInterview(sessionId)
      .then((detail) => {
        const restored: Turn[] = [];
        for (const entry of detail.transcript) {
          restored.push({
            role: "interviewer",
            text: entry.prompt_text,
            ordinal: entry.ordinal,
            isFollowup: entry.is_followup,
          });
          if (entry.answer_text) {
            restored.push({ role: "candidate", text: entry.answer_text });
          }
        }
        setTurns(restored);

        if (detail.status === "completed") {
          setFinished(true);
          return;
        }
        const last = detail.transcript.at(-1);
        if (last && !last.answer_text) {
          setQuestion({
            ordinal: last.ordinal,
            prompt_text: last.prompt_text,
            interview_type: last.interview_type,
            expects_code: false,
            asks_confidence: false,
            is_followup: last.is_followup,
            asked_at: null,
          });
        }
        setRemaining(detail.planned_minutes * 60);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load interview."));
  }, [sessionId]);

  // Countdown. Purely presentational - the backend owns the real clock.
  useEffect(() => {
    if (finished || remaining <= 0) return;
    const timer = setInterval(() => setRemaining((value) => Math.max(0, value - 1)), 1000);
    return () => clearInterval(timer);
  }, [finished, remaining]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, stage]);

  useEffect(() => {
    if (question?.expects_code) setShowEditor(true);
  }, [question]);

  function applyTurn(result: TurnResponse) {
    setRemaining(result.remaining_seconds);

    if (result.clarification?.reply) {
      setTurns((current) => [
        ...current,
        { role: "clarification", text: result.clarification!.reply },
      ]);
    }

    // Coaching Mode teaches between questions; Real Mode never sends one.
    if (result.coaching?.feedback) {
      const parts = [result.coaching.feedback];
      if (result.coaching.next_step_hint) {
        parts.push(`Going into the next one: ${result.coaching.next_step_hint}`);
      }
      setTurns((current) => [...current, { role: "coaching", text: parts.join(" ") }]);
    }

    if (result.status === "completed" || result.scorecard) {
      setFinished(true);
      setQuestion(null);
      return;
    }

    if (result.question?.prompt_text) {
      const next = result.question;
      setQuestion(next);
      setTurns((current) => {
        // The clarification branch re-serves the same question; don't repeat it.
        const lastInterviewer = [...current].reverse().find((t) => t.role === "interviewer");
        if (lastInterviewer?.text === next.prompt_text) return current;
        return [
          ...current,
          {
            role: "interviewer",
            text: next.prompt_text!,
            ordinal: next.ordinal,
            isFollowup: next.is_followup,
          },
        ];
      });
    }
  }

  async function submit() {
    const text = draft.trim();
    if (!text && !code.trim()) return;

    setBusy(true);
    setError(null);
    setStage("Sending");
    setTurns((current) => [
      ...current,
      { role: "candidate", text: text || "(code submitted)" },
    ]);
    setDraft("");

    try {
      const result = await answerStreaming(
        sessionId,
        {
          text,
          code: code.trim() ? code : null,
          language: code.trim() ? language : null,
          self_confidence: confidence,
        },
        (label) => setStage(label),
      );
      applyTurn(result);
      setConfidence(null);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Could not submit your answer. Try again.";
      setError(message);
    } finally {
      setBusy(false);
      setStage(null);
    }
  }

  async function finishNow() {
    setBusy(true);
    try {
      const result = await api.finishInterview(sessionId);
      applyTurn(result);
      setFinished(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not finish the interview.");
    } finally {
      setBusy(false);
    }
  }

  if (finished) {
    return (
      <div className="auth-shell">
        <div className="auth-card" style={{ textAlign: "center" }}>
          <h2 style={{ marginBottom: 10 }}>Interview complete</h2>
          <p className="muted small">
            Your scorecard, the concepts you were confidently wrong about, and a study plan
            are ready.
          </p>
          <button
            className="btn-primary"
            style={{ width: "100%", marginTop: 10 }}
            onClick={() => navigate(`/report/${sessionId}`)}
          >
            View scorecard
          </button>
        </div>
      </div>
    );
  }

  const lowTime = remaining > 0 && remaining < 120;

  return (
    <div className="interview">
      <header className="interview-bar">
        <div className="brand" style={{ padding: 0 }}>
          <span className="brand-mark">G</span>
          Gauntlet
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <span className={`timer${lowTime ? " timer-low" : ""}`}>
            {formatDuration(remaining)} remaining
          </span>
          <button className="btn-ghost" onClick={() => setShowEditor((v) => !v)}>
            {showEditor ? "Hide editor" : "Show editor"}
          </button>
          <button className="btn-danger" onClick={finishNow} disabled={busy}>
            End interview
          </button>
        </div>
      </header>

      <div className={`interview-body${showEditor ? " with-editor" : ""}`}>
        <div className="transcript" ref={transcriptRef}>
          {turns.map((turn, index) => (
            <div
              key={index}
              className={
                turn.role === "interviewer"
                  ? "turn turn-interviewer"
                  : turn.role === "candidate"
                    ? "turn turn-candidate"
                    : turn.role === "coaching"
                      ? "turn turn-coaching"
                      : "turn turn-clarify"
              }
            >
              {turn.role === "coaching" && <div className="turn-role">Coaching</div>}
              {turn.role !== "clarification" && turn.role !== "coaching" && (
                <div className="turn-role">
                  {turn.role === "interviewer"
                    ? turn.isFollowup
                      ? "Interviewer · follow-up"
                      : "Interviewer"
                    : "You"}
                </div>
              )}
              <div className="turn-body">{turn.text}</div>
            </div>
          ))}

          {stage && (
            <div className="turn turn-interviewer">
              <div className="thinking">
                <span className="dot" />
                {stage}…
              </div>
            </div>
          )}

          {error && (
            <div className="callout callout-bad" style={{ maxWidth: "74ch" }}>
              {error}
            </div>
          )}
        </div>

        {showEditor && (
          <div className="editor-pane">
            <div className="editor-header">
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                style={{ width: "auto" }}
              >
                {LANGUAGES.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
              <span className="small faint">
                Submitted with your answer · statically analysed, not executed
              </span>
            </div>
            <Editor
              height="100%"
              language={language}
              theme="vs-dark"
              value={code}
              onChange={(value) => setCode(value ?? "")}
              options={{
                minimap: { enabled: false },
                fontSize: 13,
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 2,
              }}
            />
          </div>
        )}
      </div>

      <div className="composer">
        {question?.asks_confidence && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span className="small muted">
              Before you answer — how confident are you on this topic?
            </span>
            <div className="chips">
              {[1, 2, 3, 4, 5].map((value) => (
                <span
                  key={value}
                  className={`chip${confidence === value ? " selected" : ""}`}
                  onClick={() => setConfidence(value)}
                  title={
                    ["No idea", "Unsure", "Somewhat confident", "Confident", "Extremely confident"][
                      value - 1
                    ]
                  }
                >
                  {value}
                </span>
              ))}
            </div>
          </div>
        )}

        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type your answer. Think out loud — reasoning is scored, not just the conclusion."
          style={{ minHeight: 96 }}
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
        />

        <div className="composer-actions">
          <button
            className="btn-primary"
            onClick={submit}
            disabled={busy || (!draft.trim() && !code.trim())}
          >
            {busy ? "Sending…" : "Send answer"}
          </button>
          <button
            onClick={() => setDraft((d) => (d ? d : "Could you clarify — "))}
            disabled={busy}
          >
            Ask a clarifying question
          </button>
          <span className="small faint">Ctrl/Cmd + Enter to send</span>
        </div>
      </div>
    </div>
  );
}
