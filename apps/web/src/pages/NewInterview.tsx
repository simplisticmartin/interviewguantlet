import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Card, ErrorNote, PageHeader } from "../components/ui";
import { api } from "../lib/api";

const INTERVIEW_TYPES = [
  { value: "java", label: "Java" },
  { value: "spring", label: "Spring" },
  { value: "database", label: "Databases" },
  { value: "distributed", label: "Distributed / Kafka" },
  { value: "system_design", label: "System design" },
  { value: "dsa", label: "Coding / DSA" },
  { value: "cloud", label: "Cloud / DevOps" },
  { value: "behavioral", label: "Behavioural" },
  { value: "hiring_manager", label: "Hiring manager" },
];

const MODES = [
  { value: "real", label: "Real interview", hint: "No hints, no live scores. Timed." },
  { value: "coaching", label: "Coaching", hint: "Same questions, guidance allowed." },
  { value: "resume_defense", label: "Resume defence", hint: "Cross-examine your own claims." },
  { value: "system_design", label: "System design", hint: "Architecture only." },
  { value: "coding", label: "Coding", hint: "DSA with the editor." },
  { value: "behavioral", label: "Behavioural", hint: "Evidence-based behavioural round." },
];

const LEVELS = ["junior", "mid", "senior", "staff", "principal"];

export default function NewInterview() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);

  const [resumeId, setResumeId] = useState<string | null>(null);
  const [resumeText, setResumeText] = useState("");
  const [jobText, setJobText] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [role, setRole] = useState("Senior Java Engineer");
  const [level, setLevel] = useState("senior");
  const [company, setCompany] = useState("");
  const [mode, setMode] = useState("real");
  const [minutes, setMinutes] = useState(20);
  const [types, setTypes] = useState<string[]>(["java", "spring", "database", "distributed", "system_design"]);
  const [error, setError] = useState<string | null>(null);

  const resumes = useQuery({ queryKey: ["resumes"], queryFn: api.listResumes });
  const companies = useQuery({ queryKey: ["companies"], queryFn: api.companies });

  const uploadResume = useMutation({
    mutationFn: (file: File) => api.uploadResume(file),
    onSuccess: (data) => {
      setResumeId(data.id);
      queryClient.invalidateQueries({ queryKey: ["resumes"] });
    },
    onError: (err) => setError(err instanceof Error ? err.message : "Upload failed."),
  });

  const pasteResume = useMutation({
    mutationFn: (text: string) => api.pasteResume(text),
    onSuccess: (data) => {
      setResumeId(data.id);
      setResumeText("");
      queryClient.invalidateQueries({ queryKey: ["resumes"] });
    },
    onError: (err) => setError(err instanceof Error ? err.message : "Could not save resume."),
  });

  const analyzeJob = useMutation({
    mutationFn: (text: string) => api.analyzeJob(text, company || undefined),
    onSuccess: (data) => setJobId(data.id),
    onError: (err) => setError(err instanceof Error ? err.message : "Could not analyse job."),
  });

  const start = useMutation({
    mutationFn: async () => {
      // Analyse the pasted job description on the fly if the user skipped the button.
      let resolvedJobId = jobId;
      if (!resolvedJobId && jobText.trim().length > 50) {
        resolvedJobId = (await api.analyzeJob(jobText, company || undefined)).id;
      }
      return api.createInterview({
        resume_id: resumeId,
        job_description_id: resolvedJobId,
        target_role: role,
        target_level: level,
        company: company || null,
        mode,
        interview_types: types,
        minutes,
      });
    },
    onSuccess: (turn) => navigate(`/interview/${turn.session_id}`),
    onError: (err) => setError(err instanceof Error ? err.message : "Could not start interview."),
  });

  const selectedResume = resumes.data?.find((item) => item.id === resumeId);
  const busy = start.isPending || uploadResume.isPending || pasteResume.isPending;

  return (
    <>
      <PageHeader
        title="Start an interview"
        subtitle="Gauntlet reads your resume and the job description, decides what a real loop would probe, then adapts as you answer."
      />

      {error && <div style={{ marginBottom: 16 }}><ErrorNote error={new Error(error)} /></div>}

      <div className="grid grid-2">
        <Card title="1. Your resume">
          {resumes.data && resumes.data.length > 0 && (
            <div className="field">
              <label htmlFor="resume-select">Use an existing resume</label>
              <select
                id="resume-select"
                value={resumeId ?? ""}
                onChange={(e) => setResumeId(e.target.value || null)}
              >
                <option value="">None — interview without a resume</option>
                {resumes.data.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.filename} · {item.claim_count} claims
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="field">
            <label>Upload a new one</label>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.txt,.md"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadResume.mutate(file);
              }}
            />
            <div className="field-hint">PDF, DOCX, TXT or Markdown, up to 5 MB.</div>
          </div>

          <details>
            <summary className="small muted" style={{ cursor: "pointer", marginBottom: 8 }}>
              Or paste the text instead
            </summary>
            <textarea
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              placeholder="Paste your resume…"
            />
            <button
              style={{ marginTop: 8 }}
              disabled={resumeText.trim().length < 50 || pasteResume.isPending}
              onClick={() => pasteResume.mutate(resumeText)}
            >
              {pasteResume.isPending ? "Parsing…" : "Save resume"}
            </button>
          </details>

          {selectedResume && (
            <div className="callout" style={{ marginTop: 12 }}>
              Parsed <strong>{selectedResume.claim_count}</strong> checkable claims and{" "}
              <strong>{selectedResume.concept_count}</strong> known concepts. High-signal claims
              will be cross-examined during the interview.
            </div>
          )}
        </Card>

        <Card title="2. The job">
          <div className="field">
            <label htmlFor="jd">Job description</label>
            <textarea
              id="jd"
              value={jobText}
              onChange={(e) => {
                setJobText(e.target.value);
                setJobId(null);
              }}
              placeholder="Paste the posting…"
              style={{ minHeight: 150 }}
            />
          </div>
          <button
            disabled={jobText.trim().length < 50 || analyzeJob.isPending}
            onClick={() => analyzeJob.mutate(jobText)}
          >
            {analyzeJob.isPending ? "Analysing…" : "Analyse job description"}
          </button>

          {analyzeJob.data && (
            <div style={{ marginTop: 12 }}>
              <div className="small muted" style={{ marginBottom: 6 }}>
                Likely to be assessed:
              </div>
              <div className="chips">
                {analyzeJob.data.weighted_concepts.slice(0, 12).map((concept) => (
                  <span key={concept.concept_key} className="chip">
                    {concept.display_name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>
      </div>

      <Card title="3. Target" className="grid-2">
        <div className="grid grid-3">
          <div className="field">
            <label htmlFor="role">Role</label>
            <input id="role" value={role} onChange={(e) => setRole(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="level">Seniority</label>
            <select id="level" value={level} onChange={(e) => setLevel(e.target.value)}>
              {LEVELS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="company">Company (optional)</label>
            <select id="company" value={company} onChange={(e) => setCompany(e.target.value)}>
              <option value="">Generic</option>
              {companies.data?.map((item) => (
                <option key={item.slug} value={item.slug}>
                  {item.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {company && (
          <div className="callout callout-warn small">
            Company simulation uses Gauntlet's <strong>estimated</strong> interview mix for this
            kind of engineering organisation. It is not based on observed reports from this
            company and does not reproduce their real process.
          </div>
        )}
      </Card>

      <Card title="4. Format">
        <div className="field">
          <label>Mode</label>
          <div className="chips">
            {MODES.map((item) => (
              <span
                key={item.value}
                className={`chip${mode === item.value ? " selected" : ""}`}
                title={item.hint}
                onClick={() => setMode(item.value)}
              >
                {item.label}
              </span>
            ))}
          </div>
          <div className="field-hint">{MODES.find((m) => m.value === mode)?.hint}</div>
        </div>

        <div className="field">
          <label>Areas to cover</label>
          <div className="chips">
            {INTERVIEW_TYPES.map((item) => (
              <span
                key={item.value}
                className={`chip${types.includes(item.value) ? " selected" : ""}`}
                onClick={() =>
                  setTypes((current) =>
                    current.includes(item.value)
                      ? current.filter((v) => v !== item.value)
                      : [...current, item.value],
                  )
                }
              >
                {item.label}
              </span>
            ))}
          </div>
        </div>

        <div className="field" style={{ maxWidth: 280 }}>
          <label htmlFor="minutes">Duration: {minutes} minutes</label>
          <input
            id="minutes"
            type="range"
            min={10}
            max={60}
            step={5}
            value={minutes}
            onChange={(e) => setMinutes(Number(e.target.value))}
          />
        </div>

        <button
          className="btn-primary"
          disabled={busy || types.length === 0}
          onClick={() => {
            setError(null);
            start.mutate();
          }}
        >
          {start.isPending ? "Preparing your interview…" : "Start interview"}
        </button>
        {types.length === 0 && (
          <div className="field-hint">Pick at least one area to cover.</div>
        )}
      </Card>
    </>
  );
}
