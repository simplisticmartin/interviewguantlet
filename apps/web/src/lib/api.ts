/**
 * Typed API client.
 *
 * All requests go through `request()`, which is the single place that attaches the
 * bearer token, normalises errors, and clears a dead session. Response types mirror
 * `apps/api/schemas.py`.
 */

const TOKEN_KEY = "gauntlet.token";
const IDENTITY_KEY = "gauntlet.identity";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface Identity {
  candidateId: string;
  displayName: string;
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getIdentity(): Identity | null {
  const raw = localStorage.getItem(IDENTITY_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as Identity;
  } catch {
    return null;
  }
}

export function setSession(token: string, identity: Identity): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(IDENTITY_KEY, JSON.stringify(identity));
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(IDENTITY_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });

  if (response.status === 401) {
    clearSession();
    throw new ApiError("Your session expired. Sign in again.", 401);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      /* non-JSON error body; keep the generic message */
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/* ---------------------------------------------------------------- types */

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  candidate_id: string;
  display_name: string;
}

export interface ResumeSummary {
  id: string;
  filename: string;
  created_at: string;
  is_primary: boolean;
  years_experience: number;
  concept_count: number;
  claim_count: number;
}

export interface ResumeDetail extends ResumeSummary {
  profile: {
    display_name?: string;
    headline?: string;
    years_experience?: number;
    primary_languages?: string[];
    frameworks?: string[];
    concept_keys?: string[];
    claims?: Array<{
      claim_text: string;
      technologies: string[];
      has_metric: boolean;
      probe_priority: number;
    }>;
  };
  excerpt: string;
}

export interface JobAnalysis {
  id: string;
  title: string;
  level: string;
  must_have: string[];
  weighted_concepts: Array<{ concept_key: string; display_name: string; weight: number }>;
  summary: string;
}

export interface QuestionView {
  ordinal: number | null;
  prompt_text: string | null;
  interview_type: string | null;
  expects_code: boolean;
  asks_confidence: boolean;
  is_followup: boolean;
  asked_at: string | null;
}

export interface Scorecard {
  overall: number;
  category_scores: Record<string, number>;
  strongest_areas: SkillReading[];
  weakest_areas: SkillReading[];
  misconceptions: Array<{
    concept_key: string | null;
    belief: string;
    correction: string;
    evidence_quote: string | null;
    severity: number;
  }>;
  resume_claims_tested: Array<{
    claim: string;
    tested: boolean;
    support: string;
    score: number | null;
    question?: string;
  }>;
  communication_notes: string[];
  missed_opportunities: string[];
  committee: {
    recommendation: string;
    scores: Record<string, number>;
    strengths: string[];
    risks: string[];
    evidence: string[];
    next_steps: string[];
    most_likely_rejection_reason: string;
  };
  study_plan: {
    summary: string;
    items: Array<{
      priority: number;
      concept_key: string;
      title: string;
      rationale: string;
      learn_items: string[];
      practice_items: Array<{ type?: string; prompt?: string }>;
      reattempt_prompt: string | null;
    }>;
  };
  replay_moments: Array<{
    ordinal: number;
    at_minute: number;
    prompt_text: string;
    concept_key: string | null;
    score: number;
    note: string;
  }>;
  questions_asked: number;
  duration_minutes: number;
}

export interface SkillReading {
  concept_key: string;
  display_name: string;
  mastery: number;
  confidence: number;
  evidence_count: number;
  self_confidence: number | null;
  is_misconception?: boolean;
}

export interface TurnResponse {
  session_id: string;
  status: string;
  question: QuestionView | null;
  clarification: { reply: string } | null;
  /** Coaching Mode only. Real Interview Mode never returns one. */
  coaching: { feedback: string; key_correction: string | null; next_step_hint: string | null } | null;
  scorecard: Scorecard | null;
  remaining_seconds: number;
  questions_asked: number;
}

export interface InterviewSummary {
  id: string;
  target_role: string;
  target_level: string;
  mode: string;
  status: string;
  company: string | null;
  planned_minutes: number;
  questions_asked: number;
  overall: number | null;
  recommendation: string | null;
  started_at: string | null;
  ended_at: string | null;
}

export interface TranscriptEntry {
  ordinal: number;
  prompt_text: string;
  interview_type: string;
  is_followup: boolean;
  answer_text: string | null;
  self_confidence: number | null;
  score: number | null;
  concept_keys: string[];
}

export interface InterviewDetail extends InterviewSummary {
  transcript: TranscriptEntry[];
  scorecard: Scorecard | null;
  plan: Record<string, unknown>;
}

export interface SkillView {
  concept_key: string;
  display_name: string;
  mastery: number;
  confidence: number;
  evidence_count: number;
  self_confidence: number | null;
  calibration: string;
  due_at: string | null;
}

export interface Analytics {
  interviews_completed: number;
  questions_answered: number;
  average_overall: number | null;
  readiness: SkillView[];
  strongest: SkillView[];
  weakest: SkillView[];
  open_misconceptions: Array<{
    concept_key: string;
    display_name: string;
    belief: string;
    correction: string;
    severity: number;
    times_observed: number;
  }>;
  improvement: Array<{ session_id: string; label: string; overall: number; role: string }>;
  due_for_review: SkillView[];
  confidence_calibration: Record<string, number>;
}

export interface StudyPlan {
  id: string | null;
  summary: string;
  created_at: string | null;
  items: Array<{
    priority: number;
    concept_key: string;
    display_name: string;
    title: string;
    rationale: string;
    learn_items: string[];
    practice_items: Array<{ type?: string; prompt?: string }>;
    status: string;
  }>;
}

export interface QuestionSearchResult {
  id: string | null;
  question: string;
  interview_type: string;
  concept_keys: string[];
  topics: string[];
  difficulty: number;
  rubric_key: string | null;
  question_origin: string;
  source_type: string;
  score: number | null;
}

export interface CompanyView {
  slug: string;
  name: string;
  sector: string;
  aliases: string[];
}

export interface CompanyPatterns {
  slug: string;
  name: string;
  sector: string;
  evidence: string;
  basis: string;
  disclaimer: string;
  distribution: Record<string, number>;
  readiness: {
    level: string;
    estimated_readiness: number | null;
    coverage: number;
    caveat: string;
    areas: Array<{
      interview_type: string;
      weight: number;
      score: number | null;
      evidence_count: number;
      measured: boolean;
    }>;
  } | null;
}

export interface Health {
  status: string;
  version: string;
  database: boolean;
  llm_provider: string;
  llm_degraded: boolean;
  durable_checkpoints: boolean;
  semantic_embeddings: boolean;
}

/* ---------------------------------------------------------------- calls */

export const api = {
  health: () => request<Health>("/health"),

  register: (email: string, password: string, displayName: string) =>
    request<TokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, display_name: displayName }),
    }),

  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  listResumes: () => request<ResumeSummary[]>("/resumes"),

  uploadResume: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ResumeDetail>("/resumes", { method: "POST", body: form });
  },

  pasteResume: (text: string, filename = "pasted-resume.txt") =>
    request<ResumeDetail>("/resumes/text", {
      method: "POST",
      body: JSON.stringify({ text, filename }),
    }),

  analyzeJob: (text: string, company?: string) =>
    request<JobAnalysis>("/jobs/analyze", {
      method: "POST",
      body: JSON.stringify({ text, company: company || null }),
    }),

  createInterview: (payload: {
    resume_id?: string | null;
    job_description_id?: string | null;
    target_role: string;
    target_level: string;
    company?: string | null;
    mode: string;
    interview_types: string[];
    minutes: number;
  }) => request<TurnResponse>("/interviews", { method: "POST", body: JSON.stringify(payload) }),

  answer: (
    sessionId: string,
    payload: {
      text: string;
      code?: string | null;
      language?: string | null;
      self_confidence?: number | null;
    },
  ) =>
    request<TurnResponse>(`/interviews/${sessionId}/answer`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  finishInterview: (sessionId: string) =>
    request<TurnResponse>(`/interviews/${sessionId}/finish`, { method: "POST" }),

  listInterviews: () => request<InterviewSummary[]>("/interviews"),
  getInterview: (id: string) => request<InterviewDetail>(`/interviews/${id}`),

  skills: () => request<SkillView[]>("/skills"),
  analytics: () => request<Analytics>("/analytics"),
  studyPlan: () => request<StudyPlan>("/study-plan"),

  searchQuestions: (params: Record<string, string>) =>
    request<QuestionSearchResult[]>(`/questions/search?${new URLSearchParams(params)}`),

  companies: () => request<CompanyView[]>("/companies"),
  companyPatterns: (slug: string) => request<CompanyPatterns>(`/companies/${slug}/patterns`),
};

/**
 * Submit an answer over SSE so the UI can show which stage of the pipeline is running.
 * Falls back to the plain POST when streaming is unavailable.
 */
export async function answerStreaming(
  sessionId: string,
  payload: { text: string; code?: string | null; language?: string | null; self_confidence?: number | null },
  onStage: (label: string) => void,
): Promise<TurnResponse> {
  const token = getToken();
  const response = await fetch(`${API_BASE}/interviews/${sessionId}/answer/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    return api.answer(sessionId, payload);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: TurnResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;

      const data = dataLines.join("\n");
      if (event === "stage") {
        try {
          onStage((JSON.parse(data) as { label: string }).label);
        } catch {
          /* ignore malformed stage frames */
        }
      } else if (event === "turn") {
        result = JSON.parse(data) as TurnResponse;
      } else if (event === "error") {
        const detail = (JSON.parse(data) as { detail?: string }).detail;
        throw new ApiError(detail ?? "The interview could not continue.", 500);
      }
    }
  }

  if (!result) throw new ApiError("The interview stream ended unexpectedly.", 500);
  return result;
}
