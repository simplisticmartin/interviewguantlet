# Architecture notes

Supplements the [README](../README.md) with the designs behind the parts that are
partially built, and the decisions that are easy to get wrong later.

---

## 1. Why the graph, and not a loop

An interview is a state machine with a human in the middle of it. The parts that make
that hard (the process can die mid-interview, the candidate can close the tab, a
question needs re-attempting later) are all *state persistence* problems, and LangGraph's
checkpointer solves them once instead of in every handler.

The alternative (a `while` loop in a request handler holding an interview in memory) works
until the first restart. The alternative-alternative (hand-rolled state in Postgres)
means writing checkpointing, resumption, and history yourself.

The cost is that **everything in `InterviewState` must be JSON-serialisable**. A rich
object in state silently breaks checkpoint portability, so `tests/integration/
test_interview_graph.py::test_state_is_json_serialisable` enforces it, and every node
writes `model_dump(mode="json")` rather than `model_dump()`.

### Interrupt semantics

`interrupt()` re-executes its node from the top on resume, so the waiting nodes contain
nothing but the interrupt call. Any work placed there would run twice.

```python
def wait_for_candidate(state):
    payload = interrupt({...})          # graph suspends here; state is checkpointed
    return {"pending_answer": payload}  # runs only after Command(resume=...)
```

### The clarification branch

A clarifying question must not consume a question slot, and must not be graded. It routes
to `answer_clarification → wait_after_clarification → classify_response`, which returns
the floor to the candidate with the same question standing. A clarification that leaks
part of the expected answer sets `gave_away_answer`, which is recorded as a hint and
reduces the evidence weight of the eventual answer.

---

## 2. Where LLM judgement is and is not allowed

The consistent pattern: **deterministic code computes the facts, the model handles
judgement, deterministic code enforces the constraints.**

| Decision | Owner | Why |
|---|---|---|
| Which concepts the candidate claims | Code (set intersection) | The model must not hallucinate the candidate's skills |
| Which areas to weight, and why | Model | Genuine judgement about a role |
| How proportions become question slots | Code (largest remainder) | Arithmetic, and it must total exactly |
| Question wording | Model | This is the craft |
| Question difficulty, concept, rubric | Code | The model owns wording, the system owns measurement |
| Whether an answer demonstrates a rubric dimension | Model | Genuine judgement |
| How judges combine into a score | Code | Weights must be auditable and stable |
| Direction of the next question | Model proposes | Interview craft |
| Follow-up budget, saturation, difficulty bands | Code enforces | Interview economics; a model that occasionally forgets wastes real time |

`InterviewerAgent._finalise()` is where this is visible: whatever the model returns, the
system overwrites `interview_type`, `concept_keys`, `difficulty`, and `rubric_key` with
what it decided. The model cannot quietly change what is being measured.

---

## 3. MCP architecture

Built, apart from the sandboxed execution behind the coding server:

```
┌──────────────┐     stdio / HTTP      ┌────────────────────┐
│  Gauntlet    │◄─────────────────────►│  coding-mcp        │
│  interviewer │                       │  compile, run,     │
│  agent       │                       │  visible + hidden  │
│              │                       │  tests, metrics    │
│              │                       └─────────┬──────────┘
│              │                                 │ spawns
│              │                       ┌─────────▼──────────┐
│              │                       │ ephemeral container │
│              │                       │ no network, capped  │
│              │                       │ CPU/mem/wall-clock  │
│              │                       └────────────────────┘
│              │                       ┌────────────────────┐
│              │◄─────────────────────►│  candidate-mcp     │
│              │                       │  resume, history,  │
│              │                       │  skill graph       │
│              │                       └────────────────────┘
│              │                       ┌────────────────────┐
│              │◄─────────────────────►│  question-bank-mcp │
└──────────────┘                       │  search, families  │
                                       └────────────────────┘
```

Two rules the current code already respects:

1. **Retrieved text never selects a tool.** Corpus and resume content arrive as fenced
   data. When MCP lands, tool selection stays driven by graph state, never by document
   content, otherwise a crafted resume becomes a remote tool-call primitive.
2. **Hidden test results are interviewer-only.** They inform the next question
   (`"what assumptions are you making about the input?"`), they are never shown. The
   existing `_public_question()` filter is the same mechanism.

`gauntlet/execution/static_check.py` is the seam. It produces `interviewer_signals` from
structure alone (nested loops, no empty-input guard, recursion) and returns
`executed: false` everywhere. Sandboxing replaces its internals and adds real test
results; neither the coding server's interface nor the interviewer's consumption of
signals has to change.

**What exists today.** Three servers under `gauntlet/mcp/`, registered as console scripts
and verified against the real protocol in `tests/integration/test_mcp_protocol.py`, which
spawns each one as a subprocess and talks to it with the SDK client.

They live at `gauntlet/mcp/` rather than the top-level `mcp/` the spec sketches, because a
top-level directory of that name shadows the installed `mcp` SDK for anything run from the
repository root. Same structure, one fewer footgun.

The coding server exposes no `compile_code`, `run_visible_tests` or `run_hidden_tests`,
and a test asserts those names are absent from the manifest. A tool that claims to run
tests and does not is a worse failure than a missing tool, because the caller believes it.

**What is still missing** is the sandbox itself: ephemeral containers, no network, hard
CPU, memory and wall-clock limits, guaranteed teardown. Until that exists the coding
server analyses and does not execute, and says so in every response.

---

## 4. Ingestion pipeline (partly built)

Deduplication, recency weighting, safety screening, the orchestrated pipeline and the
moderation queue are implemented in `gauntlet/ingestion/` and `gauntlet/services/
contributions.py`. What is still missing is the external source adapters: today the only
input is a person submitting a question they were asked, through the API.

The schema was built first, because it is the part that is expensive to retrofit:
`questions` carries full provenance, `question_families` exists for canonicalisation, and
`company_question_occurrences` stores counts and dates rather than claims.

```
source adapter                    ← not built. Today the only input is a user submission
      ↓
parser → question extractor
      ↓
PII / safety filter               ← strip names, interviewer identities, employer-confidential
      ↓
normaliser → concept tagger → company/role classifier
      ↓
deduplicator                      ← embedding similarity + algorithm tags → QuestionFamily
      ↓
provenance storage → review queue ← human approval before anything reaches production
      ↓
production corpus
```

**Deduplication** is what stops question counts inflating. "Find two numbers summing to
target", "return two indices whose values add to target", and "find a pair adding to K"
are one `QuestionFamily` with three variants, not three questions.

**Recency weighting.** `company_question_occurrences.last_reported_on` exists so
confidence can decay with age on the same principle as the mastery model: a 2019 report
is archival, a 2026 report is strong evidence. The decay function is configurable rather
than baked in.

**Safety runs first, not last.** `gauntlet/ingestion/safety.py` screens before anything
else touches the text, because screening after tagging and deduplication would mean an
NDA-covered submission had already been embedded and compared against the corpus before
anyone decided to refuse it. The filter does three different things deliberately kept
apart: it *redacts* unambiguous identifiers (emails, phones, links, handles), *blocks*
content that no amount of scrubbing makes publishable (NDA markers, leaked assessments),
and *flags for review* anything uncertain. Names are the hard case: redacting every
capitalised word would destroy "Kafka" and "Redis", so names are matched only in
constructions that genuinely introduce a person, and a removed name always escalates to a
human because the surrounding sentence can still identify someone.

**Nothing publishes automatically.** An accepted submission becomes a `pending` row in
`question_submissions`, never a row in `questions`. Separate tables rather than a draft
flag, so serving unreviewed content is not one forgotten `WHERE` clause away. Only a
moderator's approval writes to the corpus, and what it writes is always
`question_origin="user_submitted"` at low starting confidence, since one person reporting
a question is weak evidence.

**Near-duplicates go to the reviewer, not to a threshold.** Above the merge threshold the
pipeline merges; between the near floor and that threshold it queues the submission with
the similar question attached and lets a person decide. That band widens when no embedding
provider is configured and only the lexical signal is available.

**Only redacted text is stored.** The raw submission is never persisted. Keeping the
original "just in case" would defeat the entire screening step, since the personal data
would still sit in the database.

**Sources it will not accept:** paywalled databases, leaked assessments, anything whose
licence does not permit reuse. Adapters are per-source specifically so those terms are
encoded in one auditable place.

---

## 5. Deployment

The API image (`infra/api.Dockerfile`) runs migrations then serves under a non-root user.
The stack is Kubernetes-ready in the ways that matter:

**Stateless API.** All state is in Postgres, including LangGraph checkpoints, so
replicas scale horizontally.

**Two things must change before running more than one replica:**

1. **Rate limiting** is in-process (`apps/api/deps.py`). With N replicas the effective
   limit is N × configured. Swap for the Redis-backed implementation; it is a dependency
   precisely so this is a one-line change.
2. **The checkpointer connection pool** is per-process. Size `pool_size` against your
   Postgres `max_connections` divided by replica count.

**Health.** `/health` reports database reachability, active provider, whether the
provider degraded to offline, checkpoint durability, and whether embeddings are semantic.
Use `database: true` and `durable_checkpoints: true` as readiness gates,
`durable_checkpoints: false` means interviews will not survive a pod restart.

**Configuration.** Everything is read once through `gauntlet/config.py`; nothing else
touches `os.environ`. Production refuses to boot with a development secret key.

**Observability.** Structured logs via structlog: JSON in production, human-readable in
development. Every agent call logs prompt name, version, provider, model, attempt count,
and token usage, which is what makes cost-per-interview measurable. OpenTelemetry spans
around graph nodes and agent calls are the next increment; the log points are already at
the right boundaries.

**Backups.** `skill_evidence` is the irreplaceable table. Skill states are derived and
can be recomputed from it (`recompute_skill_states`); the evidence rows cannot be
reconstructed from anything.

---

## 6. Observability

An interview is one HTTP request that fans out into a dozen model calls across seven or
more graph nodes. When one is slow or expensive, per-call logs tell you *that* something
took eleven seconds, not *which* part of which turn. Spans nest, so they answer the
question logs structurally cannot.

**Instrumentation is unconditional. Collection is opt-in.**

The OpenTelemetry API ships a no-op tracer, so `span()` costs almost nothing when no SDK
is installed. That means there is no `if tracing_enabled:` scattered through the interview
logic, and no second code path that only runs in production. Turning collection on:

```
pip install -e ".[otel]"
GAUNTLET_OTEL_ENABLED=true GAUNTLET_OTEL_ENDPOINT=http://localhost:4317
```

With tracing on but no endpoint set, spans print to the console rather than vanishing.
A misconfigured collector and a working one should not look identical.

**Where spans come from.** Nodes are wrapped at registration rather than decorated
individually, so a node added tomorrow is traced without anyone remembering to do it, and
`gauntlet/graph/nodes.py` stays free of observability imports. Each node span records
which state keys it wrote, which is the most useful single fact when reading the trace of
a state machine. Agent spans are created in `Agent.invoke`, the one funnel every model
call already passes through, so they nest inside the node that made them.

Model call attributes follow the OpenTelemetry GenAI semantic conventions
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`), so the spans mean
something to tooling that was not written for this project.

**Tracing observes and never participates.** The span wrapper returns the node's result
untouched and re-raises every exception after recording it. Instrumentation that can
change a return value or swallow an error is a liability, not a diagnostic, and there are
tests asserting exactly this.

**Logs and traces are joined.** A structlog processor stamps the active trace and span id
onto every log line, and the API returns the trace id in an `X-Trace-Id` header. Without
that, tracing and logging are two accounts of the same incident that cannot be lined up.

### Cost accounting

"What did this interview cost?" cannot be answered from token counts, because every model
prices differently, so `gauntlet/observability/cost.py` holds a price table keyed by model
family prefix. Prefixes rather than exact ids: vendors append dates and version suffixes
constantly, and an exact-match table goes stale the moment a new snapshot ships.

**An unknown model returns `None`, never zero.** This is the decision worth defending.
Reporting `$0.00` for a model missing from the table produces a total that looks
authoritative and is wrong, and that is the kind of wrong that survives all the way into a
budget spreadsheet. `None` propagates as "unknown", a `CostTally` that skipped a call
reports `complete=False`, and the phrasing changes from "$0.42" to "at least $0.42". Free
and unknown are different claims and the code keeps them apart: the offline provider is
priced at zero because it genuinely is free.

The tally is a context variable rather than a parameter threaded through twenty
signatures, scoped per request. Ambient bookkeeping does not belong in the interview
logic's interface, and per-request scoping stops two concurrent interviews pooling their
spend into one number.

Prices are estimates recorded at a point in time and are labelled as such. Querying a
pricing API would be more accurate and would put a network call on a path that must work
offline, which is a bad trade for a figure shown to four decimal places.

---

## 7. Provider portability

The component most likely to change in an AI product is the model provider. Prices,
capabilities and availability all move on a monthly cadence, and being locked to one
vendor is a structural risk rather than an inconvenience.

So `gauntlet/llm/` is the only part of the codebase that knows a vendor exists. Everything
above it asks for a validated Pydantic model and receives one.

**The load bearing observation** is that most vendors deliberately implement OpenAI's
chat completions wire format. That turns "support twenty providers" from twenty
integrations into one adapter plus a data table. `providers/presets.py` holds base URLs,
default models, key environment variables and capability flags; adding a provider is a
row, not code.

Two adapters exist:

- `AnthropicProvider`, native, because forced tool use gives schema enforcement at the API
  rather than in the prompt, which is materially more reliable.
- `OpenAICompatibleProvider`, serving OpenAI, Gemini, DeepSeek, xAI, Moonshot, Qwen,
  Mistral, Cohere, Groq, Cerebras, Together, Fireworks, DeepInfra, Nebius, SambaNova,
  Hyperbolic, OpenRouter, Perplexity, GitHub Models, Azure, Ollama, LM Studio, vLLM,
  llama.cpp, and any other gateway speaking that format.

Three details that only show up once you actually try to be portable:

**Not every provider implements JSON mode, and they fail loudly rather than ignoring it.**
Passing `response_format` to an endpoint that does not support it is usually a hard 400,
not a silent no-op. The preset records support, and the adapter additionally detects the
error at runtime, disables JSON mode for the rest of the process, and retries. The schema
contract always goes in the prompt regardless, because JSON mode guarantees valid JSON,
not JSON of the right shape.

**Several strong chat providers have no embedding endpoint at all.** DeepSeek, xAI, Groq,
Moonshot and Cerebras do not offer one. Tying embeddings to the chat provider would mean
retrieval silently degrading the moment someone switched models. So embeddings resolve
independently, with their own provider, key, base URL and model, and fall back to a
deterministic local embedder that reports `semantic_embeddings: false`.

**Reasoning models sometimes return an empty `content`** with the substance in a separate
reasoning field. The adapter checks for that rather than treating it as an empty response.

The practical payoff is that the same interview can run on a frontier model, on a
7B model on a laptop with no network, or on the offline rule based engine, and nothing
above the provider boundary changes. `gauntlet-providers` prints the current state.

## 8. Things deliberately not done

Recording these so they read as decisions rather than oversights.

**No microservices.** The interview engine, evaluation, and skill graph share a
transaction boundary and a domain model. Splitting them buys distributed-systems problems
and nothing else. The boundaries that are genuinely separate, code execution and MCP tool
servers, are separate processes by nature.

**No caching layer yet.** There is no measured hot path. Redis is provisioned for the
distributed rate limiter and the future execution queue.

**No streaming of question text token-by-token.** The SSE endpoint streams *pipeline
stages*, not tokens. A question that materialises word by word invites the candidate to
start answering before they have read it, and the interruption of the interviewer
mid-question is not how interviews work.

**No live score display.** Deliberate, and the API enforces it: `_public_question()`
strips rubric, concept keys, and probe reason, and per-question scores are withheld from
the transcript until the interview completes. Showing a running score changes how people
answer the next question, which destroys the measurement.
