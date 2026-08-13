"""Seed question corpus.

Every question here is **original, authored for Gauntlet**. Nothing is scraped,
paywalled, or lifted from a proprietary question bank, and nothing here is attributed
to any company - questions carry ``origin="generated"`` and are surfaced as
Gauntlet-authored (spec section 13 and the ethics constraints in section 55).

The corpus exists to give the interviewer real material to draw from and to seed the
retrieval layer. It is intentionally scenario-shaped rather than trivia-shaped: the
adaptive engine gets more signal from "your consumer sees duplicates after a rebalance"
than from "define idempotency".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gauntlet.schemas import InterviewType

T = InterviewType


@dataclass(frozen=True, slots=True)
class QuestionSeed:
    slug: str
    question: str
    interview_type: InterviewType
    concept_keys: tuple[str, ...]
    difficulty: int
    rubric_key: str | None = None
    follow_ups: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    level: str | None = None
    expects_code: bool = False
    asks_confidence: bool = False
    role_family: str = "Software Engineering"
    based_on_patterns: tuple[str, ...] = field(default=())


def _q(
    slug: str,
    question: str,
    interview_type: InterviewType,
    concept_keys: tuple[str, ...],
    difficulty: int,
    rubric_key: str | None = None,
    *,
    follow_ups: tuple[str, ...] = (),
    expects_code: bool = False,
    asks_confidence: bool = False,
    level: str | None = None,
) -> QuestionSeed:
    return QuestionSeed(
        slug=slug,
        question=question,
        interview_type=interview_type,
        concept_keys=concept_keys,
        difficulty=difficulty,
        rubric_key=rubric_key,
        follow_ups=follow_ups,
        topics=tuple(key.split(".")[-1] for key in concept_keys),
        level=level,
        expects_code=expects_code,
        asks_confidence=asks_confidence,
    )


QUESTIONS: tuple[QuestionSeed, ...] = (
    # --- Java: collections ------------------------------------------------
    _q("java-hashmap-internals",
       "Walk me through what happens inside a HashMap when I call put with a key that "
       "collides with an existing entry.",
       T.JAVA, ("java.collections.hashmap",), 3, "java.collections.hashmap",
       follow_ups=("What changes once that bucket gets very long?",
                   "What triggers a resize, and what does it cost?"),
       asks_confidence=True),
    _q("java-hashmap-mutable-key",
       "Someone uses a mutable object as a HashMap key and mutates it after insertion. "
       "What happens, and why?",
       T.JAVA, ("java.collections.hashmap", "java.collections.equals_hashcode"), 4,
       "java.collections.hashmap"),
    _q("java-equals-hashcode",
       "You override equals but not hashCode on a class used in a HashSet. Describe the "
       "bug a user would report.",
       T.JAVA, ("java.collections.equals_hashcode",), 3, "java.collections.hashmap"),
    _q("java-arraylist-vs-linkedlist",
       "A colleague swaps an ArrayList for a LinkedList to make removals faster and the "
       "service gets slower. Explain.",
       T.JAVA, ("java.collections.arraylist",), 3),
    # --- Java: concurrency ------------------------------------------------
    _q("java-chm-vs-synchronized",
       "How does ConcurrentHashMap differ from wrapping a HashMap in "
       "Collections.synchronizedMap?",
       T.JAVA, ("java.concurrency.concurrent_hashmap",), 3,
       "java.concurrency.concurrent_hashmap",
       follow_ups=("If one lock covered the whole map, what would happen when several "
                   "threads touched unrelated keys?",),
       asks_confidence=True),
    _q("java-chm-counter",
       "You keep per-user request counts in a ConcurrentHashMap and increments are going "
       "missing under load. What is wrong?",
       T.JAVA, ("java.concurrency.concurrent_hashmap",), 4,
       "java.concurrency.concurrent_hashmap"),
    _q("java-volatile-counter",
       "A field is marked volatile and incremented from several threads. Is that "
       "thread-safe? Walk me through why.",
       T.JAVA, ("java.concurrency.volatile", "java.concurrency.memory_model"), 4,
       "java.concurrency.memory_model", asks_confidence=True),
    _q("java-visibility-loop",
       "One thread flips a boolean flag; another spins on it and never notices. No "
       "exception, no crash. What is happening?",
       T.JAVA, ("java.concurrency.memory_model",), 4, "java.concurrency.memory_model"),
    _q("java-threadpool-sizing",
       "How would you size a thread pool for a service that mostly waits on downstream "
       "HTTP calls?",
       T.JAVA, ("java.concurrency.executors",), 3),
    _q("java-deadlock-debug",
       "Production threads are stuck and CPU is near zero. Walk me through how you "
       "diagnose it.",
       T.JAVA, ("java.concurrency.locks", "cloud.incident_response"), 4),
    _q("java-virtual-threads",
       "Where do virtual threads actually help, and where do they not?",
       T.JAVA, ("java.virtual_threads", "java.concurrency.executors"), 4, level="senior"),
    # --- Java: JVM --------------------------------------------------------
    _q("java-gc-latency",
       "Your p99 latency spikes every few minutes while p50 stays flat. How do you work "
       "out whether GC is responsible?",
       T.JAVA, ("java.jvm.gc", "system_design.observability"), 4, "java.jvm.gc"),
    _q("java-memory-leak",
       "A Java service climbs to OOM over several days. Where do you start?",
       T.JAVA, ("java.jvm.gc",), 4, "java.jvm.gc"),
    _q("java-streams-pitfall",
       "When would you not use a parallel stream, even for a large collection?",
       T.JAVA, ("java.streams",), 3),
    _q("java-optional-usage",
       "Where does Optional genuinely help, and where does it just add noise?",
       T.JAVA, ("java.optional",), 2),
    _q("java-checked-exceptions",
       "How do you decide between a checked and an unchecked exception in a service layer?",
       T.JAVA, ("java.exceptions",), 3),
    # --- Spring -----------------------------------------------------------
    _q("spring-di-constructor",
       "Why is constructor injection generally preferred over field injection?",
       T.SPRING, ("spring.di",), 2, "spring.di"),
    _q("spring-singleton-state",
       "A teammate adds a mutable HashMap field to a @Service to cache results. What "
       "concerns you?",
       T.SPRING, ("spring.bean_scopes", "java.concurrency"), 4, "spring.di",
       asks_confidence=True),
    _q("spring-transactional-self-invocation",
       "A public method in a @Service calls another method in the same class annotated "
       "with @Transactional. Is the second method transactional?",
       T.SPRING, ("spring.transactions",), 4, "spring.transactions", asks_confidence=True),
    _q("spring-transaction-rollback",
       "Your @Transactional method throws a checked exception. Does the transaction roll "
       "back?",
       T.SPRING, ("spring.transactions",), 3, "spring.transactions"),
    _q("spring-transaction-boundary",
       "A transactional method also calls a third-party payment API. What is wrong with "
       "that shape?",
       T.SPRING, ("spring.transactions", "distributed.transactions"), 4,
       "spring.transactions"),
    _q("spring-n-plus-one",
       "An endpoint returning 50 orders issues 51 queries. What happened and how do you "
       "fix it?",
       T.SPRING, ("spring.data.n_plus_one", "database.query_optimization"), 3),
    _q("spring-rest-idempotency",
       "Design a POST endpoint for creating payments that is safe for clients to retry.",
       T.SPRING, ("spring.rest", "distributed.idempotency"), 4, "distributed.idempotency"),
    _q("spring-resilience",
       "A downstream dependency starts responding in 30 seconds instead of 30 "
       "milliseconds. What protects your service?",
       T.SPRING, ("spring.resilience", "system_design.fault_tolerance"), 4),
    _q("spring-testing-strategy",
       "How do you decide what to cover with @SpringBootTest versus a plain unit test?",
       T.SPRING, ("spring.testing",), 3),
    # --- Databases --------------------------------------------------------
    _q("db-index-composite",
       "You have an index on (customer_id, created_at). Which of these queries can use "
       "it, and why: filter by customer_id; filter by created_at; filter by both?",
       T.DATABASE, ("database.indexing",), 3, "database.indexing", asks_confidence=True),
    _q("db-index-cost",
       "A query is slow, so someone adds four indexes. Writes get slower and the query is "
       "unchanged. Explain both halves.",
       T.DATABASE, ("database.indexing", "database.query_optimization"), 4,
       "database.indexing"),
    _q("db-isolation-lost-update",
       "Two concurrent transactions read a balance, subtract from it, and write it back. "
       "What can go wrong, and how do you prevent it?",
       T.DATABASE, ("database.transactions.isolation", "database.locking"), 4,
       "database.transactions.isolation"),
    _q("db-isolation-default",
       "What isolation level does your production database run at, and what anomalies "
       "does that still permit?",
       T.DATABASE, ("database.transactions.isolation",), 3,
       "database.transactions.isolation"),
    _q("db-replica-lag",
       "You move reads to a replica and users start reporting that their own edit "
       "vanished. What is happening?",
       T.DATABASE, ("database.replication", "distributed.consistency"), 4),
    _q("db-schema-soft-delete",
       "Walk me through the tradeoffs of soft deletes versus hard deletes in a schema you "
       "own.",
       T.DATABASE, ("database.schema_design",), 3),
    _q("db-sharding-key",
       "You need to shard an orders table. How do you choose the shard key?",
       T.DATABASE, ("database.partitioning",), 4, level="senior"),
    _q("db-sql-vs-nosql",
       "When would you actually pick a document store over Postgres for a transactional "
       "workload?",
       T.DATABASE, ("database.nosql",), 3),
    # --- Distributed / Kafka ---------------------------------------------
    _q("kafka-ordering-scope",
       "What ordering guarantees does Kafka give you, and across what scope?",
       T.DISTRIBUTED, ("kafka.ordering",), 3, "kafka.ordering", asks_confidence=True),
    _q("kafka-ordering-account",
       "All events for one bank account must be processed in order, but you need "
       "throughput across millions of accounts. How do you arrange that?",
       T.DISTRIBUTED, ("kafka.ordering", "kafka.consumer_groups"), 4, "kafka.ordering"),
    _q("kafka-duplicate-events",
       "Your payment consumer occasionally processes the same event twice after a "
       "rebalance. Walk me through the fix.",
       T.DISTRIBUTED, ("kafka.delivery_semantics", "distributed.idempotency"), 4,
       "kafka.delivery_semantics", asks_confidence=True),
    _q("kafka-offset-commit",
       "Where in your consumer loop do you commit the offset, and what does the "
       "alternative cost you?",
       T.DISTRIBUTED, ("kafka.consumer_groups", "kafka.delivery_semantics"), 4,
       "kafka.delivery_semantics"),
    _q("kafka-poison-message",
       "One message in a partition fails every time it is processed. What happens to the "
       "rest of that partition?",
       T.DISTRIBUTED, ("kafka.delivery_semantics", "distributed.retries"), 4,
       "kafka.delivery_semantics"),
    _q("kafka-consumer-lag",
       "Consumer lag is growing steadily. Walk me through your diagnosis.",
       T.DISTRIBUTED, ("kafka.consumer_groups", "system_design.observability"), 3),
    _q("dist-idempotency-payment",
       "A payment API call times out. The client does not know whether the charge "
       "happened. Design the retry story.",
       T.DISTRIBUTED, ("distributed.idempotency", "distributed.retries"), 4,
       "distributed.idempotency"),
    _q("dist-outbox",
       "You must write to your database and publish an event, atomically. How?",
       T.DISTRIBUTED, ("distributed.transactions", "distributed.idempotency"), 5,
       level="senior"),
    _q("dist-retry-storm",
       "A downstream service degrades and your retries make it worse. What do you change?",
       T.DISTRIBUTED, ("distributed.retries", "system_design.fault_tolerance"), 4),
    _q("dist-cap-tradeoff",
       "For a shopping cart service, which would you sacrifice under a network "
       "partition - consistency or availability? Defend it.",
       T.DISTRIBUTED, ("distributed.consistency",), 4),
    _q("dist-distributed-lock",
       "Two instances of a scheduled job must not run simultaneously. How do you enforce "
       "that, and how does your solution fail?",
       T.DISTRIBUTED, ("distributed.concurrency", "distributed.consensus"), 5,
       level="senior"),
    # --- System design ----------------------------------------------------
    _q("sd-url-shortener",
       "Design a URL shortener. Start with the questions you would ask me.",
       T.SYSTEM_DESIGN, ("system_design.requirements", "system_design.api"), 3,
       "system_design.api"),
    _q("sd-payment-api",
       "Design a payment-processing service that consumes Kafka events and must never "
       "double-charge.",
       T.SYSTEM_DESIGN,
       ("system_design.api", "distributed.idempotency", "kafka.delivery_semantics"), 4,
       "system_design.api", level="senior"),
    _q("sd-rate-limiter",
       "Design a rate limiter for a public API running across 20 instances.",
       T.SYSTEM_DESIGN, ("system_design.rate_limiting", "system_design.caching"), 4,
       "system_design.api"),
    _q("sd-caching-layer",
       "You are adding Redis in front of a hot read path. Walk me through the design.",
       T.SYSTEM_DESIGN, ("system_design.caching",), 3, "system_design.caching",
       asks_confidence=True),
    _q("sd-cache-stampede",
       "A popular cache key expires and a thousand requests arrive in the same second. "
       "What happens?",
       T.SYSTEM_DESIGN, ("system_design.caching.invalidation",), 5,
       "system_design.caching"),
    _q("sd-notification-fanout",
       "Design a notification service that fans out to email, SMS, and push, where one "
       "channel being down must not block the others.",
       T.SYSTEM_DESIGN, ("system_design.microservices", "system_design.fault_tolerance"), 4),
    _q("sd-p99-doubled",
       "p99 latency on your main endpoint doubled overnight with no deploy. Walk me "
       "through the investigation.",
       T.SYSTEM_DESIGN, ("system_design.observability", "cloud.incident_response"), 4),
    _q("sd-10x-traffic",
       "Take the service you just described. What breaks first at 10x traffic?",
       T.SYSTEM_DESIGN, ("system_design.scaling",), 4, "system_design.api"),
    _q("sd-auth-service",
       "How would you handle authentication across a set of internal microservices?",
       T.SYSTEM_DESIGN, ("system_design.auth", "system_design.microservices"), 3),
    # --- DSA --------------------------------------------------------------
    _q("dsa-two-sum-family",
       "Given an array of integers and a target, return the indices of two numbers that "
       "add to the target. Talk me through your approach before you write anything.",
       T.DSA, ("dsa.arrays", "dsa.hashing", "dsa.complexity"), 2, "dsa.complexity",
       expects_code=True),
    _q("dsa-first-non-repeating",
       "Find the first non-repeating character in a string. What is your complexity, and "
       "can you do it in one pass?",
       T.DSA, ("dsa.arrays", "dsa.hashing"), 2, "dsa.complexity", expects_code=True),
    _q("dsa-lru-cache",
       "Implement an LRU cache with O(1) get and put. What structures do you need and why?",
       T.DSA, ("dsa.hashing", "dsa.linked_lists", "system_design.caching"), 4,
       "dsa.complexity", expects_code=True),
    _q("dsa-merge-intervals",
       "Given a list of intervals, merge the overlapping ones. What does sorting cost you?",
       T.DSA, ("dsa.intervals", "dsa.complexity"), 3, "dsa.complexity", expects_code=True),
    _q("dsa-graph-shortest-path",
       "You have a service dependency graph and need the shortest deployment order path "
       "between two services. Which algorithm, and why that one?",
       T.DSA, ("dsa.graphs", "dsa.complexity"), 4, "dsa.complexity"),
    _q("dsa-topological-order",
       "Given build dependencies between modules, produce a valid build order and detect "
       "cycles.",
       T.DSA, ("dsa.graphs",), 4, expects_code=True),
    _q("dsa-kth-largest",
       "Find the kth largest element in a large stream. Compare your options.",
       T.DSA, ("dsa.heaps", "dsa.complexity"), 3, "dsa.complexity"),
    _q("dsa-binary-search-rotated",
       "Search for a value in a rotated sorted array. Where does plain binary search break?",
       T.DSA, ("dsa.binary_search",), 3, expects_code=True),
    _q("dsa-sliding-window",
       "Find the longest substring without repeating characters. Start with brute force "
       "and improve it.",
       T.DSA, ("dsa.sliding_window", "dsa.complexity"), 3, "dsa.complexity",
       expects_code=True),
    _q("dsa-dp-coin-change",
       "Given coin denominations and an amount, find the fewest coins that make it. Why "
       "does greedy fail here?",
       T.DSA, ("dsa.dynamic_programming", "dsa.greedy"), 4, expects_code=True),
    # --- Cloud / DevOps ---------------------------------------------------
    _q("cloud-container-oom",
       "Your Java service gets OOM-killed in Kubernetes but heap dumps look fine. What is "
       "going on?",
       T.CLOUD, ("cloud.kubernetes", "java.jvm.gc"), 4, level="senior"),
    _q("cloud-zero-downtime",
       "Walk me through deploying a schema change and a code change that depend on each "
       "other, with zero downtime.",
       T.CLOUD, ("cloud.cicd", "database.schema_design"), 4),
    _q("cloud-incident-walkthrough",
       "Tell me about a production incident you were on point for. Start from the alert.",
       T.CLOUD, ("cloud.incident_response", "behavioral.ownership"), 3, "behavioral.star"),
    # --- Behavioural ------------------------------------------------------
    _q("beh-disagreement",
       "Tell me about a time you disagreed with a technical decision your team made. What "
       "did you do?",
       T.BEHAVIORAL, ("behavioral.conflict",), 2, "behavioral.star"),
    _q("beh-failure",
       "Tell me about something you shipped that failed. What happened?",
       T.BEHAVIORAL, ("behavioral.failure", "behavioral.ownership"), 2, "behavioral.star"),
    _q("beh-ambiguity",
       "Describe a project where the requirements were genuinely unclear. How did you "
       "proceed?",
       T.BEHAVIORAL, ("behavioral.ambiguity",), 3, "behavioral.star"),
    _q("beh-deadline-tradeoff",
       "Tell me about a time you had to ship something you were not happy with.",
       T.BEHAVIORAL, ("behavioral.tradeoffs", "behavioral.ownership"), 3, "behavioral.star"),
    _q("beh-difficult-stakeholder",
       "Tell me about a stakeholder who kept changing what they wanted. How did you handle it?",
       T.BEHAVIORAL, ("behavioral.collaboration",), 3, "behavioral.star"),
    # --- Hiring manager ---------------------------------------------------
    _q("hm-architecture-ownership",
       "Take me through the architecture of the most significant system you have owned. "
       "Why is it shaped that way?",
       T.HIRING_MANAGER, ("behavioral.leadership", "system_design.microservices"), 4,
       "resume.claim_defense"),
    _q("hm-biggest-tradeoff",
       "What is the most consequential technical tradeoff you have personally made?",
       T.HIRING_MANAGER, ("behavioral.tradeoffs",), 4, "resume.claim_defense"),
    _q("hm-influence",
       "Tell me about a time you changed the direction of a project without having "
       "authority over the people involved.",
       T.HIRING_MANAGER, ("behavioral.leadership",), 4, "behavioral.star"),
    _q("hm-what-would-you-change",
       "If you could redo one architectural decision from your last role, which and why?",
       T.HIRING_MANAGER, ("behavioral.tradeoffs", "behavioral.failure"), 4,
       "resume.claim_defense"),
)


QUESTIONS_BY_TYPE: dict[InterviewType, list[QuestionSeed]] = {}
for _seed in QUESTIONS:
    QUESTIONS_BY_TYPE.setdefault(_seed.interview_type, []).append(_seed)


def questions_for_concepts(
    concept_keys: set[str], interview_type: InterviewType | None = None
) -> list[QuestionSeed]:
    """Seeds touching any of ``concept_keys``, best-matching first."""
    scored: list[tuple[int, QuestionSeed]] = []
    for seed in QUESTIONS:
        if interview_type is not None and seed.interview_type is not interview_type:
            continue
        overlap = len(set(seed.concept_keys) & concept_keys)
        if overlap:
            scored.append((overlap, seed))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [seed for _, seed in scored]
