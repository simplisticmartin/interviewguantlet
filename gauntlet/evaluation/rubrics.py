"""Rubric library (spec section 18).

A rubric names the specific things a good answer contains, so grading is a set of
concrete judgements rather than one holistic guess. Three fields do the work:

* ``markers``  - surface forms the offline heuristic judge matches on. LLM judges read
  ``label`` and ``hint`` instead and are not limited to these strings.
* ``hint``     - an authored *probe question* for that dimension. When a dimension comes
  back missing, this is the adversarial follow-up the interviewer asks (spec section 5).
* ``common_misconceptions`` - known confidently-wrong beliefs and the phrasing that
  reveals them. This is what makes "low knowledge + high confidence" detectable rather
  than aspirational.

``negative_markers`` guard against false positives: a candidate who says "people think
ConcurrentHashMap is just a synchronized HashMap, but actually..." is demonstrating the
concept, not holding the misconception.
"""

from __future__ import annotations

from functools import lru_cache

from gauntlet.schemas import InterviewType, MisconceptionPattern, RubricDimension, RubricSpec


def _d(key: str, label: str, hint: str, markers: list[str], weight: float = 1.0) -> RubricDimension:
    return RubricDimension(key=key, label=label, hint=hint, markers=markers, weight=weight)


RUBRICS: tuple[RubricSpec, ...] = (
    RubricSpec(
        key="java.collections.hashmap",
        title="HashMap internals",
        concept_key="java.collections.hashmap",
        dimensions=[
            _d("hash_function", "Uses hashCode, spread/perturbation of bits",
               "How does the map turn a key's hashCode into a bucket position?",
               ["hashcode", "hash function", "spread", "perturb", "xor"]),
            _d("bucket_index", "Index derived from hash and table length",
               "Why is the table size a power of two, and what does that let it do "
               "instead of a modulo?",
               ["bucket", "index", "n-1 & hash", "power of two", "modulo"]),
            _d("equals", "equals is used to resolve within a bucket",
               "Two keys land in the same bucket. How does the map tell them apart?",
               ["equals", "equality", "same bucket"]),
            _d("collision_handling", "Collision strategy: chaining",
               "What happens when two different keys hash to the same bucket?",
               ["collision", "chaining", "chain"]),
            _d("linked_list", "Buckets start as linked nodes",
               "What data structure holds entries inside a single bucket?",
               ["linked list", "linked node", "node chain"]),
            _d("treeification", "Long chains become red-black trees",
               "What changes if one bucket accumulates a very large number of entries?",
               ["treeif", "red-black", "red black", "tree bin", "balanced tree"]),
            _d("resize", "Rehashing when the table grows",
               "What happens when the map outgrows its table, and what does that cost?",
               ["resize", "rehash", "double the", "grow the table"]),
            _d("load_factor", "Load factor governs resize threshold",
               "What decides the moment a resize happens?",
               ["load factor", "0.75", "threshold", "capacity"]),
            _d("time_complexity", "O(1) average, O(log n) / O(n) degenerate",
               "What is lookup cost in the average case versus the worst case, and what "
               "causes the worst case?",
               ["o(1)", "constant time", "average case", "worst case", "o(log n)", "o(n)"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="HashMap lookup is always O(1), including worst case.",
                correction=(
                    "Average case is O(1). With many colliding keys a bucket degrades to a "
                    "list (O(n)); modern JDKs treeify long bins to O(log n)."
                ),
                markers=["always o(1)", "guaranteed o(1)", "o(1) always", "constant time always"],
                negative_markers=["average", "worst case", "amortized"],
                severity=3,
                concept_key="java.collections.hashmap",
            ),
            MisconceptionPattern(
                belief="HashMap keys only need equals, or only need hashCode.",
                correction=(
                    "Both are required and must agree: equal objects must return equal "
                    "hash codes, or lookups silently miss."
                ),
                markers=["only need equals", "only needs hashcode", "just override equals",
                         "don't need hashcode"],
                severity=4,
                concept_key="java.collections.equals_hashcode",
            ),
            MisconceptionPattern(
                belief="HashMap is thread-safe.",
                correction=(
                    "HashMap is not synchronised. Concurrent writes can corrupt it; use "
                    "ConcurrentHashMap."
                ),
                markers=["hashmap is thread safe", "hashmap is thread-safe",
                         "hashmap handles concurrency"],
                negative_markers=["not thread safe", "is not thread-safe"],
                severity=5,
                concept_key="java.collections.hashmap",
            ),
        ],
    ),
    RubricSpec(
        key="java.concurrency.concurrent_hashmap",
        title="ConcurrentHashMap",
        concept_key="java.concurrency.concurrent_hashmap",
        dimensions=[
            _d("no_global_lock", "Does not serialise all access on one lock",
               "If a single lock covered the entire map, what would happen when several "
               "threads touched unrelated keys?",
               ["not a single lock", "no global lock", "per bucket", "per-bin", "fine grained",
                "fine-grained", "lock striping", "segment"]),
            _d("bin_level_locking", "Synchronises on the bin head / uses CAS",
               "What exactly does a writing thread lock, and what does that let other "
               "writers do meanwhile?",
               ["cas", "compare and swap", "synchronized on the bin", "bin head",
                "node lock", "lock the bucket"]),
            _d("lock_free_reads", "Reads are non-blocking",
               "Does a reader ever block behind a writer here?",
               ["reads are lock free", "lock-free read", "non-blocking read",
                "readers don't block", "volatile read"]),
            _d("weak_consistency", "Iterators are weakly consistent",
               "You iterate while another thread writes. What does the iterator show you, "
               "and can it throw?",
               ["weakly consistent", "weak consistency", "no concurrentmodification",
                "does not throw"]),
            _d("atomic_compound_ops", "compute/merge/putIfAbsent for compound updates",
               "How do you increment a counter stored in the map without losing updates?",
               ["computeifabsent", "compute", "merge", "putifabsent", "atomic operation"]),
            _d("null_handling", "Null keys and values are rejected",
               "What happens if you put a null value in?",
               ["null is not allowed", "no null", "nullpointerexception", "rejects null"]),
            _d("tradeoffs", "Costs versus a synchronized map",
               "When would you NOT reach for ConcurrentHashMap?",
               ["tradeoff", "trade-off", "memory overhead", "contention", "throughput"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="ConcurrentHashMap is basically a synchronized HashMap.",
                correction=(
                    "It locks per bin (with CAS on the fast path) rather than holding one "
                    "monitor for the whole map, so unrelated keys proceed in parallel and "
                    "reads do not block."
                ),
                markers=["basically a synchronized hashmap", "just a synchronized hashmap",
                         "same as synchronized hashmap", "wraps hashmap in synchronized",
                         "it synchronizes the whole map", "locks the whole map"],
                negative_markers=[
                    "not just", "unlike synchronized", "people think", "misconception",
                    "common myth",
                ],
                severity=4,
                concept_key="java.concurrency.concurrent_hashmap",
            ),
            MisconceptionPattern(
                belief="Compound operations on ConcurrentHashMap are automatically atomic.",
                correction=(
                    "Individual operations are atomic; get-then-put is not. Use compute, "
                    "merge, or putIfAbsent for read-modify-write."
                ),
                markers=["everything is atomic", "all operations are atomic",
                         "get then put is safe", "check then put is atomic"],
                severity=4,
                concept_key="java.concurrency.concurrent_hashmap",
            ),
        ],
    ),
    RubricSpec(
        key="java.concurrency.memory_model",
        title="Java memory model and visibility",
        concept_key="java.concurrency.memory_model",
        dimensions=[
            _d("visibility", "Visibility is distinct from atomicity",
               "A field is written by one thread and read by another with no synchronisation. "
               "What can go wrong even if the write itself is atomic?",
               ["visibility", "visible to other threads", "stale value", "cached in a register"]),
            _d("happens_before", "happens-before ordering",
               "What actually guarantees the reader sees the writer's earlier writes?",
               ["happens before", "happens-before", "ordering guarantee"]),
            _d("volatile_semantics", "volatile gives visibility and ordering, not atomicity",
               "Is `volatile int count; count++` thread-safe?",
               ["volatile", "not atomic", "read-modify-write", "compound operation"]),
            _d("reordering", "Compiler and CPU reordering",
               "Who is allowed to reorder your statements, and why would they?",
               ["reorder", "reordering", "instruction reorder", "out of order"]),
            _d("synchronization_alternatives", "Locks, atomics, and concurrent collections",
               "What would you use instead when you need atomic increment?",
               ["atomicinteger", "atomic", "synchronized", "lock", "longadder"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="volatile makes operations atomic / makes a class thread-safe.",
                correction=(
                    "volatile provides visibility and ordering only. Read-modify-write such "
                    "as ++ still races; use AtomicInteger or a lock."
                ),
                markers=["volatile makes it atomic", "volatile is atomic",
                         "volatile makes it thread safe", "volatile makes it thread-safe",
                         "volatile prevents race"],
                negative_markers=["not atomic", "does not make it atomic", "only visibility"],
                severity=4,
                concept_key="java.concurrency.volatile",
            ),
        ],
    ),
    RubricSpec(
        key="java.jvm.gc",
        title="Garbage collection",
        concept_key="java.jvm.gc",
        dimensions=[
            _d("reachability", "Collection is by reachability, not reference counting",
               "How does the JVM decide an object is garbage?",
               ["reachab", "gc root", "gc roots", "unreachable"]),
            _d("generational", "Generational hypothesis, young and old gen",
               "Why does the heap have separate young and old regions?",
               ["young generation", "old generation", "generational", "eden", "survivor",
                "tenured"]),
            _d("pause_behaviour", "Stop-the-world pauses and their impact",
               "Your p99 latency spikes every few minutes. How would you tell whether GC "
               "is responsible?",
               ["stop the world", "stop-the-world", "pause", "gc log", "latency spike"]),
            _d("collector_choice", "G1 / ZGC / Parallel and their tradeoffs",
               "Which collector would you pick for a low-latency service, and what do you "
               "give up?",
               ["g1", "zgc", "shenandoah", "parallel gc", "cms"]),
            _d("tuning_limits", "Leaks are not fixed by tuning",
               "Heap keeps growing until OOM. Is this a tuning problem?",
               ["memory leak", "heap dump", "retained", "oom", "out of memory"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Calling System.gc() forces a garbage collection.",
                correction="System.gc() is a hint the JVM is free to ignore.",
                markers=["system.gc forces", "calling system.gc will collect",
                         "system.gc() forces", "force garbage collection with system.gc"],
                negative_markers=["only a hint", "is a hint", "not guaranteed", "may ignore"],
                severity=3,
                concept_key="java.jvm.gc",
            ),
            MisconceptionPattern(
                belief="Java cannot have memory leaks because it has GC.",
                correction=(
                    "Unbounded caches, static collections, and unremoved listeners keep "
                    "objects reachable forever - a leak in every practical sense."
                ),
                markers=["java can't have memory leaks", "no memory leaks in java",
                         "gc prevents memory leaks", "java cannot leak"],
                severity=4,
                concept_key="java.jvm.gc",
            ),
        ],
    ),
    RubricSpec(
        key="spring.di",
        title="Spring dependency injection",
        concept_key="spring.di",
        dimensions=[
            _d("container_role", "The container owns construction and wiring",
               "Who creates your beans, and when?",
               ["application context", "ioc container", "container", "bean factory"]),
            _d("injection_styles", "Constructor injection preferred over field injection",
               "Constructor or field injection - which do you use, and why?",
               ["constructor injection", "field injection", "setter injection"]),
            _d("testability", "Explicit dependencies make testing possible",
               "How does your choice affect writing a unit test without Spring?",
               ["testab", "unit test", "mock", "without spring"]),
            _d("scopes", "Default singleton scope and its implications",
               "Two controllers depend on the same service. Do they get the same instance?",
               ["singleton", "scope", "prototype", "request scope"]),
            _d("proxying", "Proxies underpin AOP behaviour",
               "How does Spring add transactional behaviour to a plain class?",
               ["proxy", "cglib", "jdk dynamic proxy", "aop"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Singleton-scoped beans are thread-safe.",
                correction=(
                    "Singleton means one instance shared across all threads - mutable state "
                    "in a singleton bean is precisely a concurrency bug."
                ),
                markers=["singleton beans are thread safe", "singleton is thread-safe",
                         "spring handles thread safety", "beans are thread safe"],
                negative_markers=["not thread safe", "shared mutable state is a problem"],
                severity=5,
                concept_key="spring.bean_scopes",
            ),
        ],
    ),
    RubricSpec(
        key="spring.transactions",
        title="Spring transaction management",
        concept_key="spring.transactions",
        dimensions=[
            _d("proxy_mechanics", "@Transactional works via a proxy",
               "Mechanically, how does @Transactional start and commit a transaction?",
               ["proxy", "aop", "interceptor", "cglib"]),
            _d("self_invocation", "Self-invocation bypasses the proxy",
               "A public method in the same class calls another annotated method directly. "
               "Is the second one transactional?",
               ["self invocation", "self-invocation", "internal call", "same class",
                "bypasses the proxy", "this."]),
            _d("propagation", "Propagation semantics",
               "What does REQUIRES_NEW do differently from REQUIRED here?",
               ["propagation", "requires_new", "required", "nested"]),
            _d("rollback_rules", "Rollback on unchecked exceptions by default",
               "You throw a checked exception. Does the transaction roll back?",
               ["runtimeexception", "unchecked", "checked exception", "rollbackfor",
                "does not roll back"]),
            _d("boundaries", "Transaction scope should match the unit of work",
               "Where should the transaction boundary sit relative to your HTTP call to "
               "a third party?",
               ["boundary", "unit of work", "keep it short", "don't call external"]),
            _d("isolation_link", "Isolation level and locking implications",
               "What isolation level are you actually running at, and what does it permit?",
               ["isolation", "read committed", "repeatable read", "serializable"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="@Transactional applies to calls made within the same class.",
                correction=(
                    "Proxy-based AOP only intercepts calls arriving from outside the bean. "
                    "Self-invocation silently runs with no transaction."
                ),
                markers=["works for internal calls", "same class calls are transactional",
                         "self invocation works", "calling it from the same class works"],
                negative_markers=["does not work", "won't work", "bypasses"],
                severity=5,
                concept_key="spring.transactions",
            ),
            MisconceptionPattern(
                belief="@Transactional rolls back on any exception.",
                correction=(
                    "By default only unchecked exceptions and Errors trigger rollback; "
                    "checked exceptions commit unless you set rollbackFor."
                ),
                markers=["rolls back on any exception", "any exception rolls back",
                         "all exceptions roll back", "rollback on every exception"],
                negative_markers=["only runtime", "unchecked only", "rollbackfor"],
                severity=4,
                concept_key="spring.transactions",
            ),
        ],
    ),
    RubricSpec(
        key="database.indexing",
        title="Database indexing",
        concept_key="database.indexing",
        dimensions=[
            _d("structure", "B-tree structure and ordering",
               "What is actually stored in a typical index, and in what order?",
               ["b-tree", "btree", "balanced tree", "sorted", "leaf node"]),
            _d("write_cost", "Indexes slow writes and consume space",
               "What does adding this index cost you?",
               ["slows down writes", "write cost", "insert cost", "storage", "overhead"]),
            _d("composite_order", "Leftmost-prefix rule for composite indexes",
               "You have an index on (a, b). Does a query filtering only on b use it?",
               ["composite", "leftmost", "left-most", "column order", "prefix"]),
            _d("selectivity", "Selectivity/cardinality drives usefulness",
               "When would the planner ignore your index entirely?",
               ["selectivity", "cardinality", "low cardinality", "planner", "optimizer"]),
            _d("covering", "Covering indexes avoid table lookups",
               "How would you avoid the row lookup after the index seek?",
               ["covering index", "index only scan", "include", "index-only"]),
            _d("verification", "EXPLAIN verifies rather than assumes",
               "How do you confirm the index is being used?",
               ["explain", "execution plan", "query plan", "analyze"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Adding more indexes always makes the database faster.",
                correction=(
                    "Every index is maintained on write and consumes space; excess indexes "
                    "slow inserts and can mislead the planner."
                ),
                markers=["more indexes is always", "always add an index",
                         "indexes always make it faster", "index everything"],
                negative_markers=["slows down writes", "write cost", "tradeoff"],
                severity=3,
                concept_key="database.indexing",
            ),
            MisconceptionPattern(
                belief="A composite index on (a, b) serves a query filtering on b alone.",
                correction=(
                    "Composite indexes are usable left-to-right; filtering on b alone "
                    "generally cannot use the index."
                ),
                markers=["order doesn't matter", "order does not matter",
                         "works for either column", "any column in the index"],
                severity=4,
                concept_key="database.indexing",
            ),
        ],
    ),
    RubricSpec(
        key="database.transactions.isolation",
        title="Transaction isolation levels",
        concept_key="database.transactions.isolation",
        dimensions=[
            _d("anomalies", "Names the anomalies: dirty/non-repeatable/phantom",
               "Which specific anomaly are you preventing?",
               ["dirty read", "non-repeatable", "phantom", "lost update"]),
            _d("levels", "Maps levels to permitted anomalies",
               "What does READ COMMITTED still allow?",
               ["read committed", "repeatable read", "serializable", "read uncommitted"]),
            _d("default_awareness", "Knows the engine default",
               "What is the default in the database you use, and did you choose it?",
               ["default is read committed", "postgres default", "mysql default",
                "repeatable read is the default"]),
            _d("cost", "Stronger isolation costs concurrency",
               "What do you pay for SERIALIZABLE?",
               ["locking", "contention", "throughput", "serialization failure", "retry"]),
            _d("application_level", "Optimistic concurrency as an alternative",
               "How would you prevent lost updates without escalating isolation?",
               ["optimistic locking", "version column", "compare and set", "select for update"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="SERIALIZABLE just means transactions run one at a time.",
                correction=(
                    "It guarantees the *result* is equivalent to some serial order; engines "
                    "achieve it with predicate locks or SSI, not literal serial execution."
                ),
                markers=["runs one at a time", "one transaction at a time",
                         "executes sequentially", "literally serial"],
                severity=3,
                concept_key="database.transactions.isolation",
            ),
        ],
    ),
    RubricSpec(
        key="kafka.ordering",
        title="Kafka ordering guarantees",
        concept_key="kafka.ordering",
        dimensions=[
            _d("partition_scope", "Ordering holds per partition, not per topic",
               "Across how much of the topic does that ordering guarantee actually hold?",
               ["per partition", "within a partition", "partition level", "not across partitions"]),
            _d("key_routing", "Partition key determines co-location",
               "How do you make sure all events for one account stay ordered?",
               ["partition key", "message key", "same key", "hash of the key"]),
            _d("consumer_parallelism", "One partition per consumer in a group",
               "What limits how many consumers can usefully work in parallel?",
               ["consumer group", "one consumer per partition", "parallelism", "rebalance"]),
            _d("retry_impact", "Retries and async sends can reorder",
               "A send fails and is retried. What happens to ordering?",
               ["retry", "max.in.flight", "in flight", "reorder", "idempotent producer"]),
            _d("ordering_cost", "Ordering constrains throughput and scaling",
               "What does preserving strict ordering cost you at scale?",
               ["throughput", "hot partition", "scaling", "bottleneck", "skew"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Kafka guarantees ordering across the whole topic.",
                correction=(
                    "Ordering is guaranteed only within a partition. Cross-partition order "
                    "is undefined; co-locate related events with a partition key."
                ),
                markers=["kafka guarantees ordering", "ordering across the topic",
                         "topic level ordering", "guarantees global ordering",
                         "messages are always in order", "kafka is always ordered"],
                negative_markers=["per partition", "within a partition", "not across partitions"],
                severity=5,
                concept_key="kafka.ordering",
            ),
        ],
    ),
    RubricSpec(
        key="kafka.delivery_semantics",
        title="Kafka delivery semantics and idempotency",
        concept_key="kafka.delivery_semantics",
        dimensions=[
            _d("default_semantics", "At-least-once is the practical default",
               "By default, can your consumer see the same message twice?",
               ["at least once", "at-least-once", "duplicate", "twice"]),
            _d("offset_commit", "Commit timing determines duplicates vs loss",
               "When exactly do you commit the offset, and what does that choice trade?",
               ["offset", "commit", "auto commit", "manual commit", "after processing"]),
            _d("idempotent_consumer", "Consumer-side dedup makes reprocessing safe",
               "Your consumer crashes after writing to the database but before committing. "
               "What happens on restart?",
               ["idempot", "dedup", "deduplicat", "unique constraint", "processed id",
                "upsert"]),
            _d("exactly_once_scope", "EOS is bounded to Kafka transactions",
               "Exactly-once - across which systems, exactly?",
               ["transactional", "exactly once", "eos", "kafka transaction",
                "read committed"]),
            _d("external_effects", "External side effects need their own idempotency",
               "You also charge a credit card in that handler. Does exactly-once help?",
               ["external system", "side effect", "outbox", "idempotency key", "payment"]),
            _d("dlq", "Poison messages need a dead-letter path",
               "One message fails forever. What happens to the partition?",
               ["dead letter", "dlq", "poison", "skip", "retry topic"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Exactly-once semantics means a message can never be processed twice.",
                correction=(
                    "Kafka EOS covers Kafka-to-Kafka processing within a transaction. Side "
                    "effects in external systems still need idempotency keys."
                ),
                markers=["exactly once means it can never", "exactly-once solves duplicates",
                         "eos means no duplicates ever", "exactly once handles everything"],
                negative_markers=["external system", "side effect", "only within kafka"],
                severity=4,
                concept_key="kafka.delivery_semantics",
            ),
        ],
    ),
    RubricSpec(
        key="distributed.idempotency",
        title="Idempotency in distributed systems",
        concept_key="distributed.idempotency",
        dimensions=[
            _d("definition", "Repeating the operation does not change the outcome",
               "State precisely what idempotent means for a write operation.",
               ["same result", "no additional effect", "repeat", "twice", "idempotent"]),
            _d("idempotency_key", "Client-supplied key identifies the logical operation",
               "Two identical requests arrive. How do you tell a retry from a genuine "
               "second purchase?",
               ["idempotency key", "request id", "client token", "unique key",
                "correlation id"]),
            _d("storage", "The key must be recorded durably and atomically",
               "Where does that key live, and what happens if you crash between recording "
               "it and doing the work?",
               ["unique constraint", "database", "atomically", "same transaction", "upsert"]),
            _d("response_replay", "Replayed requests return the original response",
               "What do you return when the same key arrives again?",
               ["return the same response", "cached response", "stored response",
                "replay the result"]),
            _d("expiry", "Keys need a retention policy",
               "How long do you keep those keys?",
               ["ttl", "expire", "retention", "clean up"]),
            _d("http_semantics", "PUT/DELETE naturally idempotent, POST is not",
               "Which HTTP verbs give you this for free?",
               ["put", "delete", "post is not", "http method"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Retrying with the same request body makes an operation idempotent.",
                correction=(
                    "Identical payloads are indistinguishable from a legitimate duplicate "
                    "action. You need an explicit idempotency key recorded atomically with "
                    "the effect."
                ),
                markers=["same request body", "same payload means", "compare the payload",
                         "if the data is the same"],
                severity=4,
                concept_key="distributed.idempotency",
            ),
        ],
    ),
    RubricSpec(
        key="system_design.caching",
        title="Caching strategy",
        concept_key="system_design.caching",
        dimensions=[
            _d("pattern", "Names a strategy: cache-aside, read-through, write-through",
               "Which caching pattern, and who populates the cache?",
               ["cache aside", "cache-aside", "read through", "write through", "write behind"]),
            _d("invalidation", "Has an invalidation and staleness story",
               "The underlying row changes. How stale can the cache get?",
               ["invalidat", "ttl", "expire", "stale", "eviction"]),
            _d("consistency", "Acknowledges cache/database divergence",
               "What does a client see if the write succeeds but the invalidation fails?",
               ["inconsistent", "consistency", "diverge", "out of sync"]),
            _d("stampede", "Handles thundering herd on expiry",
               "A hot key expires and a thousand requests arrive at once. What happens?",
               ["stampede", "thundering herd", "dogpile", "lock", "single flight",
                "jitter"]),
            _d("hit_ratio", "Reasons about hit ratio and what to cache",
               "How would you decide this cache is worth its complexity?",
               ["hit ratio", "hit rate", "miss rate", "measure"]),
            _d("sizing", "Eviction policy and memory bounds",
               "What happens when the cache fills up?",
               ["lru", "lfu", "eviction", "memory limit", "max size"]),
        ],
        common_misconceptions=[
            MisconceptionPattern(
                belief="Adding a cache is a safe optimisation with no consistency cost.",
                correction=(
                    "A cache introduces a second copy of the truth. Every cache decision is "
                    "a staleness decision."
                ),
                markers=["cache has no downside", "caching is always", "no downside to caching",
                         "always add a cache"],
                negative_markers=["stale", "invalidat", "tradeoff"],
                severity=3,
                concept_key="system_design.caching",
            ),
        ],
    ),
    RubricSpec(
        key="system_design.api",
        title="API and service design",
        concept_key="system_design.api",
        dimensions=[
            _d("requirements", "Clarifies requirements and scale before designing",
               "Before designing - what are the read and write volumes?",
               ["requirement", "qps", "rps", "scale", "how many users", "clarify"]),
            _d("contract", "Concrete endpoints, payloads, and status codes",
               "What does the request and response actually look like?",
               ["endpoint", "post /", "get /", "status code", "payload", "schema"]),
            _d("data_model", "Storage choice justified",
               "What is the data model, and why that store?",
               ["schema", "table", "primary key", "index", "database choice"]),
            _d("failure_modes", "Handles partial failure and retries",
               "The downstream call times out. What does the caller see?",
               ["timeout", "retry", "failure", "circuit breaker", "fallback"]),
            _d("scaling", "Identifies the bottleneck under growth",
               "At 10x traffic, what breaks first?",
               ["bottleneck", "scale out", "shard", "replica", "queue", "10x"]),
            _d("observability", "Names what to measure",
               "How would you know this is unhealthy in production?",
               ["metric", "alert", "p99", "latency", "dashboard", "trace"]),
        ],
    ),
    RubricSpec(
        key="dsa.complexity",
        title="Complexity analysis",
        concept_key="dsa.complexity",
        dimensions=[
            _d("time", "States time complexity correctly",
               "What is the time complexity, and where does it come from?",
               ["o(n)", "o(1)", "o(log n)", "o(n log n)", "o(n^2)", "time complexity"]),
            _d("space", "States auxiliary space",
               "How much extra memory does that use?",
               ["space complexity", "extra space", "auxiliary", "in place", "o(1) space"]),
            _d("derivation", "Explains why, not just the label",
               "Walk me through where that bound comes from.",
               ["because", "each element", "iterations", "recurrence", "halves"]),
            _d("worst_vs_average", "Distinguishes average from worst case",
               "Is that the worst case or the average case?",
               ["worst case", "average case", "amortized", "best case"]),
            _d("tradeoff", "Articulates the time/space tradeoff taken",
               "What did you trade to get that?",
               ["tradeoff", "trade-off", "in exchange", "at the cost of"]),
        ],
    ),
    RubricSpec(
        key="behavioral.star",
        title="Behavioural answer quality",
        concept_key="behavioral",
        dimensions=[
            _d("situation", "Concrete, specific situation",
               "What was the actual situation - team, timeline, stakes?",
               ["we were", "the team", "at the time", "project", "last year"]),
            _d("personal_action", "What THEY did, not what the team did",
               "What did you personally do, as distinct from the team?",
               ["i decided", "i built", "i proposed", "i led", "i wrote", "i escalated"]),
            _d("reasoning", "Why they chose that action over alternatives",
               "What else did you consider, and why did you rule it out?",
               ["because", "the alternative", "instead of", "we considered", "tradeoff"]),
            _d("outcome", "Measurable or concrete outcome",
               "How did it turn out, and how did you know?",
               ["result", "outcome", "reduced", "improved", "shipped", "%"]),
            _d("reflection", "What they learned or would change",
               "What would you do differently now?",
               ["learned", "in hindsight", "would do differently", "next time"]),
            _d("ownership", "Takes responsibility rather than assigning blame",
               "What part of that was yours to own?",
               ["my mistake", "i should have", "my responsibility", "i owned"]),
        ],
    ),
    RubricSpec(
        key="resume.claim_defense",
        title="Resume claim depth",
        concept_key="behavioral.tradeoffs",
        dimensions=[
            _d("measurement", "Can explain how the result was measured",
               "How did you measure that?",
               ["measured", "metric", "monitoring", "dashboard", "benchmark", "p99",
                "baseline"]),
            _d("root_cause", "Knows the underlying bottleneck or mechanism",
               "What was actually the bottleneck?",
               ["bottleneck", "root cause", "profil", "the problem was", "contention",
                "n+1"]),
            _d("causality", "Establishes the change caused the outcome",
               "How do you know your change caused the improvement rather than something "
               "else?",
               ["before and after", "a/b", "control", "isolated", "rolled back",
                "canary"]),
            _d("tradeoffs", "Names what the change cost",
               "What did that optimisation make worse?",
               ["tradeoff", "trade-off", "downside", "cost", "complexity", "memory"]),
            _d("scaling", "Can reason about the change at 10x",
               "What happens to that approach at 10x traffic?",
               ["10x", "scale", "would break", "saturate", "bottleneck"]),
            _d("ownership_depth", "Personal involvement is evident in specifics",
               "Which part of that did you build yourself?",
               ["i implemented", "i wrote", "i designed", "i profiled", "i deployed"]),
        ],
    ),
)


_GENERIC_DIMENSIONS = [
    _d("correctness", "Technically accurate",
       "Can you state that more precisely?",
       []),
    _d("depth", "Goes beyond a definition into mechanism",
       "How does that actually work underneath?",
       ["because", "internally", "under the hood", "mechanism"]),
    _d("tradeoffs", "Names tradeoffs and limits",
       "When would that be the wrong choice?",
       ["tradeoff", "trade-off", "downside", "limitation", "instead"]),
    _d("experience", "Grounded in real usage",
       "Where have you actually used this?",
       ["we used", "in production", "i built", "at my", "our system"]),
    _d("edge_cases", "Considers failure and edge cases",
       "What breaks first when that assumption fails?",
       ["edge case", "fails", "what if", "null", "empty", "timeout"]),
]


@lru_cache(maxsize=1)
def rubric_index() -> dict[str, RubricSpec]:
    return {rubric.key: rubric for rubric in RUBRICS}


@lru_cache(maxsize=64)
def generic_rubric(interview_type: InterviewType) -> RubricSpec:
    """Fallback rubric for concepts without an authored one.

    Deliberately coarse. It keeps grading structured rather than holistic, and the
    absence of an authored rubric is visible in the report rather than hidden.
    """
    return RubricSpec(
        key=f"generic.{interview_type.value}",
        title=f"General {interview_type.value.replace('_', ' ')} answer quality",
        concept_key=None,
        dimensions=list(_GENERIC_DIMENSIONS),
    )


def get_rubric(key: str | None, interview_type: InterviewType) -> RubricSpec:
    if key:
        found = rubric_index().get(key)
        if found is not None:
            return found
    return generic_rubric(interview_type)


def misconception_candidates(
    concept_keys: list[str], rubric: RubricSpec | None = None
) -> list[MisconceptionPattern]:
    """Misconception patterns worth checking for on this question.

    Not just the current rubric's. Candidates volunteer confidently-wrong beliefs about
    adjacent topics all the time - someone asked about offset commits will happily assert
    that Kafka orders the whole topic - and catching that is the single highest-value
    thing this product does. Scope is the concept's top-level domain, which keeps the
    candidate set relevant and bounded rather than scanning every rubric we ship.
    """
    patterns: list[MisconceptionPattern] = list(rubric.common_misconceptions) if rubric else []
    roots = {key.split(".")[0] for key in concept_keys if key}

    for spec in RUBRICS:
        if rubric is not None and spec.key == rubric.key:
            continue
        spec_root = (spec.concept_key or spec.key).split(".")[0]
        if spec_root in roots:
            patterns.extend(spec.common_misconceptions)

    seen: list[MisconceptionPattern] = []
    beliefs: set[str] = set()
    for pattern in patterns:
        if pattern.belief not in beliefs:
            beliefs.add(pattern.belief)
            seen.append(pattern)
    return seen


def rubric_for_concept(concept_key: str, interview_type: InterviewType) -> RubricSpec:
    """Exact concept match first, then the nearest ancestor with an authored rubric."""
    index = rubric_index()
    if concept_key in index:
        return index[concept_key]
    parts = concept_key.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        ancestor = ".".join(parts[:cut])
        if ancestor in index:
            return index[ancestor]
    return generic_rubric(interview_type)
