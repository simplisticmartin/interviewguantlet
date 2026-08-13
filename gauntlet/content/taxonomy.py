"""The concept taxonomy (spec section 21).

Keys are dotted paths, so the tree is implied by the key itself and can never
disagree with a separate parent pointer: ``java.concurrency.concurrent_hashmap``
is a child of ``java.concurrency`` by construction.

``deeper`` names the concepts an interviewer descends into after a strong answer -
this is the literal edge set the adaptive router walks (spec section 2):

    HashMap -> ConcurrentHashMap -> memory visibility -> distributed concurrency

Coverage here is deliberately deepest for Java backend, which is the first
end-to-end target. Other domains carry enough structure to extend into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from gauntlet.schemas import InterviewType


@dataclass(frozen=True, slots=True)
class ConceptDef:
    key: str
    display_name: str
    domain: str
    interview_type: InterviewType
    aliases: tuple[str, ...] = ()
    difficulty_floor: int = 1
    difficulty_ceiling: int = 5
    deeper: tuple[str, ...] = field(default=())

    @property
    def parent_key(self) -> str | None:
        if "." not in self.key:
            return None
        return self.key.rsplit(".", 1)[0]


def _c(
    key: str,
    display_name: str,
    domain: str,
    interview_type: InterviewType,
    *,
    aliases: tuple[str, ...] = (),
    floor: int = 1,
    ceiling: int = 5,
    deeper: tuple[str, ...] = (),
) -> ConceptDef:
    return ConceptDef(
        key=key,
        display_name=display_name,
        domain=domain,
        interview_type=interview_type,
        aliases=aliases,
        difficulty_floor=floor,
        difficulty_ceiling=ceiling,
        deeper=deeper,
    )


T = InterviewType

CONCEPTS: tuple[ConceptDef, ...] = (
    # --- Java -------------------------------------------------------------
    _c("java", "Java", "language", T.JAVA, aliases=("java 8", "java 11", "java 17", "java 21")),
    _c("java.collections", "Java Collections", "language", T.JAVA,
       aliases=("collections framework",), deeper=("java.collections.hashmap",)),
    _c("java.collections.hashmap", "HashMap", "language", T.JAVA,
       aliases=("hash map", "hashmap"), floor=2,
       deeper=("java.concurrency.concurrent_hashmap", "java.collections.equals_hashcode")),
    _c("java.collections.arraylist", "ArrayList", "language", T.JAVA,
       aliases=("array list",), deeper=("java.collections.hashmap",)),
    _c("java.collections.equals_hashcode", "equals and hashCode contract", "language", T.JAVA,
       aliases=("hashcode", "equals contract"), floor=2),
    _c("java.generics", "Generics", "language", T.JAVA,
       aliases=("type erasure", "wildcards"), floor=3),
    _c("java.concurrency", "Java Concurrency", "language", T.JAVA,
       aliases=("multithreading", "threads"), floor=3,
       deeper=("java.concurrency.memory_model", "java.concurrency.locks")),
    _c("java.concurrency.concurrent_hashmap", "ConcurrentHashMap", "language", T.JAVA,
       aliases=("concurrent hash map",), floor=3,
       deeper=("java.concurrency.memory_model", "distributed.concurrency")),
    _c("java.concurrency.synchronized", "synchronized", "language", T.JAVA, floor=2,
       deeper=("java.concurrency.locks", "java.concurrency.memory_model")),
    _c("java.concurrency.volatile", "volatile", "language", T.JAVA, floor=3,
       deeper=("java.concurrency.memory_model",)),
    _c("java.concurrency.locks", "Locks and ReentrantLock", "language", T.JAVA,
       aliases=("reentrantlock", "lock striping"), floor=3,
       deeper=("java.concurrency.memory_model",)),
    _c("java.concurrency.memory_model", "Java Memory Model", "language", T.JAVA,
       aliases=("jmm", "happens-before", "memory visibility"), floor=4,
       deeper=("distributed.concurrency",)),
    _c("java.concurrency.executors", "Executors and thread pools", "language", T.JAVA,
       aliases=("thread pool", "executorservice"), floor=3),
    _c("java.jvm", "JVM internals", "language", T.JAVA, aliases=("jvm",), floor=3,
       deeper=("java.jvm.gc",)),
    _c("java.jvm.gc", "Garbage collection", "language", T.JAVA,
       aliases=("garbage collector", "g1", "zgc"), floor=3),
    _c("java.streams", "Streams API", "language", T.JAVA,
       aliases=("stream api", "streams"), floor=2),
    _c("java.optional", "Optional", "language", T.JAVA, floor=2),
    _c("java.exceptions", "Exception handling", "language", T.JAVA,
       aliases=("checked exceptions",), floor=2),
    _c("java.records", "Records and sealed types", "language", T.JAVA,
       aliases=("record", "sealed"), floor=3),
    _c("java.virtual_threads", "Virtual threads", "language", T.JAVA,
       aliases=("project loom", "loom"), floor=4),
    # --- Spring -----------------------------------------------------------
    _c("spring", "Spring", "framework", T.SPRING, aliases=("spring framework",)),
    _c("spring.boot", "Spring Boot", "framework", T.SPRING, aliases=("springboot",),
       deeper=("spring.di", "spring.transactions")),
    _c("spring.di", "Dependency injection", "framework", T.SPRING,
       aliases=("inversion of control", "ioc", "autowired"), floor=2,
       deeper=("spring.bean_scopes",)),
    _c("spring.bean_scopes", "Bean scopes and lifecycle", "framework", T.SPRING,
       aliases=("singleton bean", "prototype scope"), floor=3),
    _c("spring.rest", "Spring REST controllers", "framework", T.SPRING,
       aliases=("restcontroller", "rest api"), floor=2),
    _c("spring.transactions", "Spring transactions", "framework", T.SPRING,
       aliases=("transactional", "@transactional"), floor=3,
       deeper=("database.transactions.isolation", "spring.transactions.propagation")),
    _c("spring.transactions.propagation", "Transaction propagation", "framework", T.SPRING,
       aliases=("requires_new", "propagation"), floor=4),
    _c("spring.data", "Spring Data JPA", "framework", T.SPRING,
       aliases=("jpa", "hibernate", "spring data"), floor=3,
       deeper=("spring.data.n_plus_one", "database.indexing")),
    _c("spring.data.n_plus_one", "N+1 query problem", "framework", T.SPRING,
       aliases=("n+1", "lazy loading"), floor=3),
    _c("spring.security", "Spring Security", "framework", T.SPRING,
       aliases=("security filter chain",), floor=3),
    _c("spring.testing", "Spring testing", "framework", T.SPRING,
       aliases=("mockmvc", "springboottest"), floor=2),
    _c("spring.resilience", "Resilience patterns", "framework", T.SPRING,
       aliases=("circuit breaker", "resilience4j", "retry"), floor=3),
    # --- Databases --------------------------------------------------------
    _c("database", "Databases", "data", T.DATABASE, aliases=("sql", "rdbms")),
    _c("database.indexing", "Indexing", "data", T.DATABASE,
       aliases=("index", "b-tree", "covering index"), floor=2,
       deeper=("database.query_optimization",)),
    _c("database.query_optimization", "Query optimisation", "data", T.DATABASE,
       aliases=("execution plan", "explain plan", "query plan"), floor=3),
    _c("database.normalization", "Normalisation", "data", T.DATABASE,
       aliases=("normal form", "denormalization"), floor=2),
    _c("database.transactions", "Transactions", "data", T.DATABASE,
       aliases=("acid",), floor=2, deeper=("database.transactions.isolation",)),
    _c("database.transactions.isolation", "Isolation levels", "data", T.DATABASE,
       aliases=("read committed", "repeatable read", "serializable", "phantom read"), floor=3,
       deeper=("distributed.consistency",)),
    _c("database.locking", "Database locking", "data", T.DATABASE,
       aliases=("pessimistic locking", "optimistic locking", "deadlock"), floor=3),
    _c("database.schema_design", "Schema design", "data", T.DATABASE, floor=2),
    _c("database.partitioning", "Partitioning and sharding", "data", T.DATABASE,
       aliases=("shard", "partition key"), floor=4, deeper=("distributed.consistency",)),
    _c("database.replication", "Replication", "data", T.DATABASE,
       aliases=("read replica", "replica lag"), floor=3, deeper=("distributed.consistency",)),
    _c("database.nosql", "Relational vs NoSQL", "data", T.DATABASE,
       aliases=("mongodb", "dynamodb", "cassandra"), floor=2),
    # --- Distributed systems ---------------------------------------------
    _c("distributed", "Distributed systems", "distributed", T.DISTRIBUTED),
    _c("kafka", "Kafka", "distributed", T.DISTRIBUTED, aliases=("apache kafka",), floor=2,
       deeper=("kafka.ordering", "kafka.consumer_groups")),
    _c("kafka.ordering", "Kafka ordering guarantees", "distributed", T.DISTRIBUTED,
       aliases=("partition ordering", "message ordering"), floor=3,
       deeper=("kafka.consumer_groups", "distributed.idempotency")),
    _c("kafka.consumer_groups", "Consumer groups and rebalancing", "distributed", T.DISTRIBUTED,
       aliases=("consumer group", "rebalance", "offset commit"), floor=3,
       deeper=("kafka.delivery_semantics",)),
    _c("kafka.delivery_semantics", "Delivery semantics", "distributed", T.DISTRIBUTED,
       aliases=("at least once", "exactly once", "at most once"), floor=4,
       deeper=("distributed.idempotency",)),
    _c("distributed.idempotency", "Idempotency", "distributed", T.DISTRIBUTED,
       aliases=("idempotent", "deduplication", "dedupe"), floor=3,
       deeper=("distributed.transactions",)),
    _c("distributed.retries", "Retries and backoff", "distributed", T.DISTRIBUTED,
       aliases=("exponential backoff", "dead letter queue", "dlq"), floor=2,
       deeper=("distributed.idempotency",)),
    _c("distributed.consistency", "Consistency models", "distributed", T.DISTRIBUTED,
       aliases=("eventual consistency", "strong consistency", "cap theorem", "cap"), floor=3),
    _c("distributed.transactions", "Distributed transactions", "distributed", T.DISTRIBUTED,
       aliases=("saga", "two phase commit", "2pc", "outbox"), floor=4),
    _c("distributed.consensus", "Consensus and leader election", "distributed", T.DISTRIBUTED,
       aliases=("raft", "paxos", "leader election", "quorum"), floor=4),
    _c("distributed.concurrency", "Distributed concurrency", "distributed", T.DISTRIBUTED,
       aliases=("distributed lock", "optimistic concurrency"), floor=4),
    # --- System design ----------------------------------------------------
    _c("system_design", "System design", "architecture", T.SYSTEM_DESIGN),
    _c("system_design.requirements", "Requirements gathering", "architecture", T.SYSTEM_DESIGN,
       aliases=("functional requirements", "non-functional"), floor=2),
    _c("system_design.api", "API design", "architecture", T.SYSTEM_DESIGN,
       aliases=("rest api design", "api contract"), floor=2),
    _c("system_design.caching", "Caching", "architecture", T.SYSTEM_DESIGN,
       aliases=("cache", "redis", "cache invalidation"), floor=2,
       deeper=("system_design.caching.invalidation",)),
    _c("system_design.caching.invalidation", "Cache invalidation", "architecture",
       T.SYSTEM_DESIGN, aliases=("write-through", "cache stampede", "ttl"), floor=4),
    _c("system_design.load_balancing", "Load balancing", "architecture", T.SYSTEM_DESIGN,
       aliases=("load balancer",), floor=2),
    _c("system_design.rate_limiting", "Rate limiting", "architecture", T.SYSTEM_DESIGN,
       aliases=("token bucket", "throttling"), floor=3),
    _c("system_design.observability", "Observability", "architecture", T.SYSTEM_DESIGN,
       aliases=("metrics", "tracing", "logging", "p99"), floor=2),
    _c("system_design.fault_tolerance", "Fault tolerance", "architecture", T.SYSTEM_DESIGN,
       aliases=("failover", "graceful degradation", "bulkhead"), floor=3),
    _c("system_design.scaling", "Scaling", "architecture", T.SYSTEM_DESIGN,
       aliases=("horizontal scaling", "vertical scaling"), floor=2),
    _c("system_design.microservices", "Microservices", "architecture", T.SYSTEM_DESIGN,
       aliases=("microservice", "service boundaries"), floor=3),
    _c("system_design.auth", "Authentication and authorisation", "architecture", T.SYSTEM_DESIGN,
       aliases=("oauth", "jwt", "authz", "authn"), floor=2),
    # --- DSA --------------------------------------------------------------
    _c("dsa", "Data structures and algorithms", "algorithms", T.DSA, aliases=("algorithms",)),
    _c("dsa.arrays", "Arrays and strings", "algorithms", T.DSA, floor=1,
       deeper=("dsa.two_pointers", "dsa.sliding_window")),
    _c("dsa.hashing", "Hash maps", "algorithms", T.DSA, aliases=("hash table",), floor=1),
    _c("dsa.two_pointers", "Two pointers", "algorithms", T.DSA, floor=2),
    _c("dsa.sliding_window", "Sliding window", "algorithms", T.DSA, floor=2),
    _c("dsa.binary_search", "Binary search", "algorithms", T.DSA, floor=2),
    _c("dsa.linked_lists", "Linked lists", "algorithms", T.DSA, floor=1),
    _c("dsa.stacks_queues", "Stacks and queues", "algorithms", T.DSA, floor=1),
    _c("dsa.trees", "Trees", "algorithms", T.DSA, aliases=("binary tree", "bst"), floor=2,
       deeper=("dsa.graphs",)),
    _c("dsa.heaps", "Heaps and priority queues", "algorithms", T.DSA,
       aliases=("priority queue", "heap"), floor=2),
    _c("dsa.graphs", "Graphs", "algorithms", T.DSA, aliases=("bfs", "dfs", "shortest path"),
       floor=3, deeper=("dsa.dynamic_programming",)),
    _c("dsa.recursion", "Recursion", "algorithms", T.DSA, floor=2),
    _c("dsa.dynamic_programming", "Dynamic programming", "algorithms", T.DSA,
       aliases=("memoization", "dp"), floor=4),
    _c("dsa.greedy", "Greedy algorithms", "algorithms", T.DSA, floor=3),
    _c("dsa.intervals", "Intervals", "algorithms", T.DSA, floor=2),
    _c("dsa.union_find", "Union find", "algorithms", T.DSA, aliases=("disjoint set",), floor=3),
    _c("dsa.tries", "Tries", "algorithms", T.DSA, aliases=("trie", "prefix tree"), floor=3),
    _c("dsa.complexity", "Complexity analysis", "algorithms", T.DSA,
       aliases=("big o", "time complexity", "space complexity"), floor=1),
    # --- Cloud / DevOps ---------------------------------------------------
    _c("cloud", "Cloud and DevOps", "platform", T.CLOUD, aliases=("devops",)),
    _c("cloud.aws", "AWS", "platform", T.CLOUD, aliases=("amazon web services", "s3", "ec2"),
       floor=2),
    _c("cloud.docker", "Docker", "platform", T.CLOUD, aliases=("containers",), floor=2),
    _c("cloud.kubernetes", "Kubernetes", "platform", T.CLOUD, aliases=("k8s",), floor=3),
    _c("cloud.cicd", "CI/CD", "platform", T.CLOUD, aliases=("continuous integration", "jenkins"),
       floor=2),
    _c("cloud.incident_response", "Incident response", "platform", T.CLOUD,
       aliases=("on-call", "postmortem", "sre"), floor=3),
    # --- Frontend ---------------------------------------------------------
    _c("frontend", "Frontend", "frontend", T.FRONTEND, aliases=("javascript", "typescript")),
    _c("frontend.react", "React", "frontend", T.FRONTEND, aliases=("reactjs", "hooks"), floor=2),
    _c("frontend.state", "State management", "frontend", T.FRONTEND, aliases=("redux",), floor=3),
    _c("frontend.performance", "Frontend performance", "frontend", T.FRONTEND,
       aliases=("rendering performance",), floor=3),
    # --- AI engineering ---------------------------------------------------
    _c("ai", "AI engineering", "ai", T.AI_ENGINEERING, aliases=("llm", "genai")),
    _c("ai.rag", "Retrieval-augmented generation", "ai", T.AI_ENGINEERING,
       aliases=("rag", "retrieval augmented"), floor=3, deeper=("ai.evaluation",)),
    _c("ai.embeddings", "Embeddings and vector search", "ai", T.AI_ENGINEERING,
       aliases=("vector database", "embedding", "pgvector"), floor=3),
    _c("ai.agents", "Agents and tool calling", "ai", T.AI_ENGINEERING,
       aliases=("langgraph", "langchain", "mcp", "tool calling"), floor=3),
    _c("ai.evaluation", "LLM evaluation", "ai", T.AI_ENGINEERING,
       aliases=("llm eval", "hallucination", "guardrails"), floor=4),
    # --- Behavioural / ownership -----------------------------------------
    _c("behavioral", "Behavioural", "behavioral", T.BEHAVIORAL),
    _c("behavioral.ownership", "Ownership", "behavioral", T.BEHAVIORAL, floor=1),
    _c("behavioral.conflict", "Conflict and disagreement", "behavioral", T.BEHAVIORAL, floor=2),
    _c("behavioral.failure", "Failure and learning", "behavioral", T.BEHAVIORAL, floor=1),
    _c("behavioral.ambiguity", "Ambiguity", "behavioral", T.BEHAVIORAL, floor=2),
    _c("behavioral.collaboration", "Collaboration", "behavioral", T.BEHAVIORAL, floor=1),
    _c("behavioral.leadership", "Leadership and influence", "behavioral", T.HIRING_MANAGER,
       floor=3),
    _c("behavioral.incidents", "Production incidents", "behavioral", T.HIRING_MANAGER,
       aliases=("outage", "production incident"), floor=3),
    _c("behavioral.tradeoffs", "Technical tradeoffs", "behavioral", T.HIRING_MANAGER,
       aliases=("tradeoff", "trade-off"), floor=3),
)


@lru_cache(maxsize=1)
def concept_index() -> dict[str, ConceptDef]:
    return {concept.key: concept for concept in CONCEPTS}


def get_concept(key: str) -> ConceptDef | None:
    return concept_index().get(key)


def display_name(key: str) -> str:
    concept = get_concept(key)
    if concept:
        return concept.display_name
    return key.rsplit(".", 1)[-1].replace("_", " ").title()


def children_of(key: str) -> list[ConceptDef]:
    prefix = f"{key}."
    return [
        concept
        for concept in CONCEPTS
        if concept.key.startswith(prefix) and "." not in concept.key[len(prefix) :]
    ]


def ancestors_of(key: str) -> list[str]:
    """['java.concurrency.memory_model'] -> ['java', 'java.concurrency']"""
    parts = key.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts))]


def descendants_of(key: str) -> list[ConceptDef]:
    """Every concept beneath ``key`` at any depth."""
    prefix = f"{key}."
    return [concept for concept in CONCEPTS if concept.key.startswith(prefix)]


def examinable_under(key: str, difficulty: int = 3) -> list[str]:
    """Concrete concepts you can actually build a question around.

    A branch node like ``java`` or ``kafka`` is a category, not something to ask about -
    "tell me about Java" produces an unscoreable answer. This resolves such a key to the
    *leaves* beneath it (at any depth), because an intermediate node like
    ``java.collections`` is just as unaskable as its parent.

    Callers that care about rubric coverage should use
    :func:`gauntlet.graph.slate.resolve_examinable`, which additionally keeps a branch
    concept when one has an authored rubric of its own.
    """
    if key not in concept_index():
        return []

    leaves = [concept.key for concept in descendants_of(key) if not is_branch(concept.key)]
    if not leaves:
        return [key]

    in_band = [
        leaf
        for leaf in leaves
        if (concept := concept_index()[leaf]).difficulty_floor
        <= difficulty
        <= concept.difficulty_ceiling
    ]
    return in_band or leaves


def is_branch(key: str) -> bool:
    """True when the concept has children and is therefore too broad to ask about."""
    return bool(descendants_of(key))


def deeper_concepts(key: str) -> list[str]:
    """Where to descend after a strong answer: explicit edges first, then children."""
    concept = get_concept(key)
    if concept is None:
        return []
    ordered = [candidate for candidate in concept.deeper if candidate in concept_index()]
    ordered.extend(child.key for child in children_of(key) if child.key not in ordered)
    return ordered


def concepts_for_type(interview_type: InterviewType) -> list[ConceptDef]:
    return [concept for concept in CONCEPTS if concept.interview_type is interview_type]


def taxonomy_for_prompt(keys: list[str] | None = None) -> list[dict[str, object]]:
    """Compact taxonomy view injected into prompt context blocks."""
    selected = CONCEPTS if keys is None else [c for c in CONCEPTS if c.key in set(keys)]
    return [
        {
            "key": concept.key,
            "display_name": concept.display_name,
            "domain": concept.domain,
            "interview_type": concept.interview_type.value,
            "aliases": list(concept.aliases),
        }
        for concept in selected
    ]
