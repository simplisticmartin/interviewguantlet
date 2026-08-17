"""The interview graph (spec section 3).

    START
      -> parse_candidate -> analyze_job -> retrieve_company_patterns
      -> build_interview_plan
      -> select_question -> ask_question -> wait_for_candidate
      -> classify_response
             |-- clarifying --> answer_clarification -> wait_after_clarification --,
             |-- code ------> check_code ---------------------------------------.  |
             `-- otherwise ------------------------------------------------.    |  |
                                                                            v   v  v
                                                        evaluate_answer <---'---'--'
      -> update_skill_graph -> misconception_check -> adaptive_router
      -> enough_evidence? --no--> select_question
                           --yes-> report -> END

The two ``wait_*`` nodes call ``interrupt()``. Everything before them is checkpointed,
so an interview survives a process restart, a closed browser tab, and - because
checkpoints are addressable - can be rewound to any earlier turn for replay.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph

from gauntlet.config import get_settings
from gauntlet.graph.nodes import (
    adaptive_router,
    analyze_job,
    answer_clarification,
    ask_question,
    build_interview_plan,
    build_report,
    check_submitted_code,
    classify_response,
    coach_candidate,
    enough_evidence,
    evaluate_answer,
    misconception_check,
    parse_candidate,
    retrieve_company_patterns,
    route_after_classification,
    route_after_misconception,
    select_question,
    update_skill_graph,
    wait_after_clarification,
    wait_for_candidate,
)
from gauntlet.graph.state import InterviewState
from gauntlet.observability.tracing import span

log = structlog.get_logger(__name__)


def traced(name: str, node: Any) -> Any:
    """Wrap a node so every execution becomes a span.

    Applied at registration rather than as a decorator on each node function, so the
    node modules stay free of observability imports and a new node cannot be added
    without being traced. The wrapper returns the node's result untouched: tracing
    observes, it never participates.
    """

    def run(state: InterviewState, *args: Any, **kwargs: Any) -> Any:
        # GraphBubbleUp is how LangGraph pauses a graph, including the interrupt() the
        # two wait nodes use on every turn. It is control flow, not a failure, and
        # tracing it as an error made every normal interview look broken.
        with span(
            f"node.{name}", {"gauntlet.node": name}, expected=(GraphBubbleUp,)
        ) as active:
            result = node(state, *args, **kwargs)
            # Which keys a node wrote is the single most useful thing to see when
            # reading a trace of a state machine.
            if isinstance(result, dict):
                active.set_attribute("gauntlet.node.updates", ",".join(sorted(result)))
            return result

    run.__name__ = getattr(node, "__name__", name)
    return run


def build_interview_graph() -> StateGraph:
    """Assemble the graph. Compilation (and the checkpointer) is the caller's choice."""
    graph: StateGraph = StateGraph(InterviewState)

    def add_node(name: str, node: Any) -> None:
        graph.add_node(name, traced(name, node))

    # --- Intake -----------------------------------------------------------
    add_node("parse_candidate", parse_candidate)
    add_node("analyze_job", analyze_job)
    add_node("retrieve_company_patterns", retrieve_company_patterns)
    add_node("build_interview_plan", build_interview_plan)

    # --- Interview loop ---------------------------------------------------
    add_node("select_question", select_question)
    add_node("ask_question", ask_question)
    add_node("wait_for_candidate", wait_for_candidate)
    add_node("classify_response", classify_response)
    add_node("answer_clarification", answer_clarification)
    add_node("wait_after_clarification", wait_after_clarification)
    add_node("check_code", check_submitted_code)
    add_node("evaluate_answer", evaluate_answer)
    add_node("update_skill_graph", update_skill_graph)
    add_node("misconception_check", misconception_check)
    add_node("coach_candidate", coach_candidate)
    add_node("adaptive_router", adaptive_router)

    # --- Report -----------------------------------------------------------
    add_node("report", build_report)

    graph.add_edge(START, "parse_candidate")
    graph.add_edge("parse_candidate", "analyze_job")
    graph.add_edge("analyze_job", "retrieve_company_patterns")
    graph.add_edge("retrieve_company_patterns", "build_interview_plan")
    graph.add_edge("build_interview_plan", "select_question")
    graph.add_edge("select_question", "ask_question")
    graph.add_edge("ask_question", "wait_for_candidate")
    graph.add_edge("wait_for_candidate", "classify_response")

    graph.add_conditional_edges(
        "classify_response",
        route_after_classification,
        {
            "answer_clarification": "answer_clarification",
            "check_code": "check_code",
            "evaluate_answer": "evaluate_answer",
        },
    )

    # A clarification hands the floor back without consuming a question.
    graph.add_edge("answer_clarification", "wait_after_clarification")
    graph.add_edge("wait_after_clarification", "classify_response")

    graph.add_edge("check_code", "evaluate_answer")
    graph.add_edge("evaluate_answer", "update_skill_graph")
    graph.add_edge("update_skill_graph", "misconception_check")
    # Coaching Mode teaches between questions; every other mode routes straight on.
    graph.add_conditional_edges(
        "misconception_check",
        route_after_misconception,
        {"coach_candidate": "coach_candidate", "adaptive_router": "adaptive_router"},
    )
    graph.add_edge("coach_candidate", "adaptive_router")

    graph.add_conditional_edges(
        "adaptive_router",
        enough_evidence,
        {"select_question": "select_question", "report": "report"},
    )
    graph.add_edge("report", END)

    return graph


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _memory_saver() -> InMemorySaver:
    return InMemorySaver()


@contextmanager
def checkpointer() -> Iterator[BaseCheckpointSaver[Any]]:
    """Yield a checkpointer, preferring Postgres and degrading to memory.

    Postgres checkpoints are what make recovery and replay durable across restarts.
    The in-memory saver keeps the app runnable (and tests fast) when no database is
    reachable - at the cost of losing interviews on restart, which is logged.
    """
    settings = get_settings()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        # Same reason as the SQLAlchemy engine: bound the connect attempt so a
        # half-started database cannot stall application startup for minutes.
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}connect_timeout={settings.db_connect_timeout_seconds}"

        with PostgresSaver.from_conn_string(url) as saver:
            saver.setup()
            yield saver
            return
    except Exception as exc:  # pragma: no cover - depends on local infra
        log.warning(
            "graph.checkpointer.degraded",
            reason=str(exc)[:200],
            impact="interviews will not survive a restart",
        )

    yield _memory_saver()


def compile_graph(saver: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Compile with a checkpointer. Pass one explicitly in tests."""
    return build_interview_graph().compile(checkpointer=saver or _memory_saver())


def graph_mermaid() -> str:
    """Mermaid source for the README diagram; generated from the real graph."""
    return build_interview_graph().compile().get_graph().draw_mermaid()
