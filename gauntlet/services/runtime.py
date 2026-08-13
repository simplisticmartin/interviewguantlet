"""Process-wide graph runtime.

Owns the compiled graph and its checkpointer for the lifetime of the process. The
checkpointer holds a database connection pool, so it is opened once at startup and
closed at shutdown rather than per request.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from typing import Any

import structlog

from gauntlet.graph.interview_graph import build_interview_graph, checkpointer

log = structlog.get_logger(__name__)


class GraphRuntime:
    """Lazily initialised, thread-safe holder for the compiled interview graph."""

    def __init__(self) -> None:
        self._stack: ExitStack | None = None
        self._graph: Any = None
        self._durable = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._graph is not None:
                return
            self._stack = ExitStack()
            saver = self._stack.enter_context(checkpointer())
            self._durable = type(saver).__name__ != "InMemorySaver"
            self._graph = build_interview_graph().compile(checkpointer=saver)
            log.info(
                "runtime.started",
                checkpointer=type(saver).__name__,
                durable=self._durable,
            )

    def stop(self) -> None:
        with self._lock:
            if self._stack is not None:
                self._stack.close()
                self._stack = None
            self._graph = None
            log.info("runtime.stopped")

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self.start()
        return self._graph

    @property
    def durable_checkpoints(self) -> bool:
        """False when interviews would be lost on restart (in-memory fallback)."""
        if self._graph is None:
            self.start()
        return self._durable


RUNTIME = GraphRuntime()
