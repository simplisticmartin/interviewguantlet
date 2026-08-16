"""Tracing and cost accounting (spec section 43).

Kept out of the business logic: nothing in ``gauntlet.graph`` or ``gauntlet.agents``
should have to know whether anyone is collecting telemetry.
"""

from gauntlet.observability.cost import (
    CostTally,
    ModelPrice,
    cost_scope,
    estimate_cost,
    find_price,
    record_cost,
)
from gauntlet.observability.tracing import (
    add_trace_context,
    configure_tracing,
    current_trace_id,
    record_llm_call,
    span,
    tracing_active,
)

__all__ = [
    "CostTally",
    "ModelPrice",
    "add_trace_context",
    "configure_tracing",
    "cost_scope",
    "current_trace_id",
    "estimate_cost",
    "find_price",
    "record_cost",
    "record_llm_call",
    "span",
    "tracing_active",
]
