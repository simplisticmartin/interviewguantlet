"""Token cost estimation (spec section 43).

"What did this interview cost?" is a question anyone running this needs answered, and it
cannot be answered from token counts alone because every model prices differently.

**Unknown models return ``None``, not zero.** This is the entire design decision worth
defending here. Reporting ``$0.00`` for a model missing from the table is a lie that looks
like a fact, and it is the kind of lie that survives all the way to a budget spreadsheet.
``None`` propagates as "unknown" and the UI says so.

Prices are per million tokens in USD, and they go stale. They are annotated with the date
they were recorded and are treated as estimates everywhere they surface. A dedicated
pricing API would be more accurate and would also mean a network call on a code path that
must work offline, which is not a trade worth making for a number displayed to one
decimal place.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from gauntlet.llm.base import Usage

# Recorded 2026-08. Treat as an estimate: vendors change prices without notice, and
# nothing in the system should make a decision that depends on these being exact.
PRICES_RECORDED_ON = "2026-08"


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input_per_million: float
    output_per_million: float


# Keyed by a prefix of the model id, longest match wins. Vendors append dates and version
# suffixes to model ids constantly ("claude-opus-4-5-20251101"), so exact-match keys go
# stale the moment a new snapshot ships, while the family prefix stays valid.
_PRICES: dict[str, ModelPrice] = {
    # Anthropic
    "claude-opus-4": ModelPrice(15.0, 75.0),
    "claude-sonnet-4": ModelPrice(3.0, 15.0),
    "claude-haiku-4": ModelPrice(1.0, 5.0),
    "claude-3-5-haiku": ModelPrice(0.80, 4.0),
    "claude-3-5-sonnet": ModelPrice(3.0, 15.0),
    "claude-3-opus": ModelPrice(15.0, 75.0),
    # OpenAI
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4o": ModelPrice(2.50, 10.0),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    "gpt-4.1": ModelPrice(2.0, 8.0),
    "o3-mini": ModelPrice(1.10, 4.40),
    "o3": ModelPrice(2.0, 8.0),
    "o1-mini": ModelPrice(1.10, 4.40),
    "o1": ModelPrice(15.0, 60.0),
    # Google
    "gemini-2.5-pro": ModelPrice(1.25, 10.0),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    # DeepSeek
    "deepseek-reasoner": ModelPrice(0.55, 2.19),
    "deepseek-chat": ModelPrice(0.27, 1.10),
    # Alibaba
    "qwen-max": ModelPrice(1.60, 6.40),
    "qwen-plus": ModelPrice(0.40, 1.20),
    "qwen-turbo": ModelPrice(0.05, 0.20),
    # xAI
    "grok-4": ModelPrice(3.0, 15.0),
    "grok-3-mini": ModelPrice(0.30, 0.50),
    "grok-3": ModelPrice(3.0, 15.0),
    # Moonshot
    "kimi-k2": ModelPrice(0.60, 2.50),
    "moonshot-v1": ModelPrice(0.84, 0.84),
    # Meta, via hosted providers. Prices vary by host; these are a mid-market figure.
    "llama-4": ModelPrice(0.35, 1.40),
    "llama-3.3": ModelPrice(0.23, 0.40),
    "llama-3.1": ModelPrice(0.18, 0.18),
    # Mistral
    "mistral-large": ModelPrice(2.0, 6.0),
    "mistral-small": ModelPrice(0.20, 0.60),
    # Local and offline: genuinely free, as opposed to unknown.
    "scripted": ModelPrice(0.0, 0.0),
}


def find_price(model: str) -> ModelPrice | None:
    """Longest matching family prefix, or ``None`` when the model is not in the table."""
    if not model:
        return None
    normalised = model.strip().lower()
    # Strip any vendor routing prefix ("anthropic/claude-...", "accounts/fireworks/...").
    tail = normalised.rsplit("/", 1)[-1]

    best: ModelPrice | None = None
    best_length = 0
    for prefix, price in _PRICES.items():
        for candidate in (normalised, tail):
            if candidate.startswith(prefix) and len(prefix) > best_length:
                best, best_length = price, len(prefix)
    return best


def estimate_cost(model: str, usage: Usage) -> float | None:
    """Estimated USD for one call, or ``None`` when the model's price is unknown.

    Returning ``None`` rather than 0.0 is deliberate: a missing price is not a free call,
    and a total that silently omits some calls is worse than one that admits it is
    incomplete.
    """
    price = find_price(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input_per_million
        + usage.output_tokens * price.output_per_million
    ) / 1_000_000


@dataclass(slots=True)
class CostTally:
    """Running total across many calls, tracking what it could not price.

    A tally that has skipped calls reports ``complete=False``, so a caller can say
    "at least $0.42" instead of presenting a partial sum as the full number.
    """

    usd: float = 0.0
    calls: int = 0
    priced_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unpriced_models: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unpriced_models is None:
            self.unpriced_models = set()

    def add(self, model: str, usage: Usage) -> float | None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

        cost = estimate_cost(model, usage)
        if cost is None:
            self.unpriced_models.add(model)
            return None
        self.usd += cost
        self.priced_calls += 1
        return cost

    @property
    def complete(self) -> bool:
        """True when every call in the tally had a known price."""
        return self.calls == self.priced_calls

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def describe(self) -> str:
        """A phrasing that never overstates what is known."""
        if self.calls == 0:
            return "no model calls"
        if self.complete:
            return f"${self.usd:.4f} across {self.calls} calls"
        unpriced = self.calls - self.priced_calls
        return (
            f"at least ${self.usd:.4f} across {self.calls} calls "
            f"({unpriced} unpriced: {', '.join(sorted(self.unpriced_models))})"
        )


# --- Per-interview accumulation ------------------------------------------------
# A context variable rather than a parameter threaded through every call: the tally is
# ambient bookkeeping, and making twenty function signatures carry it would put an
# observability concern into the interview logic's interface.
CURRENT_TALLY: ContextVar[CostTally | None] = ContextVar("gauntlet_cost_tally", default=None)


@contextmanager
def cost_scope() -> Iterator[CostTally]:
    """Accumulate the cost of every model call made inside this block.

    The token is reset on exit, so nested scopes and concurrent requests do not leak
    into each other.
    """
    tally = CostTally()
    token = CURRENT_TALLY.set(tally)
    try:
        yield tally
    finally:
        CURRENT_TALLY.reset(token)


def record_cost(model: str, usage: Usage) -> float | None:
    """Add a call to the active tally, if there is one. Returns its estimated cost."""
    tally = CURRENT_TALLY.get()
    if tally is None:
        return estimate_cost(model, usage)
    return tally.add(model, usage)
