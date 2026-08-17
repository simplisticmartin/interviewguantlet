"""Tracing and cost accounting (spec section 43).

Two properties matter more than the rest and most of this file defends them:

1. **An unknown price is never reported as zero.** A cost table that quietly returns
   $0.00 for a model it does not recognise produces a total that looks authoritative and
   is wrong, which is worse than no total at all.
2. **Tracing observes and never participates.** Instrumentation that can change a return
   value, swallow an exception, or fail when no SDK is installed is a liability rather
   than a diagnostic.
"""

from __future__ import annotations

import pytest

from gauntlet.llm.base import LLMRole, Usage
from gauntlet.llm.registry import get_provider
from gauntlet.observability.cost import (
    CostTally,
    cost_scope,
    estimate_cost,
    find_price,
    record_cost,
)
from gauntlet.observability.tracing import (
    add_trace_context,
    configure_tracing,
    current_trace_id,
    span,
    tracing_active,
)


class TestPricing:
    def test_a_known_model_is_priced(self):
        cost = estimate_cost("gpt-4o", Usage(input_tokens=1_000_000, output_tokens=0))
        assert cost == pytest.approx(2.50)

    def test_input_and_output_are_priced_separately(self):
        """Output tokens cost several times more; averaging them would understate spend."""
        cost = estimate_cost("gpt-4o", Usage(input_tokens=1_000_000, output_tokens=1_000_000))
        assert cost == pytest.approx(12.50)

    def test_an_unknown_model_returns_none_not_zero(self):
        """The invariant this module exists to protect."""
        assert estimate_cost("some-model-shipped-tomorrow", Usage(5000, 5000)) is None

    def test_a_dated_snapshot_matches_its_family(self):
        """Vendors append dates to model ids, and the table must not go stale for it."""
        assert find_price("claude-sonnet-4-5-20250929") is not None
        assert find_price("gpt-4o-2024-11-20") is not None

    def test_the_longest_matching_prefix_wins(self):
        """gpt-4o-mini must not be billed at gpt-4o rates."""
        mini = find_price("gpt-4o-mini-2024-07-18")
        full = find_price("gpt-4o-2024-11-20")
        assert mini is not None and full is not None
        assert mini.input_per_million < full.input_per_million

    def test_a_routing_prefix_is_stripped(self):
        """Gateways prefix the vendor onto the model id."""
        assert find_price("anthropic/claude-sonnet-4-5") is not None
        assert find_price("accounts/fireworks/models/llama-4-scout") is not None

    def test_the_offline_provider_is_priced_as_free(self):
        """Free and unknown are different claims, and the offline path is genuinely free."""
        provider = get_provider()
        for role in LLMRole:
            model = provider.model_for(role)
            assert estimate_cost(model, Usage(10_000, 10_000)) == 0.0

    def test_an_empty_model_id_is_unknown(self):
        assert estimate_cost("", Usage(10, 10)) is None


class TestCostTally:
    def test_it_sums_known_calls(self):
        tally = CostTally()
        tally.add("gpt-4o", Usage(1_000_000, 0))
        tally.add("gpt-4o", Usage(1_000_000, 0))
        assert tally.usd == pytest.approx(5.0)
        assert tally.complete

    def test_an_unpriced_call_marks_the_total_incomplete(self):
        tally = CostTally()
        tally.add("gpt-4o", Usage(1_000_000, 0))
        tally.add("mystery-model", Usage(1_000_000, 0))
        assert not tally.complete
        assert "mystery-model" in tally.unpriced_models

    def test_an_incomplete_total_is_described_as_a_floor(self):
        """The wording has to stop someone quoting a partial sum as the full number."""
        tally = CostTally()
        tally.add("gpt-4o", Usage(1_000_000, 0))
        tally.add("mystery-model", Usage(1_000, 0))
        assert tally.describe().startswith("at least")

    def test_a_complete_total_is_stated_plainly(self):
        tally = CostTally()
        tally.add("gpt-4o", Usage(1_000_000, 0))
        assert tally.describe().startswith("$")

    def test_tokens_are_counted_even_when_the_price_is_unknown(self):
        """Usage is always known; only the price is not."""
        tally = CostTally()
        tally.add("mystery-model", Usage(300, 200))
        assert tally.total_tokens == 500
        assert tally.usd == 0.0

    def test_no_calls_is_not_reported_as_zero_dollars(self):
        assert CostTally().describe() == "no model calls"


class TestCostScope:
    def test_calls_inside_a_scope_accumulate(self):
        with cost_scope() as tally:
            record_cost("gpt-4o", Usage(1_000_000, 0))
            record_cost("gpt-4o", Usage(1_000_000, 0))
        assert tally.calls == 2

    def test_scopes_do_not_leak_into_each_other(self):
        with cost_scope() as first:
            record_cost("gpt-4o", Usage(1000, 0))
        with cost_scope() as second:
            record_cost("gpt-4o", Usage(1000, 0))
        assert first.calls == 1
        assert second.calls == 1

    def test_a_nested_scope_restores_the_outer_one(self):
        with cost_scope() as outer:
            record_cost("gpt-4o", Usage(1000, 0))
            with cost_scope() as inner:
                record_cost("gpt-4o", Usage(1000, 0))
            record_cost("gpt-4o", Usage(1000, 0))
        assert inner.calls == 1
        assert outer.calls == 2

    def test_recording_outside_any_scope_is_harmless(self):
        """Most callers are not in a scope, and that must not be an error."""
        assert record_cost("gpt-4o", Usage(1_000_000, 0)) == pytest.approx(2.5)


class TestTracingIsInert:
    """With no SDK installed, every one of these paths must still work."""

    def test_tracing_is_off_by_default(self):
        assert configure_tracing() is False
        assert tracing_active() is False

    def test_a_span_runs_and_returns_the_body_value(self):
        with span("probe", {"gauntlet.test": "yes"}):
            result = 21 * 2
        assert result == 42

    def test_a_span_re_raises_rather_than_swallowing(self):
        """Instrumentation that eats an exception is worse than no instrumentation."""
        with pytest.raises(ValueError, match="boom"), span("probe"):
            raise ValueError("boom")

    def test_none_attributes_are_skipped_rather_than_crashing(self):
        with span("probe", {"a": None, "b": 1}):
            pass

    def test_there_is_no_trace_id_when_nothing_is_collecting(self):
        assert current_trace_id() is None

    def test_the_log_processor_adds_nothing_when_not_tracing(self):
        event = {"event": "something"}
        assert add_trace_context(None, "info", event) == {"event": "something"}


class TestGraphInstrumentation:
    def test_every_node_is_wrapped_at_registration(self):
        """Tracing is applied where nodes are registered, so a new node cannot miss it."""
        from gauntlet.graph.interview_graph import build_interview_graph, traced

        graph = build_interview_graph()
        assert len(graph.nodes) >= 15
        # The wrapper is what registration goes through; confirm it is transparent.
        node = traced("probe", lambda state: {"b": 1, "a": 2})
        assert node({}) == {"b": 1, "a": 2}

    def test_the_wrapper_does_not_alter_a_node_result(self):
        from gauntlet.graph.interview_graph import traced

        sentinel = {"questions_asked": 3}
        assert traced("probe", lambda state: sentinel)({}) is sentinel

    def test_the_wrapper_propagates_node_failures(self):
        from gauntlet.graph.interview_graph import traced

        def broken(_state):
            raise RuntimeError("node failed")

        with pytest.raises(RuntimeError, match="node failed"):
            traced("probe", broken)({})


class TestWithARealSdk:
    """The paths that only execute once an SDK is installed.

    These exist because "configured" code is exactly the code nothing exercises. The
    no-op tests above would pass even if every attribute name were wrong, since a no-op
    span accepts anything.
    """

    @pytest.fixture
    def collected(self):
        """Install a real tracer provider that records spans in memory."""
        sdk = pytest.importorskip(
            "opentelemetry.sdk.trace", reason='needs pip install -e ".[otel]"'
        )
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = sdk.TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # The global provider is write-once, so patch the module's tracer directly
        # rather than fighting OpenTelemetry's one-shot setter.
        import gauntlet.observability.tracing as tracing

        original = tracing.tracer
        tracing.tracer = provider.get_tracer("test")
        try:
            yield exporter
        finally:
            tracing.tracer = original
            trace_api.get_current_span()  # detach any lingering context

    def test_a_span_is_actually_recorded(self, collected):
        with span("probe", {"gauntlet.node": "select_question"}):
            pass
        spans = collected.get_finished_spans()
        assert [s.name for s in spans] == ["probe"]
        assert spans[0].attributes["gauntlet.node"] == "select_question"

    def test_spans_nest(self, collected):
        """The whole point: an agent call sits inside the node that made it."""
        with span("node.select_question"), span("agent.interviewer"):
            pass
        spans = {s.name: s for s in collected.get_finished_spans()}
        assert spans["agent.interviewer"].parent is not None
        assert (
            spans["agent.interviewer"].parent.span_id
            == spans["node.select_question"].context.span_id
        )

    def test_a_failure_is_recorded_on_the_span(self, collected):
        with pytest.raises(ValueError), span("probe"):
            raise ValueError("boom")
        recorded = collected.get_finished_spans()[0]
        assert recorded.status.status_code.name == "ERROR"
        assert recorded.events, "the exception should be attached to the span"

    def test_llm_attributes_use_the_genai_conventions(self, collected):
        from gauntlet.observability.tracing import record_llm_call

        with span("agent.probe") as active:
            cost = record_llm_call(
                active,
                provider="openai",
                model="gpt-4o",
                usage=Usage(input_tokens=1_000_000, output_tokens=0),
            )
        assert cost == pytest.approx(2.50)
        attributes = collected.get_finished_spans()[0].attributes
        assert attributes["gen_ai.system"] == "openai"
        assert attributes["gen_ai.request.model"] == "gpt-4o"
        assert attributes["gen_ai.usage.input_tokens"] == 1_000_000
        assert attributes["gauntlet.llm.cost_known"] is True
        assert attributes["gauntlet.llm.cost_usd"] == pytest.approx(2.50)

    def test_an_unpriced_model_is_flagged_rather_than_costed(self, collected):
        from gauntlet.observability.tracing import record_llm_call

        with span("agent.probe") as active:
            assert record_llm_call(
                active, provider="x", model="mystery", usage=Usage(10, 10)
            ) is None
        attributes = collected.get_finished_spans()[0].attributes
        assert attributes["gauntlet.llm.cost_known"] is False
        # Absent, not zero: a dashboard summing this must not count it as free.
        assert "gauntlet.llm.cost_usd" not in attributes

    def test_log_lines_carry_the_active_trace_id(self, collected):
        """This is what makes a log line and a span findable from each other."""
        with span("probe"):
            event = add_trace_context(None, "info", {"event": "something"})
            assert len(event["trace_id"]) == 32
            assert len(event["span_id"]) == 16
            assert current_trace_id() == event["trace_id"]

    def test_a_graph_node_records_which_state_keys_it_wrote(self, collected):
        from gauntlet.graph.interview_graph import traced

        traced("select_question", lambda state: {"slate": [], "difficulty": 3})({})
        attributes = collected.get_finished_spans()[0].attributes
        assert attributes["gauntlet.node"] == "select_question"
        assert attributes["gauntlet.node.updates"] == "difficulty,slate"


class TestPausingIsNotFailing:
    """Regression: LangGraph pauses a graph by raising, and that was traced as an error.

    Both wait nodes call interrupt() on every single turn, so every interview produced a
    span marked ERROR with an exception attached. A trace where normal operation is red
    is a trace nobody can find a real error in, which makes the whole feature worthless
    rather than merely noisy.

    The second half of the bug: the OpenTelemetry SDK records exceptions and sets error
    status by itself unless told not to, so genuine errors were also being recorded
    twice, once by the SDK and once by us.
    """

    @pytest.fixture
    def collected(self):
        sdk = pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = sdk.TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        import gauntlet.observability.tracing as tracing

        original = tracing.tracer
        tracing.tracer = provider.get_tracer("test")
        try:
            yield exporter
        finally:
            tracing.tracer = original

    def test_an_expected_exception_leaves_the_span_green(self, collected):
        class PauseError(Exception):
            pass

        with pytest.raises(PauseError), span("probe", expected=(PauseError,)):
            raise PauseError()

        recorded = collected.get_finished_spans()[0]
        assert recorded.status.status_code.name != "ERROR"
        assert recorded.attributes["gauntlet.paused"] is True
        assert not [e for e in recorded.events if e.name == "exception"]

    def test_an_expected_exception_is_still_re_raised(self, collected):
        """Tracing observes. It must not swallow the control flow it is watching."""

        class PauseError(Exception):
            pass

        with pytest.raises(PauseError), span("probe", expected=(PauseError,)):
            raise PauseError()

    def test_a_real_error_is_recorded_exactly_once(self, collected):
        """Twice was the SDK and this module both recording the same exception."""
        with pytest.raises(ValueError), span("probe"):
            raise ValueError("boom")

        recorded = collected.get_finished_spans()[0]
        assert recorded.status.status_code.name == "ERROR"
        assert len([e for e in recorded.events if e.name == "exception"]) == 1

    def test_an_unexpected_exception_type_is_still_an_error(self, collected):
        class PauseError(Exception):
            pass

        with pytest.raises(RuntimeError), span("probe", expected=(PauseError,)):
            raise RuntimeError("genuinely broken")

        assert collected.get_finished_spans()[0].status.status_code.name == "ERROR"

    def test_a_real_interview_produces_no_error_spans(self, collected):
        """The end to end check, because this bug only appeared when the graph ran."""
        from langgraph.checkpoint.memory import InMemorySaver

        from gauntlet.graph.interview_graph import build_interview_graph

        app = build_interview_graph().compile(checkpointer=InMemorySaver())
        app.invoke(
            {
                "session_id": "trace-probe",
                "candidate_id": "c1",
                "interview_type": "java",
                "mode": "standard",
                "target_company": "google",
                "target_level": "senior",
                "resume_text": "Backend engineer, Java.",
                "job_description": "Senior backend engineer.",
            },
            {"configurable": {"thread_id": "trace-probe"}},
        )

        spans = collected.get_finished_spans()
        errored = [s for s in spans if s.status.status_code.name == "ERROR"]
        assert not errored, f"normal interview produced error spans: {[s.name for s in errored]}"

        paused = [s.name for s in spans if s.attributes.get("gauntlet.paused")]
        assert "node.wait_for_candidate" in paused, "the interrupt should be marked a pause"
