"""Evaluator benchmark (spec sections 20 and 47).

Treats grading as an ML systems problem: the evaluator is itself measured against
human-labelled answers before anyone trusts its numbers.

Three metrics, because each catches a different failure:

* **Band accuracy** - does the score land in the range a human would accept? Catches
  systematic generosity or harshness.
* **Ranking accuracy** - over every pair of answers whose human labels differ, does the
  evaluator order them correctly? Catches an evaluator that is miscalibrated in absolute
  terms but still discriminates - which is far more useful than the reverse, and is
  invisible to band accuracy alone.
* **Misconception precision / recall** - the product's headline claim is detecting
  confidently-wrong beliefs. A false positive here (telling someone they hold a
  misconception they do not) damages trust more than a miss, so precision is reported
  separately rather than folded into an F1.

Run it:

    python -m evals.runner              # current provider
    python -m evals.runner --json       # machine-readable, for CI diffing

Changing an evaluator prompt without re-running this is how scoring silently regresses.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gauntlet.evaluation.engine import EvaluationEngine
from gauntlet.llm.registry import get_provider
from gauntlet.schemas import AnswerPayload, InterviewType, QuestionSpec

DATASET_DIR = Path(__file__).parent / "datasets"

# Human-assigned label -> acceptable machine score range, and the rank used for
# pairwise ordering checks. Bands overlap deliberately: the boundary between
# "acceptable" and "partial" is genuinely fuzzy for human graders too.
LABELS: dict[str, tuple[float, float, int]] = {
    "excellent": (0.75, 1.01, 5),
    "acceptable": (0.50, 0.90, 4),
    "partial": (0.25, 0.65, 3),
    "wrong": (0.00, 0.30, 2),
    "confidently_wrong": (0.00, 0.45, 1),
}


@dataclass
class CaseResult:
    case_id: str
    topic: str
    label: str
    score: float
    expected_low: float
    expected_high: float
    in_band: bool
    misconception_expected: bool
    misconception_detected: bool
    judge_confidence: float
    disagreement: float

    @property
    def rank(self) -> int:
        return LABELS[self.label][2]


@dataclass
class BenchmarkReport:
    provider: str
    cases: list[CaseResult] = field(default_factory=list)

    # --- Metrics ---------------------------------------------------------

    @property
    def band_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.in_band) / len(self.cases)

    @property
    def ranking_accuracy(self) -> float:
        """Pairwise ordering accuracy within each topic."""
        correct = 0
        total = 0
        by_topic: dict[str, list[CaseResult]] = {}
        for case in self.cases:
            by_topic.setdefault(case.topic, []).append(case)

        for group in by_topic.values():
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    if left.rank == right.rank:
                        continue
                    total += 1
                    better, worse = (
                        (left, right) if left.rank > right.rank else (right, left)
                    )
                    if better.score > worse.score:
                        correct += 1
                    elif better.score == worse.score:
                        correct += 0.5  # a tie is half credit, not a win
        return correct / total if total else 0.0

    @property
    def misconception_precision(self) -> float:
        flagged = [case for case in self.cases if case.misconception_detected]
        if not flagged:
            return 1.0
        return sum(1 for case in flagged if case.misconception_expected) / len(flagged)

    @property
    def misconception_recall(self) -> float:
        expected = [case for case in self.cases if case.misconception_expected]
        if not expected:
            return 1.0
        return sum(1 for case in expected if case.misconception_detected) / len(expected)

    @property
    def false_positive_rate(self) -> float:
        """Flagging a misconception on an answer that holds none."""
        clean = [case for case in self.cases if not case.misconception_expected]
        if not clean:
            return 0.0
        return sum(1 for case in clean if case.misconception_detected) / len(clean)

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(case.judge_confidence for case in self.cases) if self.cases else 0.0

    @property
    def mean_disagreement(self) -> float:
        return statistics.fmean(case.disagreement for case in self.cases) if self.cases else 0.0

    def separates_excellent_from_wrong(self) -> bool:
        """The floor: if this fails, the evaluator is not measuring anything."""
        excellent = [c.score for c in self.cases if c.label == "excellent"]
        bad = [c.score for c in self.cases if c.label in {"wrong", "confidently_wrong"}]
        if not excellent or not bad:
            return False
        return min(excellent) > max(bad)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "cases_evaluated": len(self.cases),
            "band_accuracy": round(self.band_accuracy, 4),
            "ranking_accuracy": round(self.ranking_accuracy, 4),
            "misconception_precision": round(self.misconception_precision, 4),
            "misconception_recall": round(self.misconception_recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "mean_judge_confidence": round(self.mean_confidence, 4),
            "mean_judge_disagreement": round(self.mean_disagreement, 4),
            "separates_excellent_from_wrong": self.separates_excellent_from_wrong(),
            "cases": [asdict(case) for case in self.cases],
        }


def load_datasets(directory: Path | None = None) -> list[dict[str, Any]]:
    folder = directory or DATASET_DIR
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))
    ]


def run_benchmark(datasets: list[dict[str, Any]] | None = None) -> BenchmarkReport:
    data = datasets if datasets is not None else load_datasets()
    engine = EvaluationEngine()
    report = BenchmarkReport(provider=get_provider().name)

    for dataset in data:
        question = QuestionSpec(
            prompt_text=dataset["question"],
            interview_type=InterviewType(dataset["interview_type"]),
            agent_key="benchmark",
            concept_keys=[dataset["concept_key"]],
            difficulty=4,
            rubric_key=dataset.get("rubric_key"),
        )

        for case in dataset["cases"]:
            low, high, _ = LABELS[case["label"]]
            evaluation = engine.evaluate(
                question,
                AnswerPayload(
                    text=case["answer"], self_confidence=case.get("self_confidence")
                ),
                target_role="Senior Software Engineer",
                target_level="senior",
            )
            report.cases.append(
                CaseResult(
                    case_id=case["id"],
                    topic=dataset["topic"],
                    label=case["label"],
                    score=round(evaluation.score, 4),
                    expected_low=low,
                    expected_high=high,
                    in_band=low <= evaluation.score <= high,
                    misconception_expected=bool(case.get("expects_misconception", False)),
                    misconception_detected=evaluation.misconception.detected,
                    judge_confidence=round(evaluation.confidence, 4),
                    disagreement=round(evaluation.disagreement, 4),
                )
            )

    return report


def print_report(report: BenchmarkReport) -> None:
    print(f"\nEvaluator benchmark — provider: {report.provider}")
    print("=" * 78)
    print(f"{'case':<34}{'label':<20}{'score':>8}{'band':>10}{'misc':>6}")
    print("-" * 78)
    for case in report.cases:
        band = f"{case.expected_low:.2f}-{min(case.expected_high, 1.0):.2f}"
        flag = "OK " if case.in_band else "OFF"
        misc = "yes" if case.misconception_detected else "-"
        if case.misconception_expected != case.misconception_detected:
            misc += "!"
        print(f"{case.case_id:<34}{case.label:<20}{case.score:>8.2f}{band:>10} {flag} {misc:>4}")

    print("-" * 78)
    print(f"  band accuracy              {report.band_accuracy:.1%}")
    print(f"  ranking accuracy           {report.ranking_accuracy:.1%}")
    print(f"  misconception precision    {report.misconception_precision:.1%}")
    print(f"  misconception recall       {report.misconception_recall:.1%}")
    print(f"  false positive rate        {report.false_positive_rate:.1%}")
    print(f"  mean judge confidence      {report.mean_confidence:.2f}")
    print(f"  mean judge disagreement    {report.mean_disagreement:.2f}")
    print(f"  separates best from worst  {report.separates_excellent_from_wrong()}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gauntlet evaluator benchmark.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    report = run_benchmark()
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
