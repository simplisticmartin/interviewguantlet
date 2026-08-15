"""PII and confidentiality screening for contributed content (spec section 38).

When someone contributes an interview question they were asked, the raw text routinely
contains things that must never reach a public corpus: the interviewer's name, the
contributor's email, a recruiter's LinkedIn, or the fact that the whole thing came from a
take-home under NDA.

Two different jobs, deliberately kept apart:

**Redaction** removes high-confidence personal identifiers. Emails, phone numbers, handles
and URLs match unambiguously, so they are replaced rather than flagged.

**Blocking** rejects content that should not be in the corpus at any level of scrubbing.
An NDA-covered take-home does not become publishable by removing the author's name, and a
leaked assessment is not a contribution. These are refused with a reason.

Names are the hard case and are treated as a third category: *flagged for review* rather
than auto-redacted. Redacting every capitalised word would destroy "Kafka", "Redis" and
"Spring"; ignoring names entirely would leak "my interviewer Sarah said". So names are
detected only in the constructions that actually introduce a person, and anything
uncertain goes to a human instead of being silently published or silently mangled.

The bias throughout is toward refusing or escalating, because a corpus that wrongly
publishes one person's name is worse than a corpus that is slightly smaller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Verdict(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    BLOCK = "block"


class FindingKind(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    SOCIAL_HANDLE = "social_handle"
    PERSON_NAME = "person_name"
    CONFIDENTIAL = "confidential"
    LEAKED_ASSESSMENT = "leaked_assessment"
    CONTACT_REQUEST = "contact_request"


@dataclass(frozen=True, slots=True)
class Finding:
    kind: FindingKind
    excerpt: str
    redacted: bool
    reason: str


@dataclass
class SafetyReport:
    verdict: Verdict
    text: str
    findings: list[Finding] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.verdict is not Verdict.BLOCK

    @property
    def redactions(self) -> int:
        return sum(1 for finding in self.findings if finding.redacted)

    def kinds(self) -> set[FindingKind]:
        return {finding.kind for finding in self.findings}


# --- High confidence identifiers. Matched precisely, so they are redacted. ------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]?\d{3,4}[\s.-]?\d{0,4}(?!\d)"
)
_URL = re.compile(r"\bhttps?://[^\s<>\"]+", re.I)
_HANDLE = re.compile(r"(?<![\w/])@[A-Za-z][\w.-]{2,}\b")

# --- Names, only in constructions that genuinely introduce a person. ------------
# Requires an introducing phrase, so ordinary capitalised technology names are safe.
_NAME_CONTEXT = re.compile(
    r"\b(?:my |the |a )?"
    r"(?:interviewer|recruiter|hiring manager|engineer|manager|panelist|"
    r"co-?founder|cto|vp)\b[^.\n]{0,20}?\b(?:named|called|was|is|,)\s+"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)",
    re.I,
)
# The sign-off word is case-insensitive ("Thanks," or "thanks,") but the captured name
# stays case-sensitive, so "thanks, everyone" is not mistaken for a person.
_SIGNED_BY = re.compile(
    r"\b(?i:regards|thanks|sincerely|signed|submitted by)[,:\s]+"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)\b"
)

# --- Content that should not be published however well scrubbed. ----------------
_CONFIDENTIAL = re.compile(
    r"\b(nda|non-?disclosure|confidential|do not (?:share|distribute|redistribute)|"
    r"internal (?:only|use only)|proprietary|under embargo|not for public)\b",
    re.I,
)
_LEAKED_ASSESSMENT = re.compile(
    r"\b(?:leaked|stolen|dump(?:ed)? (?:of|from)|answer key|solution key|"
    r"actual (?:test|exam|assessment) (?:paper|content)|"
    r"question bank (?:from|of) (?:their|the company)|"
    r"screenshots? of the (?:test|assessment))\b",
    re.I,
)
# Contributors sometimes add "email me at ..." or "add me on LinkedIn", which is a
# solicitation rather than interview content.
_CONTACT_REQUEST = re.compile(
    r"\b(?:dm me|message me|contact me|reach me|add me on|connect with me|email me)\b", re.I
)

REDACTED = "[redacted]"


def _apply(
    pattern: re.Pattern[str],
    text: str,
    kind: FindingKind,
    reason: str,
    findings: list[Finding],
    *,
    group: int = 0,
    replacement: str = REDACTED,
) -> str:
    """Replace matches, recording each one."""
    result: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.span(group)
        excerpt = match.group(group)
        if not excerpt or not excerpt.strip():
            continue
        findings.append(Finding(kind=kind, excerpt=excerpt[:80], redacted=True, reason=reason))
        result.append(text[cursor:start])
        result.append(replacement)
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


def screen(text: str) -> SafetyReport:
    """Screen contributed text, returning a redacted version and a verdict."""
    findings: list[Finding] = []
    reasons: list[str] = []
    cleaned = text

    # 1. Refusals first. If it is blocked, redaction is beside the point.
    for match in _LEAKED_ASSESSMENT.finditer(text):
        findings.append(
            Finding(
                kind=FindingKind.LEAKED_ASSESSMENT,
                excerpt=match.group(0)[:80],
                redacted=False,
                reason="Appears to reference leaked or stolen assessment material.",
            )
        )
    for match in _CONFIDENTIAL.finditer(text):
        findings.append(
            Finding(
                kind=FindingKind.CONFIDENTIAL,
                excerpt=match.group(0)[:80],
                redacted=False,
                reason="Marked confidential or covered by an agreement.",
            )
        )

    blocked_kinds = {FindingKind.LEAKED_ASSESSMENT, FindingKind.CONFIDENTIAL}
    if blocked_kinds & {finding.kind for finding in findings}:
        if any(f.kind is FindingKind.LEAKED_ASSESSMENT for f in findings):
            reasons.append(
                "Content references leaked or confidential assessment material, which "
                "Gauntlet does not accept from any source."
            )
        if any(f.kind is FindingKind.CONFIDENTIAL for f in findings):
            reasons.append(
                "Content is marked confidential or NDA-covered. Removing names would not "
                "make it publishable."
            )
        return SafetyReport(
            verdict=Verdict.BLOCK, text=text, findings=findings, reasons=reasons
        )

    # 2. Redact unambiguous identifiers.
    cleaned = _apply(
        _EMAIL, cleaned, FindingKind.EMAIL, "Email address removed.", findings
    )
    cleaned = _apply(_URL, cleaned, FindingKind.URL, "Link removed.", findings)
    cleaned = _apply(
        _HANDLE, cleaned, FindingKind.SOCIAL_HANDLE, "Social handle removed.", findings
    )
    cleaned = _apply(
        _PHONE, cleaned, FindingKind.PHONE, "Phone number removed.", findings
    )

    # 3. Names: redact only where the phrasing genuinely introduces a person.
    for pattern in (_NAME_CONTEXT, _SIGNED_BY):
        cleaned = _apply(
            pattern,
            cleaned,
            FindingKind.PERSON_NAME,
            "Personal name removed. Interviewer identities are never published.",
            findings,
            group=1,
            replacement="[name]",
        )

    # 4. Solicitation is not interview content, so escalate rather than publish.
    for match in _CONTACT_REQUEST.finditer(cleaned):
        findings.append(
            Finding(
                kind=FindingKind.CONTACT_REQUEST,
                excerpt=match.group(0)[:80],
                redacted=False,
                reason="Looks like a request to be contacted rather than interview content.",
            )
        )

    verdict = Verdict.ACCEPT
    if FindingKind.CONTACT_REQUEST in {finding.kind for finding in findings}:
        verdict = Verdict.REVIEW
        reasons.append("Contains a contact solicitation; a human should look at this.")
    elif any(finding.kind is FindingKind.PERSON_NAME for finding in findings):
        # A name was found and removed, but the surrounding sentence may still identify
        # someone, so a person checks it.
        verdict = Verdict.REVIEW
        reasons.append("A personal name was removed; confirm nobody remains identifiable.")

    return SafetyReport(verdict=verdict, text=cleaned, findings=findings, reasons=reasons)


def redact(text: str) -> str:
    """Convenience wrapper returning only the scrubbed text."""
    return screen(text).text
