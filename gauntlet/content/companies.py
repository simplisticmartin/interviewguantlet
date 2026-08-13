"""Company catalogue (spec section 6).

Companies are normalised data, never hardcoded branching. New ones are added by
appending a row - nothing in the interview engine knows any company by name.

**On the interview mixes below.** Gauntlet ships zero observed interview evidence.
Every mix here is an *archetype-based estimate* derived from the publicly stated shape
of that class of engineering org, and each is stamped ``evidence="estimated"``. The API
and UI surface that label, and the planner is told the mix is estimated so it never
tells a candidate "Company X asks this" (spec sections 9, 10, 26). Real evidence arrives
only through the ingestion pipeline with provenance attached, at which point a company's
mix is recomputed from ``company_question_occurrences`` and relabelled ``observed``.
"""

from __future__ import annotations

from dataclasses import dataclass

from gauntlet.schemas import InterviewType

T = InterviewType

# Archetype -> estimated interview-type distribution. Weights sum to 1.0.
ARCHETYPES: dict[str, dict[InterviewType, float]] = {
    # Heavy algorithmic screening, strong design bar at senior+.
    "big_tech_algo": {
        T.DSA: 0.35,
        T.SYSTEM_DESIGN: 0.25,
        T.BEHAVIORAL: 0.20,
        T.JAVA: 0.10,
        T.HIRING_MANAGER: 0.10,
    },
    # Product/infra companies: practical coding, heavy design and ownership.
    "product_infra": {
        T.DSA: 0.25,
        T.SYSTEM_DESIGN: 0.30,
        T.BEHAVIORAL: 0.20,
        T.DISTRIBUTED: 0.15,
        T.HIRING_MANAGER: 0.10,
    },
    # Enterprise / finance backend: language depth, frameworks, data, correctness.
    "finance_enterprise": {
        T.JAVA: 0.25,
        T.SPRING: 0.15,
        T.DATABASE: 0.15,
        T.SYSTEM_DESIGN: 0.15,
        T.DSA: 0.15,
        T.BEHAVIORAL: 0.15,
    },
    # Consulting / services: breadth, client communication, delivery.
    "consulting": {
        T.JAVA: 0.25,
        T.SPRING: 0.20,
        T.DATABASE: 0.15,
        T.BEHAVIORAL: 0.25,
        T.SYSTEM_DESIGN: 0.15,
    },
    # AI labs: applied ML/AI engineering plus strong general engineering.
    "ai_lab": {
        T.AI_ENGINEERING: 0.35,
        T.DSA: 0.20,
        T.SYSTEM_DESIGN: 0.20,
        T.BEHAVIORAL: 0.15,
        T.DISTRIBUTED: 0.10,
    },
    # Quant / trading: algorithmic depth, low-level performance, precision.
    "trading": {
        T.DSA: 0.45,
        T.JAVA: 0.20,
        T.SYSTEM_DESIGN: 0.15,
        T.DISTRIBUTED: 0.10,
        T.BEHAVIORAL: 0.10,
    },
    # Infrastructure / data platform vendors.
    "data_platform": {
        T.DISTRIBUTED: 0.30,
        T.SYSTEM_DESIGN: 0.25,
        T.DSA: 0.20,
        T.DATABASE: 0.15,
        T.BEHAVIORAL: 0.10,
    },
}


@dataclass(frozen=True, slots=True)
class CompanySeed:
    slug: str
    name: str
    sector: str
    archetype: str
    aliases: tuple[str, ...] = ()

    def interview_mix(self) -> dict[str, object]:
        mix = ARCHETYPES[self.archetype]
        return {
            "evidence": "estimated",
            "basis": f"archetype:{self.archetype}",
            "disclaimer": (
                "Estimated from the general shape of this class of engineering "
                "organisation. Gauntlet holds no observed interview reports for this "
                "company. This is a simulation, not a description of their process."
            ),
            "distribution": {key.value: round(value, 3) for key, value in mix.items()},
        }


def _c(slug: str, name: str, sector: str, archetype: str, *aliases: str) -> CompanySeed:
    return CompanySeed(slug=slug, name=name, sector=sector, archetype=archetype, aliases=aliases)


COMPANIES: tuple[CompanySeed, ...] = (
    # --- Big tech / top technology ---------------------------------------
    _c("google", "Google", "technology", "big_tech_algo", "alphabet"),
    _c("meta", "Meta", "technology", "big_tech_algo", "facebook"),
    _c("amazon", "Amazon", "technology", "product_infra", "aws", "amazon web services"),
    _c("apple", "Apple", "technology", "big_tech_algo"),
    _c("microsoft", "Microsoft", "technology", "big_tech_algo", "msft"),
    _c("netflix", "Netflix", "technology", "product_infra"),
    _c("nvidia", "Nvidia", "technology", "big_tech_algo"),
    _c("openai", "OpenAI", "technology", "ai_lab"),
    _c("anthropic", "Anthropic", "technology", "ai_lab"),
    _c("xai", "xAI", "technology", "ai_lab"),
    _c("tesla", "Tesla", "technology", "product_infra"),
    _c("uber", "Uber", "technology", "product_infra"),
    _c("airbnb", "Airbnb", "technology", "product_infra"),
    _c("stripe", "Stripe", "fintech", "product_infra"),
    _c("datadog", "Datadog", "technology", "data_platform"),
    _c("cloudflare", "Cloudflare", "technology", "data_platform"),
    _c("snowflake", "Snowflake", "technology", "data_platform"),
    _c("databricks", "Databricks", "technology", "data_platform"),
    _c("palantir", "Palantir", "technology", "product_infra"),
    _c("linkedin", "LinkedIn", "technology", "product_infra"),
    _c("salesforce", "Salesforce", "technology", "finance_enterprise"),
    _c("adobe", "Adobe", "technology", "big_tech_algo"),
    _c("oracle", "Oracle", "technology", "finance_enterprise"),
    _c("cisco", "Cisco", "technology", "finance_enterprise"),
    _c("bloomberg", "Bloomberg", "fintech", "finance_enterprise"),
    # --- Finance / fintech ------------------------------------------------
    _c("jpmorgan-chase", "JPMorgan Chase", "finance", "finance_enterprise",
       "jpmorgan", "jp morgan", "chase"),
    _c("goldman-sachs", "Goldman Sachs", "finance", "finance_enterprise", "goldman"),
    _c("morgan-stanley", "Morgan Stanley", "finance", "finance_enterprise"),
    _c("bank-of-america", "Bank of America", "finance", "finance_enterprise", "bofa"),
    _c("wells-fargo", "Wells Fargo", "finance", "finance_enterprise"),
    _c("capital-one", "Capital One", "finance", "finance_enterprise"),
    _c("citi", "Citi", "finance", "finance_enterprise", "citigroup", "citibank"),
    _c("blackrock", "BlackRock", "finance", "finance_enterprise"),
    _c("fidelity", "Fidelity", "finance", "finance_enterprise"),
    _c("charles-schwab", "Charles Schwab", "finance", "finance_enterprise", "schwab"),
    _c("visa", "Visa", "fintech", "finance_enterprise"),
    _c("mastercard", "Mastercard", "fintech", "finance_enterprise"),
    _c("american-express", "American Express", "fintech", "finance_enterprise", "amex"),
    _c("block", "Block", "fintech", "product_infra", "square"),
    _c("coinbase", "Coinbase", "fintech", "product_infra"),
    _c("robinhood", "Robinhood", "fintech", "product_infra"),
    _c("jane-street", "Jane Street", "finance", "trading"),
    _c("citadel", "Citadel", "finance", "trading"),
    _c("citadel-securities", "Citadel Securities", "finance", "trading"),
    _c("two-sigma", "Two Sigma", "finance", "trading"),
    _c("hudson-river-trading", "Hudson River Trading", "finance", "trading", "hrt"),
    # --- Enterprise / consulting -----------------------------------------
    _c("ibm", "IBM", "enterprise", "finance_enterprise"),
    _c("accenture", "Accenture", "consulting", "consulting"),
    _c("deloitte", "Deloitte", "consulting", "consulting"),
    _c("pwc", "PwC", "consulting", "consulting", "pricewaterhousecoopers"),
    _c("ey", "EY", "consulting", "consulting", "ernst & young", "ernst and young"),
    _c("kpmg", "KPMG", "consulting", "consulting"),
    _c("cgi", "CGI", "consulting", "consulting"),
    _c("cognizant", "Cognizant", "consulting", "consulting"),
    _c("infosys", "Infosys", "consulting", "consulting"),
    _c("tcs", "TCS", "consulting", "consulting", "tata consultancy services"),
    _c("wipro", "Wipro", "consulting", "consulting"),
    _c("virtusa", "Virtusa", "consulting", "consulting"),
)

COMPANY_INDEX: dict[str, CompanySeed] = {company.slug: company for company in COMPANIES}


def find_company(needle: str) -> CompanySeed | None:
    """Resolve a company by slug, name, or alias - case and punctuation tolerant."""
    normalised = needle.strip().lower()
    if not normalised:
        return None
    if normalised in COMPANY_INDEX:
        return COMPANY_INDEX[normalised]
    for company in COMPANIES:
        if normalised == company.name.lower() or normalised in company.aliases:
            return company
    squashed = normalised.replace(" ", "").replace("-", "").replace(".", "")
    for company in COMPANIES:
        candidates = [company.slug, company.name, *company.aliases]
        if any(
            squashed == candidate.lower().replace(" ", "").replace("-", "").replace(".", "")
            for candidate in candidates
        ):
            return company
    return None
