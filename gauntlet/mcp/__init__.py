"""MCP servers (spec section 16).

Three servers, each launched over stdio by an MCP client:

* ``question_bank_server`` - search the corpus, check for duplicates, company estimates.
  Needs no database and no API key.
* ``candidate_server`` - a candidate's resume, history, skill graph and misconceptions.
  Needs the database.
* ``coding_server`` - static analysis of submitted code. Exposes no execution tools.

Living at ``gauntlet/mcp/`` rather than a top-level ``mcp/`` as the spec sketches, because
a top-level directory of that name shadows the installed ``mcp`` SDK for anything run from
the repository root. Same structure, one fewer footgun.

Registered as console scripts, which is what an MCP client config points at:

    gauntlet-mcp-questions
    gauntlet-mcp-candidate
    gauntlet-mcp-coding
"""

__all__ = ["candidate_server", "coding_server", "question_bank_server"]
