"""Guardrails: the checks that keep the app within its limits and its data.

Index of every guardrail (search the code for "GUARDRAIL" to find each one):
  1. Knowledge base validation  - src/knowledge.py, load_knowledge
  2. Retrieval threshold        - src/retriever.py, DEFAULT_MIN_SCORE
  3. Query length limit         - here, query_too_long
  4. Gemini call cap            - here, CallBudget
  5. LLM-unavailable fallback   - src/main.py, run_query_loop
  6. Grounding instruction      - src/llm.py, build_rag_prompt
  7. Catalog-only picks         - here, keep_catalog_songs
"""

MAX_QUERY_CHARS = 200
MAX_LLM_CALLS = 5


def query_too_long(query) -> bool:
    """GUARDRAIL 3, query length: stop long input before it costs a Gemini call."""
    return len(query) > MAX_QUERY_CHARS


class CallBudget:
    """GUARDRAIL 4, call cap: limits how many Gemini calls one run can make."""

    def __init__(self, limit=MAX_LLM_CALLS):
        self.limit = limit
        self.used = 0

    def try_use(self) -> bool:
        """Count one call and return True, or return False if the budget is spent."""
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def keep_catalog_songs(items, songs, limit=5):
    """GUARDRAIL 7, catalog-only picks: drop titles Gemini named that aren't in the catalog.

    items is a list of (title, reason). Returns ([(song, reason), ...], ignored_titles).
    Duplicates are collapsed and the picks are capped at `limit`.
    """
    by_title = {s["title"].lower(): s for s in songs}
    picks, ignored = [], []
    for title, reason in items:
        song = by_title.get(str(title).lower())
        if song is None:
            ignored.append(str(title))
        elif all(song is not picked for picked, _ in picks):
            picks.append((song, reason))
    return picks[:limit], ignored
