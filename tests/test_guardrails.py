from src.guardrails import (
    MAX_LLM_CALLS, MAX_QUERY_CHARS, MAX_TASKS, CallBudget, keep_catalog_songs, limit_tasks, query_too_long,
)

CATALOG = [
    {"title": "Focus Flow"},
    {"title": "Library Rain"},
]


def test_query_at_the_limit_is_allowed_and_one_over_is_not():
    assert not query_too_long("a" * MAX_QUERY_CHARS)
    assert query_too_long("a" * (MAX_QUERY_CHARS + 1))


def test_call_budget_allows_exactly_the_limit():
    budget = CallBudget()
    results = [budget.try_use() for _ in range(MAX_LLM_CALLS + 2)]
    assert results == [True] * MAX_LLM_CALLS + [False, False]


def test_call_budget_accepts_a_custom_limit():
    budget = CallBudget(limit=1)
    assert budget.try_use() is True
    assert budget.try_use() is False


def test_keep_catalog_songs_drops_unknown_titles_and_reports_them():
    picks, ignored = keep_catalog_songs([("thank u, next", "x"), ("Focus Flow", "y")], CATALOG)
    assert [(song["title"], reason) for song, reason in picks] == [("Focus Flow", "y")]
    assert ignored == ["thank u, next"]


def test_keep_catalog_songs_ignores_case_collapses_duplicates_and_caps():
    items = [("focus flow", "a"), ("Focus Flow", "b"), ("Library Rain", "c")]
    picks, _ = keep_catalog_songs(items, CATALOG, limit=1)
    assert [song["title"] for song, _ in picks] == ["Focus Flow"]


def test_limit_tasks_splits_lines_and_drops_blanks():
    assert limit_tasks("study\n\n  gym  \n") == (["study", "gym"], 0)


def test_limit_tasks_keeps_only_the_first_max_tasks_and_counts_the_rest():
    text = "\n".join(f"task {i}" for i in range(MAX_TASKS + 3))
    tasks, dropped = limit_tasks(text)
    assert tasks == [f"task {i}" for i in range(MAX_TASKS)]
    assert dropped == 3


def test_limit_tasks_on_empty_text():
    assert limit_tasks("  \n \n") == ([], 0)
