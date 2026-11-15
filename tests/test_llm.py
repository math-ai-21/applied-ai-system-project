import logging
import pytest

from src import llm
from src.llm import (
    LLMUnavailable, build_llm_only_prompt, build_plan_prompt, build_rag_prompt, generate, parse_llm_picks,
)


class FakeResponse:
    def __init__(self, text):
        self.text = text


def fake_client(reply=None, error=None):
    """Return a stand-in for genai.Client that replies with `reply` or raises `error`."""

    class FakeModels:
        def generate_content(self, model, contents):
            if error:
                raise error
            return FakeResponse(reply)

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    return FakeClient


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        generate("hi")


def test_returns_stripped_text(monkeypatch):
    monkeypatch.setattr(llm.genai, "Client", fake_client(reply="  hello  \n"))
    assert generate("hi") == "hello"


def test_client_error_is_wrapped(monkeypatch):
    monkeypatch.setattr(llm.genai, "Client", fake_client(error=RuntimeError("boom")))
    with pytest.raises(LLMUnavailable, match="boom"):
        generate("hi")


def test_empty_reply_raises(monkeypatch):
    monkeypatch.setattr(llm.genai, "Client", fake_client(reply="   "))
    with pytest.raises(LLMUnavailable, match="empty"):
        generate("hi")


def sample_inputs():
    entry = {"name": "studying and deep focus", "text": "Calm, steady, mostly instrumental music."}
    songs = [
        ({"title": "Focus Flow", "artist": "A", "genre": "lofi", "mood": "focused"}, 0.9, "energy match"),
        ({"title": "Library Rain", "artist": "B", "genre": "lofi", "mood": "chill"}, 0.8, "mood match"),
    ]
    return "music for studying", [(entry, 0.3)], songs


def test_rag_prompt_has_query_guides_songs_in_order_and_grounding_rule():
    query, results, recommendations = sample_inputs()
    prompt = build_rag_prompt(query, results, recommendations)
    assert "music for studying" in prompt
    assert "Calm, steady, mostly instrumental music." in prompt
    assert prompt.index("Focus Flow") < prompt.index("Library Rain")
    assert "ONLY" in prompt


def test_rag_prompt_handles_no_retrieved_guides():
    query, _, recommendations = sample_inputs()
    assert "music for studying" in build_rag_prompt(query, [], recommendations)


CATALOG = [
    {"title": "Focus Flow", "artist": "A", "genre": "lofi", "mood": "focused", "energy": 0.4, "mood_tags": ["dreamy"]},
    {"title": "Library Rain", "artist": "B", "genre": "lofi", "mood": "chill", "energy": 0.35, "mood_tags": []},
]


def test_llm_only_prompt_has_query_and_catalog_songs():
    prompt = build_llm_only_prompt("music for studying", CATALOG)
    assert "music for studying" in prompt
    assert "Focus Flow by A" in prompt and "Library Rain by B" in prompt
    assert "never return an empty list" in prompt


def test_parse_picks_returns_matching_songs_in_order():
    reply = '[{"title": "Library Rain", "reason": "calm"}, {"title": "Focus Flow", "reason": "steady"}]'
    picks, ignored = parse_llm_picks(reply, CATALOG)
    assert [(song["title"], reason) for song, reason in picks] == [("Library Rain", "calm"), ("Focus Flow", "steady")]
    assert ignored == []


def test_parse_picks_ignores_titles_not_in_catalog():
    reply = '[{"title": "thank u, next", "reason": "x"}, {"title": "Focus Flow", "reason": "y"}]'
    picks, ignored = parse_llm_picks(reply, CATALOG)
    assert [song["title"] for song, _ in picks] == ["Focus Flow"]
    assert ignored == ["thank u, next"]


def test_parse_picks_ignores_capitalization_and_code_fences():
    reply = '```json\n[{"title": "focus flow", "reason": "y"}]\n```'
    picks, _ = parse_llm_picks(reply, CATALOG)
    assert [song["title"] for song, _ in picks] == ["Focus Flow"]


def test_parse_picks_rejects_non_json():
    with pytest.raises(LLMUnavailable):
        parse_llm_picks("Sure! Try Focus Flow.", CATALOG)


def test_parse_picks_collapses_duplicates_and_caps_the_list():
    songs = [{"title": f"S{i}", "artist": "A", "genre": "g", "mood": "m", "energy": 0.5, "mood_tags": []} for i in range(8)]
    titles = ["S0", "S0"] + [f"S{i}" for i in range(1, 8)]
    reply = "[" + ", ".join(f'{{"title": "{t}"}}' for t in titles) + "]"
    picks, _ = parse_llm_picks(reply, songs, limit=5)
    assert [song["title"] for song, _ in picks] == ["S0", "S1", "S2", "S3", "S4"]


def test_plan_prompt_has_each_task_and_song_and_skips_unmatched_tasks():
    entry = {"name": "studying and deep focus", "text": "Calm, steady music."}
    plan = [
        {"task": "write my essay", "results": [(entry, 0.3)],
         "pick": ({"title": "Focus Flow", "artist": "A", "genre": "lofi", "mood": "focused"}, 0.9, "")},
        {"task": "xyzzy", "results": [], "pick": None},
    ]
    prompt = build_plan_prompt(plan)
    assert "write my essay" in prompt and "Calm, steady music." in prompt and "Focus Flow by A" in prompt
    assert "xyzzy" not in prompt
    assert "ONLY" in prompt


def test_failed_call_is_logged_without_the_api_key(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key-123")
    monkeypatch.setattr(llm.genai, "Client", fake_client(error=RuntimeError("bad key secret-test-key-123")))
    with caplog.at_level(logging.INFO, logger="src.llm"):
        with pytest.raises(LLMUnavailable):
            generate("hi")
    assert "gemini failed" in caplog.text
    assert "secret-test-key-123" not in caplog.text
    assert "***" in caplog.text


def test_successful_call_logs_the_reply_length_but_not_the_reply(monkeypatch, caplog):
    monkeypatch.setattr(llm.genai, "Client", fake_client(reply="a private reply"))
    with caplog.at_level(logging.INFO, logger="src.llm"):
        generate("a private prompt")
    assert "reply_chars=15" in caplog.text
    assert "a private reply" not in caplog.text and "a private prompt" not in caplog.text


def test_unreadable_reply_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="src.llm"):
        with pytest.raises(LLMUnavailable):
            parse_llm_picks("Sure! Try Focus Flow.", CATALOG)
    assert "could not read gemini's reply" in caplog.text
