import pytest

from src import llm
from src.llm import LLMUnavailable, build_rag_prompt, generate


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
