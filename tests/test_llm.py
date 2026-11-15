import pytest

from src import llm
from src.llm import LLMUnavailable, generate


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
