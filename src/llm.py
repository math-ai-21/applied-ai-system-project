"""Thin wrapper around the Gemini API so the rest of the app doesn't depend on it.

Put GEMINI_API_KEY in a .env file at the project root (and optionally
GEMINI_MODEL to pick the model). Any problem (no key, network error, empty
reply) raises LLMUnavailable so callers can fall back to the retrieval-only output.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLMUnavailable(Exception):
    """Raised when the LLM can't produce an answer."""


def generate(prompt: str) -> str:
    """Send a prompt to Gemini and return its text reply."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    try:
        # Keep a reference to the client: the SDK closes it when it is garbage collected.
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = response.text
    except Exception as exc:
        raise LLMUnavailable(f"Gemini call failed: {exc}") from exc
    if not text or not text.strip():
        raise LLMUnavailable("Gemini returned an empty reply")
    return text.strip()
