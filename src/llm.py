"""Thin wrapper around the Gemini API so the rest of the app doesn't depend on it.

Put GEMINI_API_KEY in a .env file at the project root (and optionally
GEMINI_MODEL to pick the model). Any problem (no key, network error, empty
reply) raises LLMUnavailable so callers can fall back to the retrieval-only output.
"""

import json
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


def build_rag_prompt(query, results, recommendations) -> str:
    """Build the prompt for RAG mode: the user's request, the retrieved guides, and the ranked songs."""
    guides = "\n".join(f"- {entry['name']}: {entry['text']}" for entry, _ in results)
    songs = "\n".join(
        f"{rank}. {song['title']} by {song['artist']} ({song['genre']}, {song['mood']}) - {reasons}"
        for rank, (song, _, reasons) in enumerate(recommendations, start=1)
    )
    return (
        "You are a music recommendation assistant. Use ONLY the guides and songs below. "
        "Do not add songs or facts that are not listed. If the guides don't cover part of "
        "the request, say so.\n\n"
        f"User request: {query}\n\n"
        f"Retrieved guides:\n{guides}\n\n"
        f"Recommended songs (already ranked, keep this order):\n{songs}\n\n"
        "In under 120 words, explain why these songs fit the request, citing the guides."
    )


def build_llm_only_prompt(query, songs) -> str:
    """Build the prompt for LLM-only mode: the request plus the catalog to choose from."""
    catalog = "\n".join(
        f"- {s['title']} by {s['artist']} | genre: {s['genre']} | mood: {s['mood']} "
        f"| energy: {s['energy']} | tags: {', '.join(s['mood_tags'])}"
        for s in songs
    )
    return (
        "You are a music recommendation assistant. Pick up to 5 songs for the request, "
        "ONLY from the catalog below, using the exact titles. Reply with JSON only: a list "
        'of objects like {"title": "...", "reason": "one sentence"}, best match first.\n\n'
        f"User request: {query}\n\nCatalog:\n{catalog}"
    )


def parse_llm_picks(reply, songs, limit=5):
    """Turn Gemini's JSON reply into ([(song, reason), ...], ignored_titles).

    Titles not in the catalog are ignored, so the LLM can't recommend songs the app doesn't have.
    """
    text = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        items = json.loads(text)
        titles = [(item["title"], item.get("reason", "")) for item in items]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise LLMUnavailable("could not read Gemini's reply as a list of songs") from exc

    by_title = {s["title"].lower(): s for s in songs}
    picks, ignored = [], []
    for title, reason in titles:
        song = by_title.get(str(title).lower())
        if song is None:
            ignored.append(str(title))
        elif all(song is not picked for picked, _ in picks):
            picks.append((song, reason))
    return picks[:limit], ignored
