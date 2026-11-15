"""Thin wrapper around the Gemini API so the rest of the app doesn't depend on it.

Put GEMINI_API_KEY in a .env file at the project root (and optionally
GEMINI_MODEL to pick the model). Any problem (no key, network error, empty
reply) raises LLMUnavailable so callers can fall back to the retrieval-only output.
"""

import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai

from src.guardrails import keep_catalog_songs

load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLMUnavailable(Exception):
    """Raised when the LLM can't produce an answer."""


def generate(prompt: str) -> str:
    """Send a prompt to Gemini and return its text reply, logging the call (never the key or the reply)."""
    start = time.monotonic()
    try:
        text = _call_gemini(prompt)
    except LLMUnavailable as exc:
        message = str(exc)[:200]
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            message = message.replace(key, "***")
        log.warning("gemini failed: %s", message)
        raise
    log.info("gemini ok in %.1fs reply_chars=%d", time.monotonic() - start, len(text))
    return text


def _call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    log.info("gemini call model=%s prompt_chars=%d", model, len(prompt))
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
    # GUARDRAIL 6, grounding instruction: tell the model to use only what was retrieved.
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
        "ONLY from the catalog below, using the exact titles. Always give the closest matches, "
        "even if none fit well, and say so in the reason; never return an empty list. "
        "Reply with JSON only: a list "
        'of objects like {"title": "...", "reason": "one sentence"}, best match first.\n\n'
        f"User request: {query}\n\nCatalog:\n{catalog}"
    )


def parse_llm_picks(reply, songs, limit=5):
    """Turn Gemini's JSON reply into ([(song, reason), ...], ignored_titles).

    Titles not in the catalog are ignored (guardrail 7), so the LLM can't recommend
    songs the app doesn't have.
    """
    text = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        items = json.loads(text)
        titles = [(item["title"], item.get("reason", "")) for item in items]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        log.warning("could not read gemini's reply as a list of songs")
        raise LLMUnavailable("could not read Gemini's reply as a list of songs") from exc

    return keep_catalog_songs(titles, songs, limit)


def build_plan_prompt(plan) -> str:
    """Build the prompt for the day-plan note: each task with its retrieved guides and chosen song."""
    blocks = []
    for item in plan:
        if item["pick"] is None:
            continue
        song = item["pick"][0]
        guides = "; ".join(f"{entry['name']}: {entry['text']}" for entry, _ in item["results"])
        blocks.append(
            f"Task: {item['task']}\nGuides: {guides}\n"
            f"Song: {song['title']} by {song['artist']} ({song['genre']}, {song['mood']})"
        )
    # GUARDRAIL 6, grounding instruction: tell the model to use only what was retrieved.
    return (
        "You are a music assistant planning a listening playlist for someone's day. Use ONLY the "
        "guides and songs below. Do not add songs or facts that are not listed. For each task, say "
        "in one sentence why its song fits, citing the guide.\n\n"
        + "\n\n".join(blocks)
        + "\n\nEnd with one sentence on the overall mood of the day."
    )
