"""Loads the RAG knowledge base: short guides about scenarios, genres and moods.

Each entry has retrievable text plus structured hints (related genres, moods,
tags and target audio values) that later stages use to build a scoring profile.
"""

import json
from typing import Dict, List

KINDS = ("scenario", "genre", "mood", "reference")
LIST_FIELDS = ("genres", "moods", "mood_tags")
NUMERIC_FIELDS = ("energy", "acousticness", "valence")
REQUIRED_FIELDS = ("id", "kind", "name", "text") + LIST_FIELDS + NUMERIC_FIELDS
# A "reference" entry describes one well-known song so users can ask for
# "something like" it. It also needs the era, language and vocal type.
REFERENCE_FIELDS = ("release_decade", "language", "vocal_type")


def load_knowledge(json_path: str) -> List[Dict]:
    """Load knowledge base entries from a JSON file and validate their shape."""
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)

    seen_ids = set()
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"Knowledge entry {entry.get('id', '?')} is missing: {missing}")
        if entry["kind"] not in KINDS:
            raise ValueError(f"Knowledge entry {entry['id']} has unknown kind: {entry['kind']}")
        if entry["kind"] == "reference":
            missing = [field for field in REFERENCE_FIELDS if field not in entry]
            if missing:
                raise ValueError(f"Reference entry {entry['id']} is missing: {missing}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate knowledge entry id: {entry['id']}")
        seen_ids.add(entry["id"])
        for field in NUMERIC_FIELDS:
            if not 0.0 <= entry[field] <= 1.0:
                raise ValueError(f"Knowledge entry {entry['id']}: {field} must be 0-1")
    return entries
