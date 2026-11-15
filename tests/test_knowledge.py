import json
from pathlib import Path

import pytest

from src.knowledge import KINDS, LIST_FIELDS, load_knowledge
from src.recommender import Recommender

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWLEDGE_JSON = DATA_DIR / "knowledge_base.json"
SONGS_CSV = DATA_DIR / "songs.csv"


def test_knowledge_base_loads_with_every_kind():
    entries = load_knowledge(str(KNOWLEDGE_JSON))
    assert entries
    assert {entry["kind"] for entry in entries} == set(KINDS)


def test_every_catalog_genre_has_a_genre_entry():
    songs = Recommender.load_songs(str(SONGS_CSV))
    covered = {e["name"] for e in load_knowledge(str(KNOWLEDGE_JSON)) if e["kind"] == "genre"}
    assert {song["genre"] for song in songs} <= covered


def test_hints_only_reference_values_that_exist_in_the_catalog():
    songs = Recommender.load_songs(str(SONGS_CSV))
    catalog = {
        "genres": {song["genre"] for song in songs},
        "moods": {song["mood"] for song in songs},
        "mood_tags": {tag for song in songs for tag in song["mood_tags"]},
    }
    for entry in load_knowledge(str(KNOWLEDGE_JSON)):
        for field in LIST_FIELDS:
            unknown = set(entry[field]) - catalog[field]
            assert not unknown, f"{entry['id']} has {field} not in catalog: {unknown}"


def test_there_are_ten_reference_songs():
    entries = load_knowledge(str(KNOWLEDGE_JSON))
    assert sum(1 for e in entries if e["kind"] == "reference") == 10


def test_reference_languages_and_vocal_types_exist_in_the_catalog():
    songs = Recommender.load_songs(str(SONGS_CSV))
    languages = {song["language"] for song in songs}
    vocal_types = {song["vocal_type"] for song in songs}
    for entry in load_knowledge(str(KNOWLEDGE_JSON)):
        if entry["kind"] == "reference":
            assert entry["language"] in languages, entry["id"]
            assert entry["vocal_type"] in vocal_types, entry["id"]
            assert isinstance(entry["release_decade"], int), entry["id"]


def write_entries(tmp_path, entries):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def valid_entry(**overrides):
    entry = {
        "id": "x", "kind": "genre", "name": "pop", "text": "Pop music.",
        "genres": ["pop"], "moods": ["happy"], "mood_tags": ["hopeful"],
        "energy": 0.8, "acousticness": 0.1, "valence": 0.8,
    }
    entry.update(overrides)
    return entry


def test_load_rejects_missing_field(tmp_path):
    entry = valid_entry()
    del entry["text"]
    with pytest.raises(ValueError, match="missing"):
        load_knowledge(write_entries(tmp_path, [entry]))


def test_load_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        load_knowledge(write_entries(tmp_path, [valid_entry(), valid_entry()]))


def test_load_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        load_knowledge(write_entries(tmp_path, [valid_entry(kind="artist")]))


def test_load_rejects_reference_without_era_language_or_vocals(tmp_path):
    with pytest.raises(ValueError, match="Reference entry"):
        load_knowledge(write_entries(tmp_path, [valid_entry(kind="reference")]))


def test_load_rejects_out_of_range_number(tmp_path):
    with pytest.raises(ValueError, match="0-1"):
        load_knowledge(write_entries(tmp_path, [valid_entry(energy=1.5)]))
