from pathlib import Path

import pytest

from src.knowledge import load_knowledge
from src.retriever import Retriever

KNOWLEDGE_JSON = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_knowledge(str(KNOWLEDGE_JSON)))


def top_ids(retriever, query, k=3):
    return [entry["id"] for entry, _ in retriever.retrieve(query, k=k)]


def test_study_query_finds_the_study_scenario(retriever):
    assert "scn-study" in top_ids(retriever, "music for studying and deep focus")


def test_rain_query_finds_the_rain_scenario(retriever):
    assert "scn-rain" in top_ids(retriever, "cozy rainy day at home")


def test_workout_query_finds_the_workout_scenario(retriever):
    assert top_ids(retriever, "songs for the gym and running")[0] == "scn-workout"


def test_song_name_query_finds_its_reference_entry(retriever):
    entry, _ = retriever.retrieve("something like Blinding Lights")[0]
    assert entry["kind"] == "reference"
    assert "Blinding Lights" in entry["name"]


def test_results_are_sorted_best_first_and_limited_to_k(retriever):
    results = retriever.retrieve("sad slow songs after a breakup", k=2)
    assert 0 < len(results) <= 2
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_query_returns_nothing(retriever):
    assert retriever.retrieve("xyzzy qwertyuiop") == []


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []


def test_min_score_filters_weak_matches():
    strict = Retriever(load_knowledge(str(KNOWLEDGE_JSON)), min_score=0.99)
    assert strict.retrieve("music for studying") == []
