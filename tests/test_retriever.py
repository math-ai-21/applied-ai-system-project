from pathlib import Path

import pytest

from src.knowledge import load_knowledge
from src.recommender import Recommender
from src.retriever import MAX_MOOD_TAGS, Retriever, build_profile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWLEDGE_JSON = DATA_DIR / "knowledge_base.json"
SONGS_CSV = DATA_DIR / "songs.csv"


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


def make_entry(kind="scenario", genres=(), moods=(), mood_tags=(), energy=0.5, **extra):
    return {
        "id": "x", "kind": kind, "name": "x", "text": "x",
        "genres": list(genres), "moods": list(moods), "mood_tags": list(mood_tags),
        "energy": energy, "acousticness": 0.5, "valence": 0.5, **extra,
    }


def test_empty_results_give_empty_profile():
    assert build_profile([]) == {}


def test_single_entry_becomes_the_profile():
    entry = make_entry(genres=["lofi", "jazz"], moods=["chill"], mood_tags=["dreamy"], energy=0.3)
    assert build_profile([(entry, 0.4)]) == {
        "energy": 0.3, "acousticness": 0.5, "valence": 0.5,
        "genre": "lofi", "mood": "chill", "mood_tags": ["dreamy"],
    }


def test_numeric_average_leans_toward_the_better_match():
    low = make_entry(energy=0.2)
    high = make_entry(energy=0.8)
    assert build_profile([(low, 0.3), (high, 0.1)])["energy"] == 0.35


def test_genre_vote_is_weighted_by_score():
    a = make_entry(genres=["rock"])
    b = make_entry(genres=["pop"])
    c = make_entry(genres=["pop"])
    assert build_profile([(a, 0.30), (b, 0.20), (c, 0.20)])["genre"] == "pop"
    assert build_profile([(a, 0.50), (b, 0.20), (c, 0.20)])["genre"] == "rock"


def test_mood_tags_are_capped():
    entry = make_entry(mood_tags=["a", "b", "c", "d", "e"])
    assert len(build_profile([(entry, 0.5)])["mood_tags"]) == MAX_MOOD_TAGS


def test_top_reference_entry_adds_era_language_and_vocals():
    ref = make_entry(kind="reference", release_decade=1980, language="english", vocal_type="vocals")
    profile = build_profile([(ref, 0.4), (make_entry(), 0.2)])
    assert (profile["release_decade"], profile["language"], profile["vocal_type"]) == (1980, "english", "vocals")


def test_lower_ranked_reference_entry_is_ignored():
    ref = make_entry(kind="reference", release_decade=1980, language="english", vocal_type="vocals")
    profile = build_profile([(make_entry(), 0.4), (ref, 0.2)])
    assert "release_decade" not in profile


def test_study_profile_recommends_a_calm_song(retriever):
    profile = build_profile(retriever.retrieve("music for studying and deep focus"))
    songs = Recommender.load_songs(str(SONGS_CSV))
    top_song, _, _ = Recommender.recommend(profile, songs, k=1)[0]
    assert top_song["energy"] < 0.6
