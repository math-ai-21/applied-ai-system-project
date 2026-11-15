"""Retrieves the knowledge base entries that best match a free-text query.

Each entry becomes one searchable document (name, text
and its genre, mood and tag hints), and a query is ranked against them with
TF-IDF cosine similarity. Entries below a minimum similarity are dropped, so an
unrelated query returns nothing instead of a bad match.

build_profile then turns the retrieved entries into a user_prefs dict that
Recommender.recommend can score with, weighting each entry by its retrieval score.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.knowledge import NUMERIC_FIELDS, REFERENCE_FIELDS

DEFAULT_MIN_SCORE = 0.15
MAX_MOOD_TAGS = 3


def entry_document(entry: Dict) -> str:
    """Return the searchable text for one knowledge entry."""
    hints = entry["genres"] + entry["moods"] + entry["mood_tags"]
    return " ".join([entry["name"], entry["text"], *hints])


class Retriever:
    """Ranks knowledge entries against a query by TF-IDF similarity."""

    def __init__(self, entries: List[Dict], min_score: float = DEFAULT_MIN_SCORE):
        self.entries = entries
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(entry_document(e) for e in entries)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Return up to k (entry, similarity) pairs, best first, above min_score."""
        if not query.strip():
            return []
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        ranked = sorted(zip(self.entries, scores), key=lambda pair: pair[1], reverse=True)
        return [(e, float(s)) for e, s in ranked[:k] if s >= self.min_score]


def _top_by_vote(results: List[Tuple[Dict, float]], field: str, limit: int = 1) -> List[str]:
    """Return the `limit` values of a list field with the most score-weighted votes.

    Ties go to the value seen first (best entry first, then listed order).
    """
    votes: Dict[str, float] = defaultdict(float)
    for entry, score in results:
        for value in entry[field]:
            votes[value] += score
    return sorted(votes, key=votes.get, reverse=True)[:limit]


def build_profile(results: List[Tuple[Dict, float]]) -> Dict:
    """Build a user_prefs dict from (entry, score) pairs; empty if nothing was retrieved."""
    if not results:
        return {}

    total = sum(score for _, score in results)
    profile = {
        field: round(sum(entry[field] * score for entry, score in results) / total, 2)
        for field in NUMERIC_FIELDS
    }

    genres = _top_by_vote(results, "genres")
    if genres:
        profile["genre"] = genres[0]
    moods = _top_by_vote(results, "moods")
    if moods:
        profile["mood"] = moods[0]
    tags = _top_by_vote(results, "mood_tags", limit=MAX_MOOD_TAGS)
    if tags:
        profile["mood_tags"] = tags

    # "Something like <song>": only the best match can set era, language and vocals.
    top_entry = results[0][0]
    if top_entry["kind"] == "reference":
        for field in REFERENCE_FIELDS:
            profile[field] = top_entry[field]
    return profile
