"""Retrieves the knowledge base entries that best match a free-text query.

Each entry becomes one searchable document (name, text
and its genre, mood and tag hints), and a query is ranked against them with
TF-IDF cosine similarity. Entries below a minimum similarity are dropped, so an
unrelated query returns nothing instead of a bad match.
"""

from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DEFAULT_MIN_SCORE = 0.15


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
