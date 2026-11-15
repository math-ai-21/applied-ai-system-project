"""
Command line runner for the Music Recommender Simulation.

Loads the song catalog from data/songs.csv, scores it against each sample
listener profile in PROFILES, and prints the top 5 songs with the reasons
behind each score.

Run with: python -m src.main
Or describe what you want in plain text (RAG mode): python -m src.main --rag
"""

import argparse

from src.knowledge import load_knowledge
from src.recommender import Recommender
from src.retriever import Retriever, build_profile


PROFILES = {
    "Road Trip": {
        "genre": "indie pop", "mood": "uplifting",
        "energy": 0.70, "acousticness": 0.30, "valence": 0.85,
    },
    "High-Energy Pop": {
        "genre": "pop", "mood": "happy",
        "energy": 0.90, "acousticness": 0.10, "valence": 0.80,
    },
    "Chill Lofi": {
        "genre": "lofi", "mood": "chill",
        "energy": 0.35, "acousticness": 0.80, "valence": 0.58,
    },
    "Deep Intense Rock": {
        "genre": "rock", "mood": "intense",
        "energy": 0.92, "acousticness": 0.08, "valence": 0.45,
    },
    # Uses the challenge attributes: popularity, decade, language, vocal type, mood tags.
    "Nostalgic Dreamer": {
        "energy": 0.45, "acousticness": 0.75,
        "popularity": 50, "release_decade": 1990,
        "language": "english", "vocal_type": "vocals",
        "mood_tags": ["nostalgic", "reflective", "melancholic"],
    },
}


def print_recommendations(recommendations) -> None:
    print("Recommendations:")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  {rank}. {song['title']} - Score: {score:.2f}")
        for reason in explanation.split("; "):
            print(f"       - {reason}")


def run_rag(songs) -> None:
    """Ask for plain-text queries until a blank line: retrieve, build a profile, recommend."""
    retriever = Retriever(load_knowledge("data/knowledge_base.json"))
    print("RAG mode: describe what you want to hear (blank line to quit).")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except EOFError:
            break
        if not query:
            break

        results = retriever.retrieve(query)
        if not results:
            print("No knowledge entry matched. Try rephrasing, e.g. 'studying on a rainy day'.")
            continue

        print("Retrieved:")
        for entry, score in results:
            print(f"  - {entry['name']} ({entry['kind']}, similarity {score:.2f})")
        profile = build_profile(results)
        print("Profile: " + ", ".join(f"{key}={value}" for key, value in profile.items()))
        print_recommendations(Recommender.recommend(profile, songs, k=5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument("--rag", action="store_true",
                        help="describe what you want in plain text instead of using sample profiles")
    args = parser.parse_args()

    songs = Recommender.load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from data/songs.csv")

    if args.rag:
        run_rag(songs)
        return

    for name, user_prefs in PROFILES.items():
        recommendations = Recommender.recommend(user_prefs, songs, k=5)

        profile = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
        print(f"\n{name}")
        print(f"User profile: {profile}")
        print_recommendations(recommendations)


if __name__ == "__main__":
    main()
