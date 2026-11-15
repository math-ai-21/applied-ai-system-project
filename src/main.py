"""
Command line runner for the Music Recommender Simulation.

Loads the song catalog from data/songs.csv, scores it against each sample
listener profile in PROFILES, and prints the top 5 songs with the reasons
behind each score.

Run with: python -m src.main
"""

from src.recommender import Recommender


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


def main() -> None:
    songs = Recommender.load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from data/songs.csv")

    for name, user_prefs in PROFILES.items():
        recommendations = Recommender.recommend(user_prefs, songs, k=5)

        profile = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
        print(f"\n{name}")
        print(f"User profile: {profile}")
        print("Recommendations:")
        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"  {rank}. {song['title']} - Score: {score:.2f}")
            for reason in explanation.split("; "):
                print(f"       - {reason}")


if __name__ == "__main__":
    main()
