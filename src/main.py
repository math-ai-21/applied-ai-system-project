"""
Command line runner for the Music Recommender Simulation.

Loads the song catalog from data/songs.csv and asks which mode to run:
samples scores it against each sample listener profile in PROFILES, retrieval
lets you describe what you want in plain text and matches it against the
knowledge base, and rag does the same and adds a Gemini answer grounded in what
was retrieved. All of them print the top 5 songs with the reasons behind each score.

Run with: python -m src.main, then pick a mode
"""

from src.knowledge import load_knowledge
from src.llm import LLMUnavailable, build_rag_prompt, generate
from src.recommender import Recommender
from src.retriever import Retriever, build_profile

MAX_QUERY_CHARS = 200
MAX_LLM_CALLS = 5


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


def run_query_loop(songs, use_llm: bool) -> None:
    """Ask for plain-text queries until a blank line: retrieve, build a profile, recommend.

    With use_llm, Gemini also writes an answer from the retrieved guides and the ranked songs.
    """
    retriever = Retriever(load_knowledge("data/knowledge_base.json"))
    label = "RAG" if use_llm else "Retrieval"
    print(f"{label} mode: describe what you want to hear (blank line to quit).")
    llm_calls = 0
    while True:
        try:
            query = input("\nQuery> ").strip()
        except EOFError:
            break
        if not query:
            break
        if len(query) > MAX_QUERY_CHARS:
            print(f"Query too long ({len(query)} characters, max {MAX_QUERY_CHARS}). Please shorten it.")
            continue

        results = retriever.retrieve(query)
        if not results:
            print("No knowledge entry matched. Try rephrasing, e.g. 'studying on a rainy day'.")
            continue

        print("Retrieved:")
        for entry, score in results:
            print(f"  - {entry['name']} ({entry['kind']}, similarity {score:.2f})")
        profile = build_profile(results)
        print("Profile: " + ", ".join(f"{key}={value}" for key, value in profile.items()))
        recommendations = Recommender.recommend(profile, songs, k=5)

        if use_llm:
            if llm_calls >= MAX_LLM_CALLS:
                print(f"\nReached the limit of {MAX_LLM_CALLS} Gemini answers this run; "
                      "showing retrieval results only.")
            else:
                llm_calls += 1
                try:
                    print("\nAnswer:\n" + generate(build_rag_prompt(query, results, recommendations)))
                except LLMUnavailable as exc:
                    print(f"\nLLM unavailable ({exc}); showing retrieval results only.")
        print_recommendations(recommendations)


def run_regular(songs) -> None:
    """Score the catalog against each sample profile and print the top 5 for each."""
    for name, user_prefs in PROFILES.items():
        recommendations = Recommender.recommend(user_prefs, songs, k=5)

        profile = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
        print(f"\n{name}")
        print(f"User profile: {profile}")
        print_recommendations(recommendations)


def choose_mode() -> str:
    """Ask which mode to run; returns 'samples', 'retrieval' or 'rag', or '' to quit."""
    while True:
        print("\nWhich mode do you want?")
        print("  1) samples   - sample listener profiles")
        print("  2) retrieval - describe what you want; matched against the knowledge base")
        print("  3) rag       - retrieval, plus a Gemini answer grounded in it")
        print("  4) llm-only  - coming soon")
        try:
            choice = input("Choice> ").strip().lower()
        except EOFError:
            return ""
        if choice in ("1", "samples"):
            return "samples"
        if choice in ("2", "retrieval"):
            return "retrieval"
        if choice in ("3", "rag"):
            return "rag"
        if choice in ("4", "llm-only", "llm"):
            print("LLM-only mode isn't built yet. Pick another mode for now.")
        elif choice in ("", "q"):
            return ""
        else:
            print("Please enter 1, 2, 3 or 4.")


def main() -> None:
    songs = Recommender.load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs from data/songs.csv")

    mode = choose_mode()
    if mode == "samples":
        run_regular(songs)
    elif mode == "retrieval":
        run_query_loop(songs, use_llm=False)
    elif mode == "rag":
        run_query_loop(songs, use_llm=True)


if __name__ == "__main__":
    main()
