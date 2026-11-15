"""
Command line runner for the Music Recommender Simulation.

Loads the song catalog from data/songs.csv and asks which mode to run:
samples scores it against each sample listener profile in PROFILES, retrieval
lets you describe what you want in plain text and matches it against the
knowledge base, and rag does the same and adds a Gemini answer grounded in what
was retrieved. Those three print the top 5 songs with the reasons behind each score.
llm-only skips retrieval: Gemini picks songs from the catalog using its own knowledge.

Run with: python -m src.main, then pick a mode
"""

from src.guardrails import MAX_LLM_CALLS, MAX_QUERY_CHARS, CallBudget, query_too_long
from src.knowledge import load_knowledge
from src.llm import (
    LLMUnavailable, build_llm_only_prompt, build_rag_prompt, generate, parse_llm_picks,
)
from src.recommender import Recommender
from src.retriever import Retriever, recommend_from_query


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


def ask_query():
    """Ask for a query; returns None to quit. Re-asks if the query is too long."""
    while True:
        try:
            query = input("\nQuery> ").strip()
        except EOFError:
            return None
        if not query:
            return None
        if not query_too_long(query):
            return query
        print(f"Query too long ({len(query)} characters, max {MAX_QUERY_CHARS}). Please shorten it.")


def run_query_loop(songs, use_llm: bool) -> None:
    """Ask for plain-text queries until a blank line: retrieve, build a profile, recommend.

    With use_llm, Gemini also writes an answer from the retrieved guides and the ranked songs.
    """
    retriever = Retriever(load_knowledge("data/knowledge_base.json"))
    label = "RAG" if use_llm else "Retrieval"
    print(f"{label} mode: describe what you want to hear (blank line to quit).")
    budget = CallBudget()
    while True:
        query = ask_query()
        if query is None:
            break

        results, profile, recommendations = recommend_from_query(query, retriever, songs)
        if not results:
            print("No knowledge entry matched. Try rephrasing, e.g. 'studying on a rainy day'.")
            continue

        print("Retrieved:")
        for entry, score in results:
            print(f"  - {entry['name']} ({entry['kind']}, similarity {score:.2f})")
        print("Profile: " + ", ".join(f"{key}={value}" for key, value in profile.items()))

        if use_llm:
            if not budget.try_use():  # GUARDRAIL 4, call cap
                print(f"\nReached the limit of {MAX_LLM_CALLS} Gemini answers this run; "
                      "showing retrieval results only.")
            else:
                try:
                    print("\nAnswer:\n" + generate(build_rag_prompt(query, results, recommendations)))
                except LLMUnavailable as exc:  # GUARDRAIL 5, LLM-unavailable fallback
                    print(f"\nLLM unavailable ({exc}); showing retrieval results only.")
        print_recommendations(recommendations)


def run_llm_only(songs) -> None:
    """Ask Gemini to pick songs from the catalog using its own knowledge. No retrieval."""
    print("LLM-only mode: describe what you want to hear (blank line to quit).")
    budget = CallBudget()
    while True:
        query = ask_query()
        if query is None:
            break
        if not budget.try_use():  # GUARDRAIL 4, call cap
            print(f"Reached the limit of {MAX_LLM_CALLS} Gemini calls this run.")
            break
        try:
            picks, ignored = parse_llm_picks(generate(build_llm_only_prompt(query, songs)), songs)
        except LLMUnavailable as exc:  # GUARDRAIL 5, LLM-unavailable fallback
            print(f"LLM unavailable ({exc}).")
            continue
        if ignored:
            print(f"Ignored {len(ignored)} title(s) not in the catalog: {', '.join(ignored)}")
        if not picks:
            print("Gemini didn't name any songs from the catalog.")
            continue
        print("LLM-only picks:")
        for rank, (song, reason) in enumerate(picks, start=1):
            print(f"  {rank}. {song['title']} by {song['artist']}\n       - {reason}")


def run_regular(songs) -> None:
    """Score the catalog against each sample profile and print the top 5 for each."""
    for name, user_prefs in PROFILES.items():
        recommendations = Recommender.recommend(user_prefs, songs, k=5)

        profile = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
        print(f"\n{name}")
        print(f"User profile: {profile}")
        print_recommendations(recommendations)


def choose_mode() -> str:
    """Ask which mode to run; returns 'samples', 'retrieval', 'rag' or 'llm-only', or '' to quit."""
    while True:
        print("\nWhich mode do you want?")
        print("  1) samples   - sample listener profiles")
        print("  2) retrieval - describe what you want; matched against the knowledge base")
        print("  3) rag       - retrieval, plus a Gemini answer grounded in it")
        print("  4) llm-only  - Gemini picks from the catalog, no retrieval")
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
            return "llm-only"
        if choice in ("", "q"):
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
    elif mode == "llm-only":
        run_llm_only(songs)


if __name__ == "__main__":
    main()
