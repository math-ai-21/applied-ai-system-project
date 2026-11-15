"""Streamlit app for the Music Recommender: pick a mode, ask, see results.

Run with: streamlit run app.py
"""

import streamlit as st

from src.guardrails import MAX_LLM_CALLS, CallBudget, query_too_long
from src.knowledge import load_knowledge
from src.llm import LLMUnavailable, build_llm_only_prompt, build_rag_prompt, generate, parse_llm_picks
from src.main import PROFILES
from src.recommender import Recommender
from src.retriever import Retriever, recommend_from_query

st.set_page_config(page_title="Music Recommender")


@st.cache_resource
def load_resources():
    return (Recommender.load_songs("data/songs.csv"),
            Retriever(load_knowledge("data/knowledge_base.json")))


def show_songs(recommendations):
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        st.markdown(f"**{rank}. {song['title']}** by {song['artist']} - score {score:.2f}")
        st.markdown("\n".join(f"- {reason}" for reason in explanation.split("; ")))


songs, retriever = load_resources()
if "budget" not in st.session_state:
    st.session_state.budget = CallBudget()  # GUARDRAIL 4, call cap: one budget per browser session
budget = st.session_state.budget

st.title("Music Recommender")
mode = st.sidebar.radio("Mode", ["samples", "retrieval", "rag", "llm-only"])
st.sidebar.write(f"Gemini calls used: {budget.used}/{budget.limit}")

if mode == "samples":
    name = st.selectbox("Sample profile", list(PROFILES))
    st.write(PROFILES[name])
    show_songs(Recommender.recommend(PROFILES[name], songs, k=5))
    st.stop()

with st.form("query"):
    query = st.text_input("Describe what you want to hear").strip()
    submitted = st.form_submit_button("Search")
if not submitted or not query:
    st.stop()
if query_too_long(query):  # GUARDRAIL 3, query length
    st.error("Query too long. Please shorten it.")
    st.stop()

if mode == "llm-only":
    if not budget.try_use():  # GUARDRAIL 4, call cap
        st.error(f"Reached the limit of {MAX_LLM_CALLS} Gemini calls this session.")
        st.stop()
    try:
        picks, ignored = parse_llm_picks(generate(build_llm_only_prompt(query, songs)), songs)
    except LLMUnavailable as exc:  # GUARDRAIL 5, LLM-unavailable fallback
        st.warning(f"LLM unavailable ({exc}).")
        st.stop()
    if ignored:
        st.warning(f"Ignored {len(ignored)} title(s) not in the catalog: {', '.join(ignored)}")
    for rank, (song, reason) in enumerate(picks, start=1):
        st.markdown(f"**{rank}. {song['title']}** by {song['artist']}  \n{reason}")
    st.stop()

results, profile, recommendations = recommend_from_query(query, retriever, songs)
if not results:
    st.warning("No knowledge entry matched. Try rephrasing, e.g. 'studying on a rainy day'.")
    st.stop()

if mode == "rag":
    if not budget.try_use():  # GUARDRAIL 4, call cap
        st.info(f"Reached the limit of {MAX_LLM_CALLS} Gemini answers; showing retrieval results only.")
    else:
        try:
            answer = generate(build_rag_prompt(query, results, recommendations))
            st.subheader("Answer")
            st.write(answer)
        except LLMUnavailable as exc:  # GUARDRAIL 5, LLM-unavailable fallback
            st.warning(f"LLM unavailable ({exc}); showing retrieval results only.")

st.subheader("Recommendations")
show_songs(recommendations)
with st.expander("How this was retrieved"):
    for entry, score in results:
        st.write(f"{entry['name']} ({entry['kind']}, similarity {score:.2f})")
    st.write(profile)
