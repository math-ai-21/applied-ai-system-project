"""Streamlit app for the Music Recommender: pick a tab, ask, see results.

Run with: streamlit run app.py
"""

import streamlit as st

from src.guardrails import MAX_LLM_CALLS, MAX_TASKS, CallBudget, query_too_long, task_list_full
from src.knowledge import load_knowledge
from src.llm import (
    LLMUnavailable, build_llm_only_prompt, build_plan_prompt, build_rag_prompt, generate,
    parse_llm_picks,
)
from src.main import PROFILES
from src.recommender import Recommender
from src.retriever import Retriever, plan_day, recommend_from_query

st.set_page_config(page_title="Music Recommender", page_icon="🎵")

MODE_HELP = """
- **Samples**: five ready-made listener profiles, scored by the recommender.
- **Retrieval**: describe what you want; it's matched against the knowledge base. No AI model.
- **RAG**: retrieval, then Gemini explains the picks using only what was retrieved.
- **LLM-only**: Gemini picks songs from the catalog on its own, with no retrieval.
- **Day planner**: list your tasks for the day and get one song for each, explained by Gemini.
"""


@st.cache_resource
def load_resources():
    return (Recommender.load_songs("data/songs.csv"),
            Retriever(load_knowledge("data/knowledge_base.json")))


def show_songs(recommendations):
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {song['title']}** by {song['artist']} · score {score:.2f}")
            st.caption("\n".join(f"- {reason}" for reason in explanation.split("; ")))


def ask_query(key):
    """Show a search box; return the query once submitted, or None."""
    with st.form(key):
        query = st.text_input("Describe what you want to hear", key=f"{key}_text").strip()
        submitted = st.form_submit_button("Search")
    if not submitted or not query:
        return None
    if query_too_long(query):  # GUARDRAIL 3, query length
        st.error("Query too long. Please shorten it.")
        return None
    return query


def show_samples():
    name = st.selectbox("Sample profile", list(PROFILES))
    st.write(PROFILES[name])
    show_songs(Recommender.recommend(PROFILES[name], songs, k=5))


def show_query_tab(key, use_llm):
    """Retrieval tab, or the RAG tab when use_llm is set (adds a Gemini answer)."""
    query = ask_query(key)
    if query is None:
        return
    results, profile, recommendations = recommend_from_query(query, retriever, songs)
    if not results:
        st.warning("No knowledge entry matched. Try rephrasing, e.g. 'studying on a rainy day'.")
        return

    if use_llm:
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


def show_llm_only_tab():
    query = ask_query("llm_only")
    if query is None:
        return
    if not budget.try_use():  # GUARDRAIL 4, call cap
        st.error(f"Reached the limit of {MAX_LLM_CALLS} Gemini calls this session.")
        return
    try:
        picks, ignored = parse_llm_picks(generate(build_llm_only_prompt(query, songs)), songs)
    except LLMUnavailable as exc:  # GUARDRAIL 5, LLM-unavailable fallback
        st.warning(f"LLM unavailable ({exc}).")
        return
    if ignored:
        st.warning(f"Ignored {len(ignored)} title(s) not in the catalog: {', '.join(ignored)}")
    if not picks:
        st.info("Gemini didn't name any songs from the catalog.")
        return
    for rank, (song, reason) in enumerate(picks, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {song['title']}** by {song['artist']}")
            st.caption(reason)


def plan_note(items):
    """One Gemini note for the whole plan: returns (kind, text), or None if no task got a song."""
    if not any(item["pick"] for item in items):
        return None
    if not budget.try_use():  # GUARDRAIL 4, call cap
        return ("info", f"Reached the limit of {MAX_LLM_CALLS} Gemini calls; showing the songs only.")
    try:
        return ("ok", generate(build_plan_prompt(items)))
    except LLMUnavailable as exc:  # GUARDRAIL 5, LLM-unavailable fallback
        return ("warning", f"LLM unavailable ({exc}); showing the songs only.")


def show_planner():
    """A to-do checklist: add tasks, tick them off, get one song per task and a Gemini note."""
    state = st.session_state
    state.setdefault("tasks", [])        # [{"id": int, "text": str}]
    state.setdefault("next_id", 0)
    state.setdefault("plan", {})         # task id -> plan_day result for that task
    state.setdefault("plan_note", None)  # (kind, text) for the Gemini note

    if task_list_full(len(state.tasks)):  # GUARDRAIL 8, task list limit
        st.info(f"Task limit reached ({MAX_TASKS}). Remove one to add another.")
    else:
        with st.form("add_task", clear_on_submit=True):
            text = st.text_input("Add a task").strip()
            added = st.form_submit_button("Add")
        if added and text:
            if query_too_long(text):  # GUARDRAIL 3, query length
                st.error("Task too long. Please shorten it.")
            else:
                state.tasks.append({"id": state.next_id, "text": text})
                state.next_id += 1
                state.plan, state.plan_note = {}, None
                st.rerun()

    for task in list(state.tasks):
        row, remove = st.columns([9, 1])
        done = state.get(f"done_{task['id']}", False)
        row.checkbox(f"~~{task['text']}~~" if done else task["text"], key=f"done_{task['id']}")
        if remove.button("✕", key=f"remove_{task['id']}"):
            state.tasks.remove(task)
            state.plan, state.plan_note = {}, None
            st.rerun()
        item = state.plan.get(task["id"])
        if item and item["pick"] is None:
            row.caption("No match for this task. Try describing it differently.")
        elif item:
            song, score, _ = item["pick"]
            row.caption(f"🎵 {song['title']} by {song['artist']} · score {score:.2f} "
                        f"(matched: {item['results'][0][0]['name']})")

    if state.tasks and st.button("Plan my day"):
        items = plan_day([task["text"] for task in state.tasks], retriever, songs)
        state.plan = {task["id"]: item for task, item in zip(state.tasks, items)}
        state.plan_note = plan_note(items)
        st.rerun()

    if state.plan_note:
        kind, text = state.plan_note
        if kind == "ok":
            st.subheader("Your day, explained")
            st.write(text)
        else:
            getattr(st, kind)(text)


songs, retriever = load_resources()
if "budget" not in st.session_state:
    st.session_state.budget = CallBudget()  # GUARDRAIL 4, call cap: one budget per browser session
budget = st.session_state.budget

st.title("🎵 Music Recommender")
st.markdown(MODE_HELP)
counter = st.empty()  # filled in at the end, so it never lags a call behind

samples_tab, retrieval_tab, rag_tab, llm_tab, planner_tab = st.tabs(
    ["Samples", "Retrieval", "RAG", "LLM-only", "Day planner"]
)
with samples_tab:
    show_samples()
with retrieval_tab:
    show_query_tab("retrieval", use_llm=False)
with rag_tab:
    show_query_tab("rag", use_llm=True)
with llm_tab:
    show_llm_only_tab()
with planner_tab:
    show_planner()

counter.caption(f"Gemini calls used this session: {budget.used}/{budget.limit}")
