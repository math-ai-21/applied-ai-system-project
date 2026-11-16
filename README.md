# 🎵 TaskTunes

A music recommender that turns what you're doing, or your whole to-do list, into songs, using retrieval-augmented generation (RAG) with Gemini.

## Original Project: Music Recommender Simulation (Modules 1–3)

My original project was a **Music Recommender Simulation**. It compared a listener's taste profile (favorite genre and mood, target energy, acousticness and valence) to a catalog of 18 songs and gave each song a score from 0 to 1, with energy weighted most heavily and genre and mood scoring only on an exact match. It returned the top five songs with the reasons each feature added to the score, and I later extended it with popularity, release decade, language, vocal type and mood tags.

## Title and Summary

TaskTunes lets you describe what you want to hear ("cozy rainy day at home") or list your tasks for the day ("study for my exam", "run in the park") and recommends songs from the catalog. A retriever finds matching guides in a 46-entry knowledge base, the guides become a taste profile, my original scorer ranks the songs, and Gemini can explain the picks using only what was retrieved.

It matters because the AI part is checkable. You can see which guides were retrieved and why each song scored the way it did, and everything still works, without Gemini, when the API is unavailable.

## Architecture Overview

A query or task list passes the input guardrails (query length, task limit, Gemini call cap), then the retriever matches it against the knowledge base. The profile builder turns the matched guides into a target profile, and the recommender scores and ranks the catalog songs against it. Gemini explains the ranked songs, and if it fails the plain ranked list is shown. In LLM-only mode, Gemini picks the songs itself and an output guardrail drops any that aren't in the catalog. pytest covers the retriever, profile builder, recommender, guardrails and prompts, and a person checks the retrieved guides, score reasons, answer and log. The full diagram is in [diagrams/architecture.mmd](diagrams/architecture.mmd).

The app has five tabs: **Samples** (hand-written profiles, no AI), **Retrieval** (no Gemini), **RAG** (retrieval plus a Gemini explanation), **LLM-only** (Gemini picks from the catalog with no retrieval, as a comparison) and **Day planner** (one song per task).

---

## Setup Instructions

1. Create and activate a virtual environment. I used Python 3.14.

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

   Keep it activated whenever you run the project. If you see `No module named 'dotenv'`, the environment isn't active.

2. Install the dependencies (pinned to the versions I tested):

   ```bash
   pip install -r requirements.txt
   ```

3. Add a Gemini API key. Only the Gemini features need it; everything else works without it.

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and replace `your-key-here` with your own key from https://aistudio.google.com/apikey.

4. Run it:

   ```bash
   python -m src.main                 # terminal menu
   python -m streamlit run app.py     # web app
   ```

5. Run the tests (85 tests, no key or internet needed, because Gemini is replaced with a fake client):

   ```bash
   pytest
   ```

The app writes a log to `logs/app.log`.

---

## Sample Interactions

**1. Sample profile, "Chill Lofi" (no AI):** a hand-written profile scored by the original recommender.

```
Chill Lofi
User profile: genre=lofi, mood=chill, energy=0.35, acousticness=0.8, valence=0.58
  1. Library Rain - Score: 0.99
       - energy 0.35 vs target 0.35 (+0.30)
       - acousticness 0.86 vs target 0.80 (+0.14)
       - valence 0.60 vs target 0.58 (+0.10)
       - mood match: chill (+0.25)
       - genre match: lofi (+0.20)
  2. Midnight Coding - Score: 0.96
  3. Spacewalk Thoughts - Score: 0.75
```

**2. Retrieval, input: "cozy rainy day at home":** the matching guides become the profile, and the same scorer ranks the songs.

```
Retrieved:
  - rainy day and cozy evenings (scenario, similarity 0.38)
  - Take Me Home, Country Roads by John Denver (reference, similarity 0.23)
Profile: energy=0.36, acousticness=0.8, valence=0.58, genre=folk, mood=relaxed, mood_tags=['nostalgic', 'reflective', 'dreamy']
  1. Coffee Shop Stories - Score: 0.76
       - energy 0.37 vs target 0.36 (+0.26)
       - acousticness 0.89 vs target 0.80 (+0.12)
       - valence 0.71 vs target 0.58 (+0.08)
       - mood match: relaxed (+0.22)
       - mood tags match: nostalgic, reflective (+0.09)
  2. Autumn Letters - Score: 0.68
  3. Library Rain - Score: 0.60
```

**3. RAG, input: "cozy rainy day at home":** the same retrieval, then Gemini explains the picks using only the retrieved guides.

```
Retrieved:
  - rainy day and cozy evenings (scenario, similarity 0.38)
  - Take Me Home, Country Roads by John Denver (reference, similarity 0.23)
Profile: energy=0.36, acousticness=0.8, valence=0.58, genre=folk, mood=relaxed, mood_tags=['nostalgic', 'reflective', 'dreamy']

Answer:
Based on your request for a cozy rainy day at home, these songs fit the provided guides because a rainy day and cozy evenings call for staying in, a warm blanket, tea or coffee, and window watching. The guides specify gentle slow music with a soft nostalgic or dreamy feel, low energy, acoustic warmth, instrumental textures, and a hint of melancholy. Additionally, the guides note that a warm, sing-along country-folk style brings a homesick, nostalgic feeling that is cozy, sincere, and heartfelt.

Here are the recommended songs in order:
1. Coffee Shop Stories by Slow Stereo
2. Autumn Letters by Willow & Pine
3. Library Rain by Paper Lanterns
4. Midnight Coding by LoRoom
5. Spacewalk Thoughts by Orbit Bloom
Recommendations:
  1. Coffee Shop Stories - Score: 0.76
       - energy 0.37 vs target 0.36 (+0.26)
       - acousticness 0.89 vs target 0.80 (+0.12)
       - valence 0.71 vs target 0.58 (+0.08)
       - mood match: relaxed (+0.22)
       - mood tags match: nostalgic, reflective (+0.09)
  2. Autumn Letters - Score: 0.68
  3. Library Rain - Score: 0.60
  4. Midnight Coding - Score: 0.54
  5. Spacewalk Thoughts - Score: 0.52
```

**4. LLM-only, input: "funny song":** no retrieval; Gemini picks from the catalog, and any title not in the catalog is dropped.

```
LLM-only picks:
  1. Sunrise City by Neon Echo
       - None of the songs in the catalog are explicitly funny, but this upbeat pop track has a bright and cheerful energy that feels lighthearted.
  2. Rooftop Lights by Indigo Parade
       - This song is not comedic, but it was chosen because its happy indie pop mood brings a fun and lively vibe compared to the other tracks.
  3. Gym Hero by Max Pulse
       - While not a comedy song, the high-energy and intense tone could playfully fit an over-the-top workout parody.
```

**5. Day planner, tasks: "running the park", "eating lunch", "doing math homework", "painting with my friend", "yoga for 3 hours":** each task gets a different song, and a task with no matching guide says so.

```
running the park        -> Sunrise City by Neon Echo (score 0.77; matched: outdoor exercise, jogging and sports)
eating lunch            -> Coffee Shop Stories by Slow Stereo (score 0.89; matched: cooking, baking and making dinner)
doing math homework     -> No match for this task. Try describing it differently.
painting with my friend -> Rooftop Lights by Indigo Parade (score 0.75; matched: road trip and summer drive)
yoga for 3 hours        -> Winter Cathedral by Elara Voss (score 0.74; matched: winding down, meditation and sleep)
```

Gemini's "Your day, explained" note for the same plan:

```
Task: running the park Song: Sunrise City by Neon Echo (pop, happy) This song fits running the park because it is upbeat, steady, and motivating with a bouncy tempo, medium-high energy, and a positive feeling, which suits outdoor exercise, jogging, and sports.

Task: eating lunch Song: Coffee Shop Stories by Slow Stereo (jazz, relaxed) This song fits eating lunch because it is warm, relaxed, and groovy with a mid tempo, medium energy, and a friendly feeling, which suits cooking, baking, and making dinner.

Task: painting with my friend Song: Rooftop Lights by Indigo Parade (indie pop, happy) This song fits painting with my friend because it is an upbeat, bright, sing-along song with warm vocals, a positive feeling, and medium-high energy, matching a road trip and summer drive or a party, dancing, and celebration.

Task: yoga for 3 hours Song: Winter Cathedral by Elara Voss (classical, peaceful) This song fits yoga for 3 hours because it is very slow, very quiet, and soft instrumental music with almost no rhythm, featuring the lowest energy of any scenario to support stretching or yoga, winding down, meditation, and sleep.

The overall mood of the day transitions smoothly from high-energy motivation and friendly warmth to feel-good connection and deeply peaceful relaxation.
```

## Design Decisions

- **I kept the original scorer as the ranking engine.** It scores every song from 0 to 1 by comparing it to a profile, and returns the top five with the reasons each feature added. The weights are my own judgment, not learned from data: energy 0.30, mood 0.25, genre 0.20, acousticness 0.15 and valence 0.10, plus optional mood tags (0.15) and popularity, decade, language and vocal type (0.05 each), which only count when a profile includes them. Energy, acousticness and valence score by closeness (`1 - |song - target|`), and genre and mood by exact match. The trade-off is that exact matching gives no partial credit, so "chill" and "relaxed" count as different, and the system mostly recommends songs that look like what was asked for.
- **Retrieval uses TF-IDF with a small stemmer, not embeddings.** It runs offline, gives the same answer every time, and uses no API quota. The cost is that it matches words and not meaning, so a task like "meeting with my boss" finds nothing.
- **The knowledge base includes 10 reference entries for well-known songs** (such as Blinding Lights, Levitating and Clair de Lune) that are not in the 18-song catalog. A query that names one retrieves its entry, which describes the song's feel, and the recommender then finds catalog songs that fit that description.
- **Gemini explains the ranking, and the recommender decides it.** The recommender picks the songs, which keeps results consistent and testable, and Gemini only writes the explanation from the retrieved guides. The trade-off is that it's less flexible than letting the LLM choose.
- **LLM-only mode exists as a baseline** to compare RAG against. A catalog-only filter stops it from recommending songs the app doesn't have.
- **Guardrails.** Each is labeled `GUARDRAIL` in the code and indexed in `src/guardrails.py`:

  | # | Guardrail | What it does |
  |---|---|---|
  | 1 | Knowledge base validation | Refuses entries with missing fields, unknown kinds, duplicate ids or out-of-range numbers |
  | 2 | Retrieval threshold | Drops guides below 0.15 similarity, so vague queries return no match |
  | 3 | Query length limit | 200 characters |
  | 4 | Gemini call cap | 5 calls per run (terminal) or per session (app) |
  | 5 | LLM-unavailable fallback | Shows the plain ranked results if Gemini fails |
  | 6 | Grounding instruction | Tells Gemini to use only the retrieved guides and the listed songs |
  | 7 | Catalog-only picks | In LLM-only mode, drops any title Gemini picks that isn't in the catalog |
  | 8 | Task list limit | 5 tasks per plan |

- **Logging.** Queries, retrieved guides, Gemini calls and guardrail triggers go to a rotating `logs/app.log`. The API key and the full prompts and replies are never logged.

---

## Testing Summary

- **Automated tests:** 85 tests, all passing, with no key or internet needed because Gemini is replaced with a fake client. They are split across six files in `tests/`:
  - `test_retriever.py` (33): checks that queries find the right guide, that unrelated queries return nothing, how the profile is built from the guides, and that the day planner gives each task a different song.
  - `test_llm.py` (16): checks the Gemini wrapper's error handling, the contents of the prompts, that songs not in the catalog are dropped, and that the API key is never logged.
  - `test_recommender.py` (12): checks that the original scorer keeps scores between 0 and 1, sorts results, and handles the extra fields.
  - `test_guardrails.py` (11): checks the query and task limits, the 5-call cap, and that each guardrail logs when it triggers.
  - `test_knowledge.py` (10): checks that every catalog genre has a guide and that bad knowledge base entries are rejected.
  - `test_log.py` (3): checks that the log file is written, handlers aren't duplicated, and nothing prints to the terminal.
- **Manual checks:** I ran the sample profiles, then tried queries by hand in each mode and checked which guides were retrieved and whether the songs fit. I ran the live Gemini calls myself for the RAG, LLM-only and Day planner samples above.
- **What didn't, and how I fixed it:**
  - The default Gemini model timed out with a 504, so I switched models.
  - Generic words like "like" and "want" matched unrelated guides, so I added a list of filler words to ignore.
  - "run in the park" matched nothing because the search didn't treat "run" and "running" as the same word, so I added a stemmer and four everyday knowledge entries.
  - LLM-only printed nothing for "funny song" when Gemini returned an empty list, so it now says so and the prompt asks for the closest matches.
- **Original experiment:** doubling energy's weight (0.30 to 0.60) and halving genre's (0.20 to 0.10) kept the top song the same for all four profiles but changed the middle of two lists, for example Rooftop Lights moving above Gym Hero for High-Energy Pop. The results were different but not better, so I kept the original weights.
- **What I learned:** Retrieval quality came down to small details, like filler words and word forms, more than to the model. Writing the fallback and the fake-Gemini tests first let me change the AI parts without breaking things.

---

## Reflection

Building TaskTunes showed me that the AI part of a system is only as trustworthy as the checks around it. The Gemini calls were the easiest part to add and the hardest to rely on as the default model timed out, an empty reply printed nothing, and filler words sometimes pulled in the wrong guides. What helped most was keeping my original scorer in charge of the ranking, so Gemini only explains results I can verify, and writing tests with a fake Gemini so I could change things safely. It also reminded me to start with the simplest approach that works, such as using TF-IDF before embeddings.

See AI reflection and limitations in [model_card.md](model_card.md).
