# 🎧 Model Card: TaskTunes

## 1. Model Name  
**TaskTunes**, which extends **TuneMatcher 1.0**, my original Music Recommender Simulation from Modules 1–3.  

---

## 2. Intended Use  

This recommender suggests up to five songs from a small catalog that best fit what a listener wants. In the original version (TuneMatcher 1.0), the listener gave a favorite genre and mood plus target levels for energy, acousticness and valence, and the system ranked the songs by how closely they matched, with a short list of reasons for each one. TaskTunes adds a second way in: the listener describes what they want in plain text ("cozy rainy day at home") or lists the tasks for their day, and a retriever turns that description into the same kind of profile. Gemini can then explain the picks, using only the guides the retriever found.

It is meant for exploring how recommenders and retrieval-augmented systems turn data into suggestions, run by one person on their own machine. The catalog has only 18 songs, the knowledge base has 46 short guides that I wrote with help from Claude, and it hasn't been tested with actual listeners yet. It assumes that the listener's words overlap with the words in the guides, and that the genre and mood labels mean the same thing to them as to the system. It is a general guide for music choices, and it is not a source of facts about songs or artists, since Gemini is told to use only the retrieved guides.

---

## 3. How the Model Works  

- The recommender gives each song a score from 0 to 1 based on how well it matches the user's profile, with 1 being the best match. It considers energy, mood, genre, acousticness and valence. For energy, acousticness and valence (how positive a song sounds), the closer a song is to what the user prefers, the more points it gets. For mood and genre, a song gets points only if it matches the user's choice.
- Some parts matter more than others. Energy makes up 30% of the score, mood 25%, genre 20%, acousticness 15% and valence 10%. These percentages were chosen by hand, with energy weighted most because it is one of the main parts of how a song feels. A profile can also include mood tags (0.15) and popularity, release decade, language and vocal type (0.05 each), which count only when the profile asks for them. If a profile leaves a feature out, the model ignores it and scales the remaining percentages up. The recommender shows the five highest-scoring songs and how much each part contributed.
- **TaskTunes extends this scorer.** When the listener types a description, a retriever compares the words (after trimming word endings, so "running" matches "run") against the 46 guides in the knowledge base and keeps the ones that are similar enough. If none are, the system says there is no match. The matching guides vote on a profile: genre and mood go to the label with the most weight, energy, acousticness and valence are averaged, and the top three mood tags are kept. That profile goes into the same scorer. Gemini only explains the result, using the retrieved guides and the ranked songs. 

---

## 4. Data  
The catalog has 18 songs in data/songs.csv. Each song has a title, artist, genre, mood, five numbers (energy, tempo, valence, danceability and acousticness), and extra fields for popularity, release decade, language, vocal type and mood tags. The songs cover 15 genres and 14 moods, but most appear only once. Only lofi (3 songs) and pop (2 songs) have more than one, and only chill, happy and intense appear for more than one song. The starter file had 10 songs, and 8 more were added to bring in genres and moods that were missing, such as hip hop, classical, country, r&b, EDM, reggae, metal and folk. None of the songs were removed. Many parts of musical taste are still missing: there are no lyrics and no artist details beyond the name, and the energy values skip the range 0.58 to 0.75, so listeners who want medium-high energy have few options.

TaskTunes also uses a knowledge base of 46 short guides in data/knowledge_base.json. Each guide has an id, a kind, a name and some text, plus the genres, moods, number targets and mood tags it stands for. There are four kinds: genre guides (15), mood guides (6), scenario guides for everyday situations such as studying, a rainy day, cooking, a commute or running outdoors (15), and reference guides for specific well-known songs (10). I wrote the guides with help from Claude, and I checked them by running test queries. The knowledge base is small and in English only. Matching is by words, not meaning, so a request like "meeting with my boss" finds nothing, and a request phrased very differently from a guide's wording can miss it. The guides also reflect my own view of what a "study" or "workout" song sounds like, which other listeners may not share.

---

## 5. Strengths  

The system works best for listeners with one consistent taste, especially in a genre with several songs in the catalog. For the Chill Lofi profile, Library Rain (0.99) and Midnight Coding (0.96) were strong matches. The scoring also separates listeners well: Chill Lofi shares no songs with the other three standard profiles, and High-Energy Pop and Deep Intense Rock both want energy around 0.9 but get different top songs (Sunrise City vs. Storm Runner) because of their genre, mood and valence. The results matched what is expected: Deep Intense Rock gets loud, dark songs like Storm Runner and Iron Requiem, and Chill Lofi gets quiet, acoustic ones like Library Rain. Also, every recommendation lists the reasons behind its score, so it is easy to see why a song was chosen.

The extra fields work too. The Nostalgic Dreamer profile asks for fairly low energy, high acousticness, a 1990s release and the tags nostalgic, reflective and melancholic. Its top results were Dusty Porch Light (0.89, from 1990) and Autumn Letters (0.88, from 1970), which both match its mood tags and acousticness, though only Dusty Porch Light is from the decade it asked for.

TaskTunes also works well when the request matches a guide closely. In my test queries, "study for my exam" found the studying guide and ranked Focus Flow first (0.92), "cook dinner" found the cooking guide and ranked Coffee Shop Stories first (0.86), and "cozy rainy day at home" found the rainy-day guide. The results can be checked: the app shows which guides were retrieved and why each song scored the way it did. The system also keeps working without Gemini, because the plain ranked list is shown if the API fails, and in LLM-only mode any song Gemini names that isn't in the catalog is dropped.

---

## 6. Limitations and Bias

The system gives better recommendations to listeners whose favorite genres have more songs in the catalog. The catalog has only 18 songs, and most genres appear just once. Since a song only gets credit for genre if it matches exactly, some listeners have fewer good options.

For example, a jazz profile (jazz, relaxed, energy 0.38, acousticness 0.8, valence 0.6) gets one strong match, Coffee Shop Stories, with a score of 0.97, and the rest of its top five score only around 0.5. Those songs are mostly lofi and country songs with similar energy and acousticness, because there are no other jazz songs in the catalog. A lofi profile with the same numbers (lofi, chill) gets two matches near 0.97 and two more around 0.75. This means the system might favor listeners whose favorite genres are more common in the catalog.

TaskTunes adds limits of its own. Retrieval matches words, not meaning, so a request like "meeting with my boss" finds nothing, and one phrased very differently from a guide's wording can miss it. A query that names a song, such as "something like Blinding Lights", also blends in other retrieved guides, so the results drift from the song named. The guides are my own judgment of what a task sounds like, so the system pushes everyone toward my idea of "study music" or "workout music". Gemini's explanations are not checked automatically for invented claims, and it can only be told to stay inside the retrieved guides. The system is also in English only, has 18 songs, and has not been tested with real listeners.

The day planner shows these limits clearly. "doing math homework" found no match, even though the studying guide includes the word "homework", because the extra word "doing" pulled the similarity below the 0.15 cutoff ("math homework" on its own does match it). "eating lunch" matched the cooking guide and "painting with my friend" matched the road trip guide, both through loosely related words, and Gemini then explained them as if they fit.

---

## 7. Evaluation

All 85 automated tests pass, with Gemini replaced by a fake client so they need no key or internet. I also checked the system by hand in each mode.

- **RAG, "cozy rainy day at home":** it found the rainy-day guide and a country-folk reference, and Gemini's list matched the recommender's ranking in the same order, so it explained the ranking and didn't change it.
- **LLM-only, "funny song":** Gemini said nothing in the catalog is funny and picked upbeat songs instead, so it stretched the request. RAG, in contrast, stays inside the retrieved guides.
- **Day planner, five tasks:** "running the park" and "yoga for 3 hours" matched guides that fit, "eating lunch" and "painting with my friend" matched weakly, and "doing math homework" found nothing.

The original recommender was checked with five sample profiles (Road Trip, High-Energy Pop, Chill Lofi, Deep Intense Rock and Nostalgic Dreamer). Each got a different kind of music, with top scores like Library Rain at 0.99 for Chill Lofi and Storm Runner at 0.99 for Deep Intense Rock. Doubling energy's weight and halving genre's on the four standard profiles changed the middle of two lists but didn't improve them, so I kept the original weights.

---

## 8. Misuse and Prevention

Someone could misuse TaskTunes in a few ways. They could run up my Gemini quota with many or very long requests, so queries are limited to 200 characters and each run or session is limited to 5 Gemini calls. They could type instructions into a query to try to change what Gemini does, since the query becomes part of its prompt. The instruction to use only the retrieved guides, and in LLM-only mode the filter that drops any song not in the catalog, limit the harm, but they do not stop it completely. Gemini could also state something the guides don't say, which is not checked automatically yet. Finally, queries are saved in the log file, so anyone who can read it can see what was typed, and the API key is kept in a .env file that is not committed and is never written to the log. The system makes no decisions about people and recommends only songs from a fixed list, so the harm from misuse is small, and it would need stronger checks before real listeners used it.

---

## 9. Personal Reflection

### What surprised me in testing

What surprised me most was that the Gemini part stopped working even though my code was fine. The default model returned a 504 timeout, so nothing printed, and I assumed I had made a mistake. When I switched to a different model, the same code worked. It showed me that the AI part can fail for reasons outside my code, so the app needs to keep working when Gemini doesn't answer.

### How I worked with AI
I primarily used AI to help me implement the RAG for my original music recommender project and include the specific features I wanted.

One of the suggestion I accepted was that Claude pointed out that Gemini could name songs that are not in my catalog, so I added a check that drops any title it returns that isn't in the catalog. I accepted it because it was a small change that fixed a real problem, since the app can only recommend songs it actually has, and I could test it without calling Gemini.

A suggestion that I did not accept was when I was adding retrieval, Claude's plan was to put it in several new files. The plan wasn't wrong, but it added more structure than a project this size needed and made it harder to see how the pieces fit together. We changed it to add the new code to the files that already existed. It made me realize sometimes AI doesn't give the most clean or readable output.

### What I learned

Through this project, I learned that implementing RAG meant that the answer is only as good as what gets retrieved. Gemini only explains the picks, so when the wrong guide was found, everything after it was wrong too. Small details in retrieval mattered more than the model, such as filler words like "want" and "like" pulled in unrelated guides, and without a stemmer "run in the park" matched nothing because the search didn't treat "run" and "running" as the same word.

Also, matching a song to a task was the hardest part. A task like "study for my exam" worked because the knowledge base had a guide for it, but everyday tasks without one found nothing. I fixed the ones I found by adding guides for everyday situations (outdoors, chores, cooking, commuting), and I learned that improving the database means writing guides for the tasks people actually type, covering the different ways they say them, and testing real queries against it. It would also help to add more songs per genre, since the catalog limits what any task can be matched to.