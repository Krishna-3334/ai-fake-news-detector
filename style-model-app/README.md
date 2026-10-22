# Style Model + Evidence-Based Fact-Check

A fake-news detector with two deliberately separate parts, so a reader never mistakes one for the other.

**Live demo:** https://fake-news-detector-rho-ten.vercel.app

## How it works

| Part | What it does | What it cannot do |
|---|---|---|
| **Writing-style model** | TF-IDF features + Linear SVM score how much the writing resembles fake or real articles in the training data. The page lists the words that pushed the score each way. | It cannot check whether the claims are true. |
| **Web fact-check** | A Groq-hosted model (`openai/gpt-oss-120b`) with built-in browser search checks each key claim. Code then drops any source whose URL was not actually retrieved and any excerpt that is not in a retrieved page. Fixed rules turn the results into **likely real**, **likely fake** or **unverified**. | It depends on what the web search can find. Thin evidence gives "unverified". |

## Endpoints

| Route | Purpose |
|---|---|
| `POST /predict` | Style score and top words. Body: `{"title": "...", "text": "..."}` |
| `POST /explain` | Web fact-check for the same body. Takes roughly 10 to 40 seconds. |

Requests over about 60 KB are rejected to protect API quota.

## Run locally

Needs Python 3.12 or newer.

```bash
cd style-model-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Groq key in .env
python app.py               # http://localhost:8000
```

Without a key the style model still works and the page says web verification is unavailable.

| Variable | Meaning |
|---|---|
| `GROQ_API_KEY` | Groq key. Preferred when set. Free keys at console.groq.com. |
| `GROQ_MODEL` | Optional. Defaults to `openai/gpt-oss-120b`. |
| `NVIDIA_API_KEY`, `NVIDIA_MODEL` | Optional fallback. Used only when no Groq key is set, and runs locally only. |

The `.env` file is re-read on every request, so key changes need no restart.

## Deploy to Vercel

```bash
cd style-model-app
vercel link
printf '%s' "$GROQ_API_KEY" | vercel env add GROQ_API_KEY production --sensitive
vercel deploy --prod
```

`pyproject.toml` declares the dependencies and the entry point (`detector:Handler`). `vercel.json` sets a 60-second limit and bundles the model. The Vercel project is not linked to GitHub, so pushes do not redeploy it.

## Files

| Path | Role |
|---|---|
| `detector.py` | Model loading, scoring, request handler. This is the Vercel entry point. |
| `app.py` | Local launcher. |
| `factcheck.py` | Fact-check pipeline, evidence validation and verdict rules. |
| `index.html` | Web UI. |
| `final_pipeline.joblib` | The deployed style model. |
| `final_pipeline_original.joblib` | Previous model, kept as a backup. |
| `experiment3.ipynb` to `experiment6.ipynb` | Lab notebooks: preprocessing, EDA, training, evaluation. |
| `tests/`, `audit/` | Fact-check tests and a model audit. |
| `checkpoints/` | Earlier snapshots of the app. |
| `about_project.md` | Original project notes and checkpoint history. |

## Model and results

- **Data.** ISOT Fake/True news dataset. `Fake.csv` and `True.csv` are not in the repo because they are large. Run `experiment3.ipynb` on the source files to recreate the splits.
- **Notebook model.** Linear SVM, 98.40% test accuracy, ROC-AUC 0.998.
- **Deployed model.** The notebook model learned publisher habits such as weekday names, dateline cities, "said" and image-credit boilerplate. I retrained it without those words and with log-scaled counts. Test accuracy is 98.02% and ROC-AUC 0.998. On 10 hand-written articles it scored 9 right against 8 for the notebook model. That sample is small and is only a sanity check.
- **Reproducibility.** The notebooks produce the original model. The deployed retrain was run outside the notebooks, so the notebooks do not reproduce it.

## Known limits

- The style model is trained on 2016 and 2017 US political news. Topic words like "obama" and "republican" still carry weight, and it can misjudge other subjects.
- A calm, official-sounding hoax can still score as real on style. This is why the fact-check exists.
- Style scores are not probabilities. The "lean" bar is a rescaled score and is not calibrated.
- The live site runs on one shared API key with no per-user limit.
