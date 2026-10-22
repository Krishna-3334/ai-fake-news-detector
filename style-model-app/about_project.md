# About this project — Fake News Detection

## Checkpoints

| Checkpoint | Date | State | Folder |
|---|---|---|---|
| **Checkpoint 4 (latest)** | 2026-10-16 | Evidence-based claim verification + NVIDIA NIM (GLM 5.3 Flash) integration + fast concurrent live web search (`ddgs`) + Groq fallback + updated 3D UI | `checkpoints/checkpoint-4/` |
| Checkpoint 3 | 2026-10-14 | Checkpoint 2 + final verdict combines style model and web fact-check, bar follows final verdict, article card flattened (textarea overflow fix), shallower result-card depth | `checkpoints/checkpoint-3/` |
| Checkpoint 2 | 2026-10-14 | 3D UI + SVM detector + "Why this verdict?" Groq fact-check panel | `checkpoints/checkpoint-2/` |
| Checkpoint 1 | 2026-10-14 | 3D UI + SVM detector, before the fact-check panel | `checkpoints/checkpoint-1/` |

Each folder holds exact copies of the app files at that point: `app.py` (local web server and
endpoints), `index.html` (the 3D web UI), and (starting in Checkpoint 4) `factcheck.py`. The notebooks,
datasets and saved models are never changed by the app work, so they don't need restoring.

**To revert to the latest checkpoint (4):**

```
cp checkpoints/checkpoint-4/app.py checkpoints/checkpoint-4/index.html checkpoints/checkpoint-4/factcheck.py .
```

**To revert to checkpoint 3:**

```
cp checkpoints/checkpoint-3/app.py checkpoints/checkpoint-3/index.html .
rm -f factcheck.py
```

**To revert to checkpoint 2:**

```
cp checkpoints/checkpoint-2/app.py checkpoints/checkpoint-2/index.html .
```

**To revert further, to checkpoint 1** (removes the fact-check panel):

```
cp checkpoints/checkpoint-1/app.py checkpoints/checkpoint-1/index.html .
```

Then restart the server (`.venv/bin/python app.py`).

Not stored in checkpoints: `.env` (holds your private Groq API key — it stays in place and is never
copied). Checkpoint 2 also relies on the `groq` package (1.7.0) installed in `.venv`.

---

## What the project is

A fake news classifier trained on the ISOT Fake & True news dataset (`Fake.csv`, `True.csv`), built
across four lab notebooks, plus a local web app for trying it on any article.

### Notebooks (run in order: 3 → 4 → 5 → 6)

| Notebook | What it does | Outputs |
|---|---|---|
| `experiment3.ipynb` — Pre-processing | Loads both CSVs, labels fake = 1 / real = 0, removes 631 empty and 5,793 duplicate articles (38,658 left), cleans text (lowercase, strips "reuters", URLs, HTML, non-letters), splits 70/15/15 stratified, fits TF-IDF (50,000 features, English stop words) on train only | `train.csv`, `val.csv`, `test.csv`, `tfidf.joblib` |
| `experiment4.ipynb` — EDA | Article length by class, top words and bigrams per class, most class-distinctive words | `fig1_length.png`, `fig2_distinctive.png` |
| `experiment5.ipynb` — Training | Majority baseline (54.8%), compares Logistic Regression / Linear SVM / Random Forest, 5-fold CV, grid search; Linear SVM (C = 5) wins. Leak check retrains without boilerplate words ("featured", "image", "getty", "via", …) | `model_best.joblib`, `model_logreg.joblib`, `model_best_noleak.joblib`, `tfidf_noleak.joblib`, `fig5_1_models.png`, `fig5_2_coefficients.png` |
| `experiment6.ipynb` — Evaluation | Test-set metrics, confusion matrix, ROC curve, threshold sensitivity, error analysis; saves TF-IDF + SVM as one pipeline | `final_pipeline.joblib`, `fig6_1_confusion.png`, `fig6_2_roc.png` |

All four notebooks were verified to run end to end with the project's `.venv` (Python 3.14,
scikit-learn 1.9.1).

### Final model results (held-out test set, 5,798 articles)

| Metric | Linear SVM (selected) |
|---|---|
| Accuracy | 98.40% |
| Precision (fake) | 0.9880 |
| Recall (fake) | 0.9763 |
| F1 (fake) | 0.9821 |
| ROC-AUC | 0.9981 |

### Web app (`app.py` + `index.html`)

Run `.venv/bin/python app.py` and open http://localhost:8000.

- `app.py` — standard-library HTTP server (no extra packages). Loads `final_pipeline.joblib`, applies
  the same text cleaning as Experiment 3, and serves:
  - `GET /` → the UI (`index.html`, re-read on every request so UI edits show on refresh)
  - `POST /predict` → JSON with `label` (real/fake), `score` (SVM margin; > 0 = fake),
    `lean` (0–1 position for the bar), `words`, and the top 8 `fake_words` / `real_words`
    (each word's TF-IDF value × SVM coefficient — these sum, with the intercept, to the score)
- `index.html` — the UI:
  - Three.js WebGL background (wireframe globe that tints blue/orange with the verdict, floating
    lit shapes, starfield, mouse/scroll parallax) over a CSS perspective grid floor and glow orbs
  - Glass "slab" cards angled like an open book, tilting with the mouse, with layered depth
  - Left card: headline + article input, word count, ⌘/Ctrl+Enter, Real/Fake sample buttons
  - Right card: flips to reveal the verdict — 3D coin badge, "Likely real/fake" with signal strength
    and SVM margin, a 3D Real ↔ Fake certainty bar, word-contribution bars, and a caveat note
  - Header stats (98.4% accuracy, 0.998 ROC-AUC, 38,658 articles); dark/light mode;
    honours reduced-motion; course code and student footer removed on request

### Known limitation (important for the report)

The model learned **writing style**, not truth: "real" in the dataset means 2016–17 Reuters wire
copy, "fake" means articles from unreliable US sites. Words like "said", "Tuesday", "Washington" push
towards real; "breaking", "video", "featured image" push towards fake. It cannot fact-check, and it
knows nothing after 2017. Example: an India Today opinion column on Navi Pillay's Nobel Peace Prize
scored −0.15 ("Likely real", weak) — replacing "said" with "stated" flipped it to fake (+0.10).

---

## Added in Checkpoint 2 — "Why this verdict?" AI fact-check (2026-10-14)

Added under the certainty bar in the result card. After the SVM verdict appears, the page calls a new
`POST /explain` endpoint, which sends the article (plus the SVM verdict and top words) to a Groq-hosted
LLM, **`openai/gpt-oss-120b`**, with Groq's built-in browser search. It checks the article's main claims
and returns a plain explanation ("This article is most probably fake/real because …"), reasons, and
evidence cards (Supported / Contradicted / No evidence found) with links to the sources it found, and
says whether it agrees with the SVM.

- The Groq key goes in the `.env` file in the project folder: `GROQ_API_KEY=gsk_...`. The file is
  re-read on every request, so no restart is needed. Optional `GROQ_MODEL=` overrides the model.
  Without a key the panel shows how to add one; the detector itself still works.
- Model choice: Groq shut down `groq/compound` and `groq/compound-mini` (its web-search systems) on
  2026-09-21 and moved Llama 3.x to enterprise-only, so `openai/gpt-oss-120b` (production, browser
  search supported) is used.
- Added the `groq` package to `.venv` (an earlier Claude-based version of this trial was replaced;
  the `anthropic` package was removed again).
- Only links that actually came back from the browser search are shown (anything else is dropped).
- Results are cached in memory per article while the server runs.

Tested live with a Groq key: for a made-up "GTA 6 postponed to 2030" claim, the SVM said "Likely real"
(margin −0.095, weak — only 6 of its 19 words were known to the model, and numbers are stripped), while
the fact-check panel correctly said "most probably fake" and disagreed with the model, citing Rockstar's
announced release date.

To remove this feature, restore Checkpoint 1 (see the top of this file).

---

## Checkpoint 4 — Evidence-based verification & NVIDIA GLM 5.3 NIM integration (2026-10-16)

Saved in `checkpoints/checkpoint-4/` (`app.py`, `index.html`, `factcheck.py`). To undo, restore Checkpoint 3.

### Why a credible fake RBI article came out "Likely Real" (diagnosis)

1. **The style model can't catch well-written fakes.** It scored a fictional RBI wallet-circular story
   −1.13 (strongly "real") because of wire-agency wording: "said", "Tuesday", "Rs", "banks", "India".
   The audit (below) shows the dataset's labels are readable from formatting alone.
2. **The headline fell back to the style model.** In Checkpoint 3, whenever the web check was inconclusive
   or failed (rate limit, timeout, parse error), the big verdict showed the style model's "Likely real".
3. **The fact-check's overall verdict wasn't tied to its claim results.** The LLM chose the verdict itself:
   in a live run every claim was "no evidence" yet it answered "fake", and the same freedom lets a topic
   that resembles a real RBI rule come out "real".

### What changed

- `factcheck.py` (new): claim-level verification on Groq (`openai/gpt-oss-120b` + browser search,
  reasoning effort `medium`, overridable with `GROQ_REASONING_EFFORT` in `.env`). The prompt asks for
  2–6 specific claims (names, dates, figures, institutions, circular numbers, policy status, quotes), primary
  sources first (rbi.org.in, sebi.gov.in, pib.gov.in, egazette.gov.in, … for Indian regulatory news), and an
  exact-claim comparison. Then, in code:
  - a source is kept only if the search actually retrieved that URL;
  - an excerpt or publication date is shown only if it is found in the retrieved page text;
  - a claim marked supported / contradicted / conflicting with no kept source is downgraded to unresolved;
  - the category is computed by fixed rules: any contradicted central claim → **Likely fake or misleading**;
    any contradicted detail → fake or misleading ("misleading: …"); satire, old news presented as new or a
    misleading headline (with evidence) → fake or misleading; every central claim supported → **Likely real**;
    otherwise **Unverified** ("Partially supported" when some central claims are confirmed). Absence of
    evidence alone never makes an article fake;
  - evidence strength (none / weak / moderate / strong) from the kept sources: primary-source tier, excerpts
    matched word-for-word, number of independent domains. It is separate from the style model.
- `app.py`: `/explain` now calls `factcheck.check`; the Groq code moved out of `app.py`.
- `index.html`: the result card shows the evidence-based three-way verdict (with "Checking claims…" while it
  runs, never the style model's guess), an evidence-strength meter, a claim-by-claim panel (status, central
  vs. detail, entities, why, official-source check, sources with tier, publisher, verified date and verified
  excerpt), a red "Web verification unavailable — not independently verified" warning and an **Unverified**
  verdict when the web check can't run, and a separate "Writing-style model (SVM)" section whose bar is
  labelled as a style score, not a probability. Stat tiles now say they describe the style model.
- `audit/ml_audit.py` → `audit/audit_report.md`, `audit/audit_metrics.json`, `audit/fig_audit_calibration_pr.png`.
- `tests/test_factcheck.py`: 21 offline unit tests for validation and verdict rules
  (`.venv/bin/python -m unittest discover -s tests`).
- `tests/challenge_set.json` + `tests/run_challenge.py` → `tests/challenge_results.md` / `.json`:
  13 challenge cases run live through both stages.
- **NVIDIA NIM & GLM 5.3 Flash Integration**: Added support for NVIDIA NIM API (`https://integrate.api.nvidia.com/v1`) using the OpenAI client SDK. Default model set to `z-ai/glm-5.3-flash` with low reasoning effort for ~18s response latency.
- **Fast Concurrent Web Search**: Replaced sequential web queries with multi-threaded parallel search (`ddgs` via `ThreadPoolExecutor`), reducing search time from 13s to ~1.2s.
- **Fallback Compatibility**: Kept seamless fallback to Groq (`openai/gpt-oss-120b`) if only Groq key is present.
