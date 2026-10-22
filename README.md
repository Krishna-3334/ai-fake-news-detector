# AI Fake News Detector

Two fake-news detection projects live in this repository. They are independent and share no code.

| Folder | What it is | Stack |
|---|---|---|
| [`style-model-app/`](style-model-app/) | Writing-style classifier plus an evidence-based web fact-check, with a web UI. **Live on Vercel.** | Python, scikit-learn, Groq |
| `frontend/` + `backend/` | Web app that runs pretrained transformer models on an article or URL. | React + Vite, FastAPI |

**Live demo (style model + fact-check):** https://fake-news-detector-rho-ten.vercel.app

---

## 1. Style model and fact-check (`style-model-app/`)

Paste an article and get two separate analyses:

1. **Writing style.** A TF-IDF + Linear SVM model trained on the ISOT Fake/True news dataset. It shows which words pushed the score toward real or fake. It judges style only and cannot check facts.
2. **Evidence-based fact-check.** A language model searches the web for each key claim. Sources and excerpts are validated in code, and fixed rules produce the verdict: likely real, likely fake or unverified.

Setup, usage, deployment and known limits are in [`style-model-app/README.md`](style-model-app/README.md).

## 2. Transformer app (`frontend/` and `backend/`)

A React + Vite frontend and a FastAPI backend that runs pretrained Hugging Face models for fake-news and AI-text detection.

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The GitHub Actions workflow builds `frontend/` and publishes it to GitHub Pages on every push to `main`: https://Krishna-3334.github.io/ai-fake-news-detector/

## Security

API keys live in a local `.env` file that is git-ignored. Never commit keys. `style-model-app/.env.example` shows which variables the app reads.
