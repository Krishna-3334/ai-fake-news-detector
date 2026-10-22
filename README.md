# AI Fake News Detector

Simple repo with a React + Vite frontend and a FastAPI backend for detecting AI-generated or fake news.

Structure
- `frontend/` — Vite + React app. Build output goes to `frontend/dist`.
- `backend/` — FastAPI app and Python dependencies listed in `backend/requirements.txt`.

Run locally

Frontend

```bash
cd frontend
npm install
npm run dev
```

Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Deployment

This repository includes a GitHub Actions workflow that builds the frontend and deploys the `frontend/dist` folder to GitHub Pages (branch `gh-pages`) on pushes to `main`.

Public site (after deploy)

https://Krishna-3334.github.io/ai-fake-news-detector/

If you prefer Netlify or Vercel, link the `frontend/` folder in their UI — they will handle builds and previews automatically.

License

Add a license file if you want to make this explicit (MIT is a common choice).

## Local app: style model + evidence-based fact-check

`style-model-app/` holds a separate, self-contained local app: a TF-IDF + Linear SVM writing-style
classifier (trained on the ISOT Fake/True dataset) combined with an evidence-based web fact-check
(Groq or NVIDIA NIM). See `style-model-app/about_project.md` for the notebooks, metrics and checkpoints.

To run it, copy `style-model-app/.env.example` to `.env`, add your key, then run `python app.py`
inside that folder. The `.env` file is git-ignored; never commit API keys.

Live demo of the local app (Vercel): https://fake-news-detector-rho-ten.vercel.app
