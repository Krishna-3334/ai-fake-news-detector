"""Fake News Detector - local web app.

Run:   .venv/bin/python app.py
Open:  http://localhost:8000
Needs final_pipeline.joblib (saved by experiment6.ipynb) and index.html (the UI).

Two separate analyses:
  /predict - the writing-style model (TF-IDF + Linear SVM): judges style only, cannot check facts.
  /explain - evidence-based claim verification (factcheck.py): a Groq-hosted LLM with built-in browser
             search checks each claim; sources and excerpts are validated in code and the three-way verdict
             (likely_real / likely_fake / unverified) is computed by fixed rules.
Put your Groq API key in the .env file next to this script:  GROQ_API_KEY=gsk_...
The file is re-read on every request, so no restart is needed after editing it.
Without a key the style model still works and the page says web verification is unavailable.
"""
from http.server import ThreadingHTTPServer

from detector import Handler, PORT, analyse, clean, explain   # noqa: F401  (re-exported for scripts)

if __name__ == "__main__":
    print(f"Fake News Detector running at http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
