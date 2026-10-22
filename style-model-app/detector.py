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
import json, math, re, joblib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import factcheck

HERE = Path(__file__).parent
PORT = 8000
MAX_BODY = 60_000                                   # bytes accepted per request

pipe = joblib.load(HERE / "final_pipeline.joblib")
tfidf, svm = pipe.steps[0][1], pipe.steps[1][1]
vocab = tfidf.get_feature_names_out()
coef = svm.coef_[0]


def clean(t):                                      # same as Experiment 3
    t = t.lower()
    t = re.sub(r"reuters|http\S+|<.*?>", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(t.split())


def analyse(title, text):
    words = clean(f"{title} {text}")
    if len(words.split()) < 5:
        return {"error": "Please paste a bit more text - at least a few sentences works best."}
    X = tfidf.transform([words])
    score = float(svm.decision_function(X)[0])     # > 0 = fake, < 0 = real
    contrib = X.multiply(coef).tocsr()             # per-word push towards fake / real
    pairs = sorted(zip(contrib.indices, contrib.data), key=lambda p: p[1])
    fmt = lambda ps: [{"word": vocab[i], "weight": round(float(w), 4)} for i, w in ps]
    return {
        "label": "fake" if score > 0 else "real",
        "score": round(score, 3),
        "lean": round(1 / (1 + math.exp(-2 * score)), 3),   # 0 = real ... 1 = fake
        "words": len(words.split()),
        "fake_words": fmt([p for p in reversed(pairs) if p[1] > 0][:8]),
        "real_words": fmt([p for p in pairs if p[1] < 0][:8]),
    }


# ---------- evidence-based fact-check (see factcheck.py) ----------

def explain(title, text):
    pred = analyse(title, text)
    if "error" in pred:
        return pred
    return factcheck.check(title, text, pred)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_text(), "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        routes = {"/predict": analyse, "/explain": explain}
        if self.path not in routes:
            return self._send(404, "Not found", "text/plain")
        if int(self.headers.get("Content-Length", 0)) > MAX_BODY:       # keeps one request from burning API quota
            return self._send(200, json.dumps({"error": "That article is too long. Paste a shorter excerpt (under about 50,000 characters)."}),
                              "application/json")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            result = routes[self.path](str(body.get("title", "")), str(body.get("text", "")))
        except (ValueError, TypeError):
            result = {"error": "Bad request."}
        self._send(200, json.dumps(result), "application/json")

    def log_message(self, fmt, *args):
        pass
