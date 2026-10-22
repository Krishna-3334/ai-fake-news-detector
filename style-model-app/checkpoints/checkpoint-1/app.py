"""Fake News Detector - local web app.

Run:   .venv/bin/python app.py
Open:  http://localhost:8000
Needs final_pipeline.joblib (saved by experiment6.ipynb) and index.html (the UI).
"""
import json, math, re, joblib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8000

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


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_text(), "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, "Not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            result = analyse(str(body.get("title", "")), str(body.get("text", "")))
        except (ValueError, TypeError):
            result = {"error": "Bad request."}
        self._send(200, json.dumps(result), "application/json")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Fake News Detector running at http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
