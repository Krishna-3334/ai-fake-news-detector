"""Fake News Detector - local web app.

Run:   .venv/bin/python app.py
Open:  http://localhost:8000
Needs final_pipeline.joblib (saved by experiment6.ipynb) and index.html (the UI).

The "Why this verdict?" explanation uses a Groq-hosted LLM with built-in browser search.
Put your Groq API key in the .env file next to this script:
    GROQ_API_KEY=gsk_...
The file is re-read on every request, so no restart is needed after editing it.
Without a key the detector still works; the explanation panel says the key is missing.
"""
import datetime, hashlib, json, math, re, joblib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import groq
except ImportError:                                # detector still works without the SDK
    groq = None

HERE = Path(__file__).parent
PORT = 8000
DEFAULT_MODEL = "openai/gpt-oss-120b"              # Groq production model with browser search

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


# ---------- "Why this verdict?" - fact-check with a Groq LLM + browser search ----------

def load_env():
    """Read KEY=value lines from .env (comments and blank lines ignored)."""
    env, path = {}, HERE / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


SYSTEM = """You are a careful, neutral fact-checker working alongside a fake-news classifier.

The classifier is a TF-IDF + Linear SVM trained on the ISOT dataset (2016-17 US political news). \
It only judges writing style: "real" in its training data meant Reuters wire copy, "fake" meant \
articles from unreliable sites. It cannot check facts. Your job is to check the facts.

Steps:
1. Identify the 2-4 most important, checkable factual claims in the article (events, quotes, \
dates, numbers, who did what). For opinion pieces, check the factual claims the opinion rests on.
2. Use browser search to verify each claim against reputable sources. Prefer primary sources, \
major news organisations and fact-checkers.
3. Decide how likely the article is to be real or fake, based mainly on the evidence.

Rules:
- Only cite URLs that you actually opened or saw in search results. Never invent a source, quote or URL.
- If you could not find evidence either way, say so plainly and use the verdict "unverified". \
Absence of coverage is weak evidence, not proof.
- You may disagree with the classifier. Say so when you do.
- Be concise and plain-spoken; the reader is a student, not a specialist.

When you are done researching, reply with ONLY a JSON object (no prose before or after, no code \
fence) in exactly this shape:
{
  "verdict": "likely_real" | "likely_fake" | "mixed" | "unverified",
  "summary": "1-2 sentences, e.g. 'This article is most probably fake because ...' or 'This article is most probably real because ...'",
  "reasons": ["2-5 short reasons, each one sentence"],
  "evidence": [
    {"claim": "the claim from the article, short",
     "finding": "supported" | "contradicted" | "no_evidence",
     "detail": "one sentence on what the sources say",
     "source_title": "title of the page",
     "source_url": "https://... (a page you found, or null)"}
  ],
  "agrees_with_model": true | false,
  "style_note": "one sentence on what the classifier's top words suggest about the writing style"
}"""

_cache = {}


def _prompt(title, text, pred):
    words = lambda key: ", ".join(w["word"] for w in pred[key]) or "none"
    return (
        f"Today's date: {datetime.date.today().isoformat()}\n\n"
        f"Classifier verdict: {pred['label'].upper()} (SVM margin {pred['score']:+.3f}; "
        f"> 0 means fake, magnitude above 1 is a strong signal)\n"
        f"Words pushing towards fake: {words('fake_words')}\n"
        f"Words pushing towards real: {words('real_words')}\n\n"
        f"<article>\n<headline>{title or '(none given)'}</headline>\n<body>\n{text}\n</body>\n</article>\n\n"
        "Fact-check this article and explain the verdict."
    )


def _parse(raw):
    """Return the last JSON object in the reply that looks like our answer."""
    dec, best = json.JSONDecoder(), None
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(raw, i)
            except ValueError:
                continue
            if isinstance(obj, dict) and "verdict" in obj:
                best = obj
    if best is None:
        raise ValueError("no JSON answer in reply")
    return best


def _sources(message):
    """Every URL the browser search actually returned, mapped to its page title."""
    found = {}
    for tool in message.executed_tools or []:
        for r in (tool.search_results.results or []) if tool.search_results else []:
            if r.url:
                found[r.url] = r.title or ""
        for r in tool.browser_results or []:
            found[r.url] = r.title or ""
        for url in re.findall(r"https?://[^\s\"'<>)\]】]+", tool.output or ""):
            found.setdefault(url.rstrip(".,;"), "")
    return found


def explain(title, text):
    pred = analyse(title, text)
    if "error" in pred:
        return pred
    key = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    if groq is None:
        return {"error": "The groq package is not installed. Run: .venv/bin/python -m pip install groq",
                "code": "setup"}
    env = load_env()
    api_key = env.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("paste"):
        return {"error": "No Groq API key found in the .env file.", "code": "no_key"}

    client = groq.Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=env.get("GROQ_MODEL") or DEFAULT_MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": _prompt(title, text, pred)}],
            tools=[{"type": "browser_search"}],
            tool_choice="required",
            reasoning_effort="low",                # Groq's advice for browser search: keeps sessions short
            max_completion_tokens=8192,
        )
    except groq.AuthenticationError:
        return {"error": "Groq rejected the API key. Check GROQ_API_KEY in the .env file.", "code": "no_key"}
    except groq.RateLimitError:
        return {"error": "Groq rate limit reached. Wait a minute and try again.", "code": "retry"}
    except groq.APIConnectionError:
        return {"error": "Could not reach Groq. Check your internet connection.", "code": "retry"}
    except groq.APIStatusError as e:
        return {"error": f"Groq API error ({e.status_code}): {e.message}", "code": "retry"}

    message = resp.choices[0].message
    found = _sources(message)
    try:
        data = _parse(message.content or "")
    except ValueError:
        return {"error": "The fact-check reply could not be read. Try again.", "code": "retry"}

    # keep only sources the search really returned, so no link is made up
    evidence = []
    for ev in data.get("evidence", [])[:6]:
        url = ev.get("source_url")
        ok = isinstance(url, str) and url in found and urlparse(url).scheme in ("http", "https")
        evidence.append({
            "claim": str(ev.get("claim", "")), "detail": str(ev.get("detail", "")),
            "finding": ev.get("finding") if ev.get("finding") in ("supported", "contradicted") else "no_evidence",
            "source_url": url if ok else None,
            "source_title": str(ev.get("source_title") or found.get(url) or "") if ok else "",
            "domain": urlparse(url).netloc.removeprefix("www.") if ok else "",
        })
    verdict = data.get("verdict")
    result = {
        "verdict": verdict if verdict in ("likely_real", "likely_fake", "mixed") else "unverified",
        "summary": str(data.get("summary", "")),
        "reasons": [str(r) for r in data.get("reasons", [])][:5],
        "evidence": evidence,
        "agrees_with_model": bool(data.get("agrees_with_model")),
        "style_note": str(data.get("style_note", "")),
        "model": resp.model,
        "searched": len(found),
    }
    _cache[key] = result
    return result


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
        routes = {"/predict": analyse, "/explain": explain}
        if self.path not in routes:
            return self._send(404, "Not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            result = routes[self.path](str(body.get("title", "")), str(body.get("text", "")))
        except (ValueError, TypeError):
            result = {"error": "Bad request."}
        self._send(200, json.dumps(result), "application/json")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Fake News Detector running at http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
