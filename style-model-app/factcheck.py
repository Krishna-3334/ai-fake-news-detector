"""Evidence-based claim verification for the Fake News Detector.

Supports:
  1. NVIDIA NIM (GLM 5.3 Flash) via OpenAI-compatible API with fast concurrent web search.
  2. Groq (openai/gpt-oss-120b) with built-in browser search as fallback.

Pipeline:
  * Extract claims & retrieve live web evidence for the article.
  * Model evaluates claims against retrieved sources.
  * Validation ensures:
      - a source is kept only if its URL was actually retrieved;
      - an excerpt is kept only if its text appears in retrieved page content;
      - unsupported claims are downgraded to unresolved;
      - verdict is computed by deterministic rules (see `aggregate`).
"""
import concurrent.futures, datetime, hashlib, json, os, re
from pathlib import Path
from urllib.parse import urlparse

try:
    import openai
except ImportError:
    openai = None

try:
    import groq
except ImportError:
    groq = None

HERE = Path(__file__).parent
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
CATEGORIES = ("likely_real", "likely_fake", "unverified")
STATUSES = ("supported", "contradicted", "conflicting", "unresolved")

# Primary / official sources: governments, regulators, courts, intergovernmental bodies, original research.
PRIMARY = ("gov.in", "nic.in", "rbi.org.in", "sebi.gov.in", "npci.org.in", "isro.gov.in", "pib.gov.in",
           "sci.gov.in", "ecourts.gov.in", "egazette.gov.in", "indiacode.nic.in", "eci.gov.in",
           ".gov", ".gov.uk", ".gc.ca", ".gov.au", "europa.eu", "who.int", "un.org", "worldbank.org", "imf.org",
           "nasa.gov", "esa.int", "doi.org", "nature.com", "science.org", "thelancet.com", "nejm.org",
           "jamanetwork.com", "bmj.com", "arxiv.org", "ncbi.nlm.nih.gov")
NEWS = ("reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com", "indianexpress.com", "livemint.com",
        "hindustantimes.com", "ndtv.com", "economictimes.indiatimes.com", "timesofindia.indiatimes.com",
        "business-standard.com", "moneycontrol.com", "theprint.in", "scroll.in", "thewire.in", "nytimes.com",
        "washingtonpost.com", "theguardian.com", "ft.com", "bloomberg.com", "cnn.com", "aljazeera.com",
        "npr.org", "pti.in", "ptinews.com", "altnews.in", "boomlive.in", "factly.in", "politifact.com",
        "factcheck.org", "snopes.com", "fullfact.org")

AD_DOMAINS = ("bing.com/aclick", "doubleclick.net", "googleadservices.com", "adservice.google.com")

SYSTEM_NVIDIA = """You are a careful, neutral fact-checker. You verify the specific factual claims in a news article against evidence from web and news search results provided to you. A separate writing-style classifier has already scored the article; it cannot check facts, so ignore its opinion when judging the evidence.

Method:
1. Extract the article's 1-4 most important checkable factual claims. Keep exact specifics: names, dates, deadlines, figures, institutions, document or circular numbers, policy status (proposed vs. final) and quoted statements. Mark the claims the article's main point depends on as "central". Include the headline's main claim.
2. Compare the EXACT claim with the provided web search evidence. A source about a similar or related topic does not support the claim.
3. Status for each claim:
   - "supported": a retrieved source confirms the specific claim, including its key details.
   - "contradicted": a retrieved source states something incompatible with the claim.
   - "conflicting": credible retrieved sources disagree with each other on the claim.
   - "unresolved": not enough evidence found either way.

Rules:
- Every source you cite MUST be one of the URLs provided in the web search evidence. Never invent a URL, title, date, quote or result.
- "excerpt" must be copied word for word (10-40 words) from the provided evidence text. "published" must be copied exactly as shown or null.
- Be concise. Output ONLY the JSON object.

When done, reply with ONLY one JSON object (no markdown code fence, no surrounding prose):
{
  "article_type": "news" | "opinion" | "satire" | "other",
  "summary": "2 sentences on what the evidence shows",
  "headline_check": {"misleading": true | false, "note": "one sentence"},
  "timeliness": {"old_news_presented_as_new": true | false, "note": "one sentence"},
  "claims": [
    {"claim": "the specific claim, short",
     "type": "event" | "policy" | "quote" | "figure" | "date" | "attribution" | "other",
     "central": true | false,
     "entities": ["key names, figures, institutions"],
     "status": "supported" | "contradicted" | "conflicting" | "unresolved",
     "reason": "why: compare claim with sources",
     "checked_official_source": true | false,
     "sources": [{"url": "https://...", "publisher": "name", "published": "date or null",
                  "excerpt": "verbatim words from the page"}]}
  ]
}"""

SYSTEM = """You are a careful, neutral fact-checker. You verify the specific factual claims in a news article \
against evidence you retrieve with browser search. A separate writing-style classifier has already scored the \
article; it cannot check facts, so ignore its opinion when judging the evidence.

Method:
1. Extract the article's 2-6 most important checkable factual claims. Keep the exact specifics: names, dates, \
deadlines, figures, institutions, document or circular numbers, policy status (proposed vs. final) and quoted \
statements. Mark the claims the article's main point depends on as "central". Include the headline's main claim.
2. Search for each claim. Prefer primary sources: the issuing government body, regulator, court, official \
gazette, press release or original research. For Indian regulatory news check the institution's own \
publications first (e.g. rbi.org.in notifications and press releases for banking claims, sebi.gov.in for \
markets, pib.gov.in and the relevant ministry for government announcements, egazette.gov.in for laws).
3. Compare the EXACT claim with the evidence. A source about a similar or related topic does not support the \
claim. Separate confirmed facts from unsupported additions (e.g. a real policy with a different deadline, \
figure, quote or status).
4. Status for each claim:
   - "supported": a retrieved source confirms the specific claim, including its key details.
   - "contradicted": a retrieved source states something incompatible with the claim (different figure, date, \
status, quote, or the event is shown not to have happened / to be old).
   - "conflicting": credible retrieved sources disagree with each other on the claim.
   - "unresolved": you could not find enough evidence either way. Absence of coverage alone is NOT a \
contradiction - use "unresolved" and say what you checked.
5. Also judge: is the article news, opinion or satire; is old information presented as new; does the headline \
overstate or distort what the body and evidence support.

Rules:
- Every source you list must be a page you actually retrieved. Never invent a URL, title, date, quote or result.
- "excerpt" must be copied word for word (10-40 words) from the retrieved page. "published" must be copied \
exactly as the page shows it, or null.
- Write plainly for a student reader.

When done, reply with ONLY one JSON object (no prose, no code fence):
{
  "article_type": "news" | "opinion" | "satire" | "other",
  "summary": "2 sentences on what the evidence shows",
  "headline_check": {"misleading": true | false, "note": "one sentence"},
  "timeliness": {"old_news_presented_as_new": true | false, "note": "one sentence"},
  "claims": [
    {"claim": "the specific claim, short",
     "type": "event" | "policy" | "quote" | "figure" | "date" | "attribution" | "other",
     "central": true | false,
     "entities": ["names, dates, figures, institutions in the claim"],
     "status": "supported" | "contradicted" | "conflicting" | "unresolved",
     "reason": "why: compare the exact claim with what the sources say; name the detail that matches or differs",
     "checked_official_source": true | false,
     "sources": [{"url": "https://...", "publisher": "name", "published": "as shown on the page or null",
                  "excerpt": "verbatim words from the page"}]}
  ]
}"""

_cache = {}


def load_env():
    """Read KEY=value lines from .env next to this file (re-read on every call, so edits apply immediately)."""
    env, path = {}, HERE / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_REASONING_EFFORT", "NVIDIA_API_KEY", "NVIDIA_MODEL"):
        if os.environ.get(k) and not env.get(k):   # hosted deploys (Vercel) supply keys as environment variables
            env[k] = os.environ[k]
    return env


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def _matches(domain, suffix):
    return domain.endswith(suffix) if suffix.startswith(".") else domain == suffix or domain.endswith("." + suffix)


def domain_of(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def tier(domain):
    if any(_matches(domain, p) for p in PRIMARY):
        return "primary"
    if any(_matches(domain, n) for n in NEWS):
        return "news"
    return "other"


def retrieved_pages(message):
    """URL -> {title, text} for every page the browser search returned, plus all tool output text."""
    pages, corpus = {}, []
    for tool in getattr(message, "executed_tools", None) or []:
        results = (tool.search_results.results or []) if getattr(tool, "search_results", None) else []
        for r in results:
            if r.url:
                pages.setdefault(r.url, {"title": r.title or "", "text": ""})["text"] += " " + (r.content or "")
        for r in getattr(tool, "browser_results", None) or []:
            pages.setdefault(r.url, {"title": r.title or "", "text": ""})["text"] += " " + (r.content or "")
        out = getattr(tool, "output", None) or ""
        corpus.append(out)
        for url in re.findall(r"https?://[^\s\"'<>)\]】]+", out):
            pages.setdefault(url.rstrip(".,;"), {"title": "", "text": ""})
    return pages, " ".join(corpus)


def _search_query(q):
    """Worker function for concurrent news and web search."""
    items = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            # Check news first
            for n in ddgs.news(q, max_results=3):
                url = n.get("url") or n.get("href")
                if url and url.startswith("http") and not any(ad in url.lower() for ad in AD_DOMAINS):
                    items.append((url, n.get("title") or "", n.get("body") or ""))
            # If news had few results, supplement with web text search
            if len(items) < 2:
                for r in ddgs.text(q, max_results=3):
                    url = r.get("href") or r.get("url")
                    if url and url.startswith("http") and not any(ad in url.lower() for ad in AD_DOMAINS):
                        items.append((url, r.get("title") or "", r.get("body") or ""))
    except Exception:
        pass
    return items


def fetch_web_evidence(title, text):
    """Retrieve actual news and web search pages quickly via concurrent queries."""
    pages, corpus = {}, []
    queries = []

    if title and len(title.strip()) > 3:
        clean_title = re.sub(r"[^\w\s]", " ", title)
        words = clean_title.split()[:8]
        if words:
            queries.append(" ".join(words))

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 20]
    for s in sentences[:2]:
        clean_s = re.sub(r"[^\w\s]", " ", s)
        words = clean_s.split()[:8]
        if words:
            queries.append(" ".join(words))

    seen = set()
    uniq = [q for q in queries if q.lower() not in seen and not seen.add(q.lower())]

    # Run searches in parallel to keep latency under ~2 seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for results in executor.map(_search_query, uniq[:2]):
            for url, title_str, body_str in results:
                pages.setdefault(url, {"title": title_str, "text": ""})
                pages[url]["text"] += " " + body_str
                corpus.append(f"{title_str}: {body_str}")

    return pages, " ".join(corpus)


def format_evidence_for_prompt(pages):
    """Format retrieved web results into a concise text block for the LLM."""
    if not pages:
        return "No external web search results were found for this query."
    lines = []
    for i, (url, info) in enumerate(list(pages.items())[:6], 1):
        lines.append(f"Source [{i}]: {url}")
        lines.append(f"Title: {info.get('title', '')}")
        snippet = info.get("text", "").strip()
        lines.append(f"Content: {snippet[:500]}")
        lines.append("")
    return "\n".join(lines)


def parse_answer(raw):
    """Last JSON object in the reply that contains a claims list."""
    dec, best = json.JSONDecoder(), None
    for i, ch in enumerate(raw or ""):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(raw, i)
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("claims"), list):
                best = obj
    if best is None:
        raise ValueError("no JSON answer with claims in reply")
    return best


def validate(data, pages, corpus):
    """Keep only retrieved sources and verifiable excerpts; downgrade claims that lose their evidence."""
    norm_corpus = _norm(corpus)
    official_page_retrieved = any(tier(domain_of(u)) == "primary" for u in pages)
    claims = []
    for c in data.get("claims", [])[:8]:
        status = c.get("status") if c.get("status") in STATUSES else "unresolved"
        sources = []
        for s in (c.get("sources") or [])[:4]:
            url = s.get("url")
            if not (isinstance(url, str) and url in pages and urlparse(url).scheme in ("http", "https")):
                continue
            page = _norm(pages[url]["text"])
            excerpt, ex_ok = str(s.get("excerpt") or "").strip(), False
            ne = _norm(excerpt)
            if len(ne) >= 15:
                ex_ok = ne[:220] in page or ne[:220] in norm_corpus
            published = str(s.get("published") or "").strip()
            pub_ok = bool(published) and _norm(published) in (page + " " + norm_corpus) and len(_norm(published)) >= 4
            domain = domain_of(url)
            sources.append({
                "url": url, "domain": domain, "tier": tier(domain),
                "publisher": str(s.get("publisher") or pages[url]["title"] or domain)[:80],
                "published": published if pub_ok else None,
                "excerpt": excerpt[:400] if ex_ok else None,
                "excerpt_verified": ex_ok,
            })
        note = ""
        if status in ("supported", "contradicted", "conflicting") and not sources:
            note = f"Marked {status} by the checker, but none of its sources were among the pages actually retrieved, so it is treated as unresolved."
            status = "unresolved"
        if status == "conflicting" and len({s["domain"] for s in sources}) < 2:
            note = "Marked conflicting, but fewer than two retrieved sources back it up, so it is treated as unresolved."
            status = "unresolved"
        claims.append({
            "claim": str(c.get("claim", ""))[:300],
            "type": c.get("type") if c.get("type") in ("event", "policy", "quote", "figure", "date", "attribution") else "other",
            "central": bool(c.get("central")),
            "entities": [str(e)[:60] for e in (c.get("entities") or [])][:8],
            "status": status,
            "reason": str(c.get("reason", ""))[:600],
            "checked_official_source": bool(c.get("checked_official_source")) and official_page_retrieved,
            "sources": sources,
            "note": note,
        })
    if claims and not any(c["central"] for c in claims):
        for c in claims:
            c["central"] = True
    return claims


def evidence_strength(claims):
    """How solid the evidence behind the deciding claims is."""
    decided = [c for c in claims if c["central"] and c["status"] in ("supported", "contradicted")]
    srcs = [s for c in decided for s in c["sources"]]
    if not srcs:
        return "none"
    verified = [s for s in srcs if s["excerpt_verified"]]
    primary_verified = [s for s in verified if s["tier"] == "primary"]
    domains = {s["domain"] for s in srcs}
    if primary_verified or (len(verified) >= 2 and len(domains) >= 3):
        return "strong"
    if verified or any(s["tier"] == "primary" for s in srcs) or len(domains) >= 2:
        return "moderate"
    return "weak"


def aggregate(claims, data):
    """Fixed rules from validated claims to one of three categories, with the rule that fired."""
    central = [c for c in claims if c["central"]]
    contra = [c for c in central if c["status"] == "contradicted"]
    supported = [c for c in central if c["status"] == "supported"]
    misleading_headline = bool((data.get("headline_check") or {}).get("misleading"))
    old_as_new = bool((data.get("timeliness") or {}).get("old_news_presented_as_new"))
    satire = data.get("article_type") == "satire"
    any_contra = any(c["status"] == "contradicted" for c in claims)
    if contra:
        return "likely_fake", f"{len(contra)} of {len(central)} central claim{'s' if len(central) != 1 else ''} contradicted by retrieved sources"
    if any_contra:
        n = sum(c["status"] == "contradicted" for c in claims)
        return "likely_fake", f"misleading: {n} supporting detail{'s' if n != 1 else ''} contradicted by retrieved sources, although the central claims are not"
    if satire:
        return "likely_fake", "identified as satire, not factual reporting"
    if old_as_new and supported:
        return "likely_fake", "old information presented as new"
    if misleading_headline and supported:
        return "likely_fake", "the headline misrepresents what the article and the evidence support"
    if central and len(supported) == len(central):
        return "likely_real", f"all {len(central)} central claim{'s' if len(central) != 1 else ''} supported by retrieved sources"
    if supported:
        return "unverified", f"partially supported: {len(supported)} of {len(central)} central claims confirmed, the rest unresolved or conflicting"
    if any(c["status"] == "conflicting" for c in central):
        return "unverified", "credible sources conflict on the central claims"
    return "unverified", "not enough evidence found to confirm or refute the central claims"


def build_prompt(title, text, pred):
    words = lambda key: ", ".join(w["word"] for w in pred.get(key, [])) or "none"
    return (f"Today's date: {datetime.date.today().isoformat()}\n\n"
            f"(Writing-style classifier, for context only: {pred['label'].upper()}, margin {pred['score']:+.3f}; "
            f"top fake-leaning words: {words('fake_words')}; top real-leaning words: {words('real_words')})\n\n"
            f"<article>\n<headline>{title or '(none given)'}</headline>\n<body>\n{text}\n</body>\n</article>\n\n"
            "Verify this article's claims.")


def _check_nvidia(title, text, pred, env, key, api_key):
    """Fact-check using NVIDIA NIM (GLM 5.3 Flash) with fast live web search."""
    if openai is None:
        return {"error": "The openai package is not installed (.venv/bin/pip install openai).", "code": "setup"}
    model = env.get("NVIDIA_MODEL") or env.get("GLM_MODEL") or DEFAULT_NVIDIA_MODEL
    pages, corpus = fetch_web_evidence(title, text)
    evidence_text = format_evidence_for_prompt(pages)

    words = lambda k: ", ".join(w["word"] for w in pred.get(k, [])) or "none"
    user_prompt = (
        f"Today's date: {datetime.date.today().isoformat()}\n\n"
        f"(Writing-style classifier, for context only: {pred['label'].upper()}, margin {pred['score']:+.3f}; "
        f"top fake-leaning words: {words('fake_words')}; top real-leaning words: {words('real_words')})\n\n"
        f"<article>\n<headline>{title or '(none given)'}</headline>\n<body>\n{text}\n</body>\n</article>\n\n"
        f"<retrieved_web_evidence>\n{evidence_text}\n</retrieved_web_evidence>\n\n"
        "Verify this article's claims against the retrieved web evidence. Reply with ONLY the required JSON object."
    )

    client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=90)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_NVIDIA},
                      {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=2048,
            extra_body={"reasoning_effort": "low"}
        )
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        return {"error": "NVIDIA rejected the API key. Check NVIDIA_API_KEY in the .env file.", "code": "no_key"}
    except openai.RateLimitError:
        return {"error": "NVIDIA rate limit reached. Wait a minute and try again.", "code": "retry"}
    except openai.APITimeoutError:
        return {"error": "The web check took too long and was stopped. Try again.", "code": "retry"}
    except openai.APIConnectionError:
        return {"error": "Could not reach NVIDIA API. Check your internet connection.", "code": "retry"}
    except openai.APIStatusError as e:
        return {"error": f"NVIDIA API error ({e.status_code}): {e.message}", "code": "retry"}

    msg = resp.choices[0].message
    raw = msg.content or msg.reasoning_content or ""
    try:
        data = parse_answer(raw)
    except ValueError:
        return {"error": "The fact-check reply could not be read. Try again.", "code": "retry"}
    claims = validate(data, pages, corpus)
    if not claims:
        return {"error": "The fact-check returned no checkable claims. Try again.", "code": "retry"}
    category, rule = aggregate(claims, data)
    result = {
        "category": category,
        "rule": rule,
        "evidence_strength": evidence_strength(claims) if category != "unverified" else
                             ("weak" if any(c["sources"] for c in claims) else "none"),
        "summary": str(data.get("summary", ""))[:600],
        "article_type": data.get("article_type") if data.get("article_type") in ("news", "opinion", "satire") else "other",
        "headline_check": {"misleading": bool((data.get("headline_check") or {}).get("misleading")),
                           "note": str((data.get("headline_check") or {}).get("note", ""))[:300]},
        "timeliness": {"old_news_presented_as_new": bool((data.get("timeliness") or {}).get("old_news_presented_as_new")),
                       "note": str((data.get("timeliness") or {}).get("note", ""))[:300]},
        "claims": claims,
        "pages_retrieved": len(pages),
        "model": f"{resp.model} (NVIDIA NIM)",
    }
    _cache[key] = result
    return result


def _check_groq(title, text, pred, env, key, api_key):
    """Fact-check fallback using Groq with built-in browser search."""
    if groq is None:
        return {"error": "The groq package is not installed (.venv/bin/python -m pip install groq).", "code": "setup"}
    model = env.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    effort = env.get("GROQ_REASONING_EFFORT") or "medium"
    client = groq.Groq(api_key=api_key, timeout=180)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_prompt(title, text, pred)}],
            tools=[{"type": "browser_search"}],
            tool_choice="required",
            reasoning_effort=effort,
            max_completion_tokens=12000,
        )
    except groq.AuthenticationError:
        return {"error": "Groq rejected the API key. Check GROQ_API_KEY in the .env file.", "code": "no_key"}
    except groq.RateLimitError:
        return {"error": "Groq rate limit reached. Wait a minute and try again.", "code": "retry"}
    except groq.APITimeoutError:
        return {"error": "The web check took too long and was stopped. Try again.", "code": "retry"}
    except groq.APIConnectionError:
        return {"error": "Could not reach Groq. Check your internet connection.", "code": "retry"}
    except groq.APIStatusError as e:
        return {"error": f"Groq API error ({e.status_code}): {e.message}", "code": "retry"}

    message = resp.choices[0].message
    pages, corpus = retrieved_pages(message)
    try:
        data = parse_answer(message.content or "")
    except ValueError:
        return {"error": "The fact-check reply could not be read. Try again.", "code": "retry"}
    claims = validate(data, pages, corpus)
    if not claims:
        return {"error": "The fact-check returned no checkable claims. Try again.", "code": "retry"}
    category, rule = aggregate(claims, data)
    result = {
        "category": category,
        "rule": rule,
        "evidence_strength": evidence_strength(claims) if category != "unverified" else
                             ("weak" if any(c["sources"] for c in claims) else "none"),
        "summary": str(data.get("summary", ""))[:600],
        "article_type": data.get("article_type") if data.get("article_type") in ("news", "opinion", "satire") else "other",
        "headline_check": {"misleading": bool((data.get("headline_check") or {}).get("misleading")),
                           "note": str((data.get("headline_check") or {}).get("note", ""))[:300]},
        "timeliness": {"old_news_presented_as_new": bool((data.get("timeliness") or {}).get("old_news_presented_as_new")),
                       "note": str((data.get("timeliness") or {}).get("note", ""))[:300]},
        "claims": claims,
        "pages_retrieved": len(pages),
        "model": resp.model,
    }
    _cache[key] = result
    return result


def check(title, text, pred):
    """Fact-check one article. Dispatches to NVIDIA NIM (GLM 5.3 Flash) or Groq."""
    key = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]

    env = load_env()
    nvidia_key = env.get("NVIDIA_API_KEY", "").strip()
    groq_key = env.get("GROQ_API_KEY", "").strip()

    valid_nvidia = bool(nvidia_key and not nvidia_key.startswith("paste") and not nvidia_key.startswith("nvapi-paste"))
    valid_groq = bool(groq_key and not groq_key.startswith("paste") and not groq_key.startswith("gsk-paste") and not groq_key.startswith("gsk_.."))

    if valid_groq:                      # Groq is preferred when a key is set (NVIDIA's free endpoint can be very slow)
        return _check_groq(title, text, pred, env, key, groq_key)
    if valid_nvidia:
        return _check_nvidia(title, text, pred, env, key, nvidia_key)

    return {
        "error": "No valid API key found in the .env file. Please paste your key: NVIDIA_API_KEY=nvapi-... (for GLM 5.3)",
        "code": "no_key",
    }
