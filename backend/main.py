from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import requests
import trafilatura
from textblob import TextBlob
from urllib.parse import urlparse
import re
import threading
import nltk

# Initialize necessary NLP data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass

app = FastAPI(title="Vérifié — Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# FAULT-TOLERANT NEURAL HUB (v3.3)
# ---------------------------------------------------------------------

MODEL_1 = "vikram71198/distilroberta-base-finetuned-fake-news-detection"
MODEL_2 = "jy46604790/Fake-News-Bert-Detect"
MODEL_3 = "Hello-SimpleAI/chatgpt-detector-roberta"

model_status = {
    "engine_1": "DOWNLOADING",
    "engine_2": "DOWNLOADING",
    "engine_3": "DOWNLOADING",
    "entity_engine": "DOWNLOADING"
}

classifier_1 = None
classifier_2 = None
classifier_3 = None
ner_engine = None

def load_models():
    global classifier_1, classifier_2, classifier_3, ner_engine
    
    print(f"Initializing Neural Hub...")
    
    # Engine 1 [DistilRoBERTa] - High Accuracy
    try:
        print(f" - Engine 1 [News-Core] SYNCING...")
        classifier_1 = pipeline("text-classification", model=MODEL_1)
        model_status["engine_1"] = "ONLINE"
        print(f" - Engine 1 [News-Core] ONLINE")
    except Exception as e:
        model_status["engine_1"] = f"OFFLINE: {str(e)}"
        print(f" - Engine 1 [News-Core] OFFLINE: {e}")

    # Engine 2 [BERT-Verification] - Secondary Polish
    try:
        print(f" - Engine 2 [Verification] SYNCING...")
        classifier_2 = pipeline("text-classification", model=MODEL_2)
        model_status["engine_2"] = "ONLINE"
        print(f" - Engine 2 [Verification] ONLINE")
    except Exception as e:
        model_status["engine_2"] = f"OFFLINE: {str(e)}"
        print(f" - Engine 2 [Verification] OFFLINE: {e}")

    # Engine 3 [AI-Content Detection]
    try:
        print(f" - Engine 3 [AI-Content] SYNCING...")
        classifier_3 = pipeline("text-classification", model=MODEL_3)
        model_status["engine_3"] = "ONLINE"
        print(f" - Engine 3 [AI-Content] ONLINE")
    except Exception as e:
        model_status["engine_3"] = f"OFFLINE: {str(e)}"
        print(f" - Engine 3 [AI-Content] OFFLINE: {e}")

    # NER Engine
    try:
        ner_engine = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        model_status["entity_engine"] = "ONLINE"
        print(f" - Entity Engine [NER] ONLINE")
    except Exception as e:
        model_status["entity_engine"] = f"OFFLINE: {str(e)}"

# Start background loading
threading.Thread(target=load_models, daemon=True).start()

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "theguardian.com", 
    "npr.org", "wsj.com", "bloomberg.com", "afp.com", "axios.com", "dw.com", 
    "france24.com", "aljazeera.com", "reuters.co.uk", "news.sky.com"
}

SUSPICIOUS_DOMAINS = {
    "infowars.com", "naturalnews.com", "thegatewaypundit.com", "breitbart.com", 
    "yournewswire.com", "worldnewsdailyreport.com", "dailywire.com",
    "theepochtimes.com", "bignewsnetwork.com", "americanthinker.com"
}

FACT_CHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "reuters.com/fact-check"
}

def get_readability(text: str):
    """Simple Flesch-Kincaid implementation."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    if not words or not sentences: return 50.0
    
    def count_syllables(word):
        word = word.lower()
        if len(word) <= 3: return 1
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels: count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"): count -= 1
        return max(1, count)

    syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
    return min(max(score, 0), 100)

def analyze_linguistics(text: str):
    blob = TextBlob(text)
    subj = blob.sentiment.subjectivity
    pola = blob.sentiment.polarity
    
    words = text.split()
    if not words: return {}
    
    exclamation_count = text.count('!')
    caps_count = sum(1 for word in words if word.isupper() and len(word) > 2)
    emotional_words_count = sum(1 for word in words if word.lower() in {"shocking", "urgent", "must", "alert", "hidden", "truth", "exposed"})
    
    clickbait_index = (exclamation_count * 0.1 + caps_count * 0.2 + emotional_words_count * 0.3) / (len(words) / 50 + 1)
    clickbait_index = min(clickbait_index, 1.0)
    
    readability = get_readability(text)
    
    bias_keywords = {
        "radical", "conspiracy", "mainstream media", "msm", "deep state", 
        "hoax", "propaganda", "brainwashed", "censorship", "unmasking",
        "woke", "regime", "corrupt", "illegal", "traitor", "scandal"
    }
    bias_hits = sum(1 for word in words if word.lower() in bias_keywords)
    bias_intensity = min(bias_hits / (len(words)/20 + 1), 1.0)
    
    return {
        "subjectivity": subj,
        "sentiment_intensity": abs(pola),
        "formality": 1.0 - clickbait_index,
        "readability": readability,
        "bias_intensity": bias_intensity,
        "is_sensational": clickbait_index > 0.5 or (abs(pola) > 0.4 and subj > 0.5) or bias_hits > 2
    }

def check_domain_trust(url: str):
    if not url: return "UNKNOWN", 0.5, []
    domain = urlparse(url).netloc.replace("www.", "")
    if domain in TRUSTED_DOMAINS: return "TRUSTED", 0.95, ["Recognized high-credibility news outlet."]
    if domain in SUSPICIOUS_DOMAINS: return "UNRELIABLE", 0.05, ["History of distributing misinformation."]
    if any(fc in domain for fc in FACT_CHECK_DOMAINS): return "FACT-CHECKER", 0.9, ["Independent fact-checking organization."]
    return "UNKNOWN", 0.5, ["Unverified source credibility."]

@app.get("/api/status")
async def get_status():
    return {
        "status": "operational" if any(s == "ONLINE" for s in model_status.values()) else "syncing",
        "engines": model_status
    }

@app.get("/api/health")
async def health():
    return {"status": "operational", "active_engines": [k for k, v in model_status.items() if v == "ONLINE"]}

class NewsRequest(BaseModel):
    text: str

@app.post("/api/detect")
async def detect_news(req: NewsRequest):
    if not classifier_1 and not classifier_2:
        return {"error": "Intelligence Core is still initializing. Try again in 30s."}
    
    actual_text = req.text.strip()
    if not actual_text: return {"error": "Empty content."}
    
    is_url = actual_text.startswith("http")
    source_url = actual_text if is_url else None
    
    if is_url:
        try:
            downloaded = trafilatura.fetch_url(actual_text)
            actual_text = trafilatura.extract(downloaded)
            if not actual_text: return {"error": "Extraction failure."}
        except: return {"error": "Scraping failure."}

    try:
        metrics = analyze_linguistics(actual_text)
        m1_fake, m2_fake = False, False
        m1_score, m2_score = 0.5, 0.5
        is_ai, ai_score = False, 0.5
        
        if classifier_1:
            inf1 = classifier_1(actual_text[:1200], truncation=True)[0]
            m1_fake = inf1['label'] == 'LABEL_1'
            m1_score = inf1['score']
            
        if classifier_2:
            inf2 = classifier_2(actual_text[:1200], truncation=True)[0]
            m2_fake = inf2['label'] == 'LABEL_0'
            m2_score = inf2['score']

        if classifier_3:
            inf3 = classifier_3(actual_text[:1200], truncation=True)[0]
            is_ai = inf3['label'] == 'LABEL_1'
            ai_score = inf3['score']

        signals = []
        p_weight, n_weight = 0.0, 0.0
        
        if classifier_1 and classifier_2:
            if m1_fake == m2_fake:
                conf = (m1_score + m2_score)/2
                if m1_fake:
                    n_weight += 0.6 * conf
                    signals.append({"type": "negative", "text": "Neural Consensus: Deep-learning models identify significant disinformation patterns."})
                else:
                    p_weight += 0.6 * conf
                    signals.append({"type": "positive", "text": "Structure Verified: Journalistic integrity and factual tone detected by both engines."})
            else:
                signals.append({"type": "neutral", "text": "Engine Variance: Conflicting markers detected between neural clusters."})
                if m1_fake: n_weight += 0.3 * m1_score
                else: p_weight += 0.3 * m1_score
                if m2_fake: n_weight += 0.2 * m2_score
                else: p_weight += 0.2 * m2_score
        
        if classifier_3:
            if is_ai:
                n_weight += 0.25 * ai_score
                signals.append({"type": "negative", "text": f"Synthetic Attribution: {ai_score:.0%} probability of non-human origin."})
            else:
                p_weight += 0.15 * ai_score
                signals.append({"type": "positive", "text": "Human Authenticity: Natural language variance suggests human authorship."})

        if metrics.get("is_sensational"):
            n_weight += 0.2
            signals.append({"type": "negative", "text": "Sensationalism Alert: High emotional load and clickbait markers detected."})
            
        if is_url:
            status, score, notes = check_domain_trust(source_url)
            if status == "TRUSTED": p_weight += 0.5
            elif status == "UNRELIABLE": n_weight += 0.5
            signals.append({"type": "neutral", "text": f"Source Check: {status} ({urlparse(source_url).netloc})"})

        eps = 0.05
        trust_index = (p_weight + eps) / (p_weight + n_weight + 2*eps)
        verdict = "REAL" if trust_index > 0.6 else "FAKE" if trust_index < 0.4 else "UNCERTAIN"
        
        entities = []
        if ner_engine:
            try:
                res = ner_engine(actual_text[:800])
                unique = {e['word']: e['entity_group'] for e in res if e['entity_group'] in {'PER', 'ORG'}}
                entities = [{"text": k, "type": v} for k, v in list(unique.items())[:5]]
            except: pass

        return {
            "prediction": verdict,
            "overall_trust": trust_index,
            "confidence": (m1_score + m2_score)/2,
            "metrics": metrics,
            "signals": signals,
            "intelligence": {
                "entities": entities,
                "is_ai": is_ai,
                "ai_score": ai_score,
                "correlation": "HIGH" if m1_fake == m2_fake else "CONFLICT",
                "summary": actual_text[:200] + "..."
            }
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"Internal Intel Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
