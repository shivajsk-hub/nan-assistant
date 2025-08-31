# ================== NaN — Your Personal AI Assistant ==================
# Free, Fresh Meta-Search: Google News RSS + SearXNG + GDELT + DDG(today) + HN + arXiv + Wikipedia
# Groq LLM (free), Whisper STT, gTTS TTS, RAG (FAISS optional), Rolling Summary (no repetition),
# Self-Learning, Human-like tone, Model switcher, Sources panel, Shareable link.
# Creator: shiva jsk
# ----------------------------------------------------------------------

import os, json, tempfile, re, time, datetime
import numpy as np
import gradio as gr
from dotenv import load_dotenv

# FAISS optional
try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from pypdf import PdfReader
import docx, markdown
from gtts import gTTS
from faster_whisper import WhisperModel
import torch
from groq import Groq
import requests, feedparser

# ========== boot & config ==========
load_dotenv()
BASE = "./nan_data"; os.makedirs(BASE, exist_ok=True)
MEM_FILE      = f"{BASE}/memory.json"
SUMMARY_FILE  = f"{BASE}/summary.json"
INDEX_FILE    = f"{BASE}/faiss.index"
STORE_FILE    = f"{BASE}/docs.json"
FACTS_FILE    = f"{BASE}/facts.json"
FEEDBACK_FILE = f"{BASE}/feedback.json"

CREATOR_NAME = "shiva jsk"

LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_KEY      = os.getenv("GROQ_API_KEY", "").strip()

QUALITY_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL    = "llama-3.1-8b-instant"
CURRENT_MODEL = os.getenv("MODEL_ID", QUALITY_MODEL)

SEARXNG_URL   = os.getenv("SEARXNG_URL", "").strip()  # optional, no key

print(f"[NaN] Provider: {LLM_PROVIDER.upper()} | Model: {CURRENT_MODEL} | FAISS: {FAISS_OK} | SearXNG: {bool(SEARXNG_URL)}")

# ========== memory & summary ==========
def _load_json(path, default):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
    except:
        pass
    return default

memory   = _load_json(MEM_FILE, [])
facts    = _load_json(FACTS_FILE, {})
feedback = _load_json(FEEDBACK_FILE, [])
summary_state = _load_json(SUMMARY_FILE, {"brief": ""})

MAX_TURNS_WINDOW = 8  # last N user+assistant turns

def save_memory():   json.dump(memory,   open(MEM_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
def save_facts():    json.dump(facts,    open(FACTS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
def save_feedback(): json.dump(feedback, open(FEEDBACK_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
def save_summary():  json.dump(summary_state, open(SUMMARY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def build_context_messages():
    recent = memory[-(MAX_TURNS_WINDOW*2):] if len(memory) > (MAX_TURNS_WINDOW*2) else memory[:]
    return summary_state.get("brief", ""), recent

def update_running_summary():
    if len(memory) < (MAX_TURNS_WINDOW*2 + 4):
        return
    compact = "\n".join([f"{m['role']}: {m['content'][:300]}" for m in memory[-(MAX_TURNS_WINDOW*4):]])
    msgs = [
        {"role":"system","content":"You write ultra-brief, factual conversation summaries."},
        {"role":"user","content": compact + "\n\nSummarize key facts, goals, decisions, and personal info in 1-3 short sentences. Do NOT repeat dialogue; only essentials."}
    ]
    try:
        client = Groq(api_key=GROQ_KEY)
        r = client.chat.completions.create(
            model=CURRENT_MODEL, messages=msgs, temperature=0.3, max_tokens=140
        )
        summary_state["brief"] = r.choices[0].message.content.strip()
        save_summary()
    except Exception:
        pass

# ========== embeddings / RAG ==========
embed = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index, doc_store = None, []
if FAISS_OK and os.path.exists(INDEX_FILE) and os.path.exists(STORE_FILE):
    try:
        faiss_index = faiss.read_index(INDEX_FILE)
        doc_store = json.load(open(STORE_FILE, "r", encoding="utf-8"))
        print(f"[NaN] Loaded FAISS with {len(doc_store)} chunks")
    except:
        faiss_index, doc_store = None, []

def _read_any_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return "\n".join([(p.extract_text() or "") for p in PdfReader(path).pages])
        if ext == ".docx":
            return "\n".join([p.text for p in docx.Document(path).paragraphs])
        if ext in [".md", ".markdown"]:
            html = markdown.markdown(open(path, "r", encoding="utf-8", errors="ignore").read())
            return re.sub(r"<[^>]+>", " ", html)
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    except:
        return ""

def add_docs(files):
    global faiss_index, doc_store
    if not files:
        return "No files."
    if not FAISS_OK:
        return "FAISS not available (install faiss-cpu). Skipping indexing."
    added = 0
    for f in files:
        path = f if isinstance(f, str) else f.name
        text = _read_any_file(path)
        if not text.strip(): continue
        chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
        for c in chunks:
            vec = embed.encode(c, normalize_embeddings=True).astype("float32")
            if faiss_index is None:
                faiss_index = faiss.IndexFlatIP(vec.shape[0])
            faiss_index.add(np.array([vec]))
            doc_store.append(c); added += 1
    if faiss_index:
        faiss.write_index(faiss_index, INDEX_FILE)
        json.dump(doc_store, open(STORE_FILE, "w", encoding="utf-8"))
    return f"Indexed {added} chunks."

def rag_search(q, k=4):
    if not (FAISS_OK and faiss_index and doc_store): return []
    vec = embed.encode([q], normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(vec, k)
    return [doc_store[i] for i in I[0] if 0 <= i < len(doc_store)]

# ========== whisper (stt) / gtts (tts) ==========
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    whisper = WhisperModel("small",
        device=whisper_device,
        compute_type="float16" if whisper_device == "cuda" else "int8"
    )
    print("[NaN] Whisper ready.")
except Exception as e:
    print("[NaN] Whisper init failed; falling back to tiny. Error:", e)
    whisper = WhisperModel("tiny", device=whisper_device)

def transcribe(audio_path: str) -> str:
    try:
        segs, _ = whisper.transcribe(audio_path, beam_size=1)
        return " ".join([s.text for s in segs]).strip()
    except Exception as e:
        return f"(STT error: {e})"

def speak(text: str) -> str:
    t = gTTS(text=text, lang="en")
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    t.save(out); return out

# ========== self-learning ==========
FACT_PATTERNS = [
    (r"\bmy name is ([A-Za-z][\w ]+)", "user_name"),
    (r"\bi am (\d+)\s*years? old\b", "user_age"),
    (r"\bmy birthday is ([A-Za-z0-9 ,/-]+)", "user_birthday"),
    (r"\bi live in ([A-Za-z][\w ,.-]+)", "user_location"),
    (r"\bi (?:like|love) ([A-Za-z][\w ,.-]+)", "user_likes"),
    (r"\bi (?:dislike|hate) ([A-Za-z][\w ,.-]+)", "user_dislikes"),
    (r"\bmy email is ([A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+)", "user_email"),
]

def autolearn_scan_and_store(user_text: str) -> list[str]:
    learned = []
    low = user_text.lower()
    if any(t in low for t in ["remember","note that","save this","store this","keep this","my name","birthday","i like","i love"]):
        for pat, key in FACT_PATTERNS:
            m = re.search(pat, low)
            if m:
                val = m.group(1).strip()
                if key in ["user_likes","user_dislikes"]:
                    arr = set(facts.get(key, [])); arr.add(val); facts[key] = sorted(list(arr))
                else:
                    facts[key] = val
                learned.append(f"{key} = {val}")
        if learned: save_facts()
    return learned

def teach_fact_manual(fact_text: str) -> str:
    if not fact_text or not fact_text.strip(): return "Nothing to learn."
    arr = facts.get("notes", [])
    arr.append(fact_text.strip())
    facts["notes"] = arr; save_facts()
    return "Learned and saved."

def facts_context() -> str:
    if not facts: return ""
    lines = [f"- creator: {CREATOR_NAME}"]
    for k, v in facts.items():
        if isinstance(v, list):
            lines.append(f"- {k}: {', '.join(v)}")
        else:
            lines.append(f"- {k}: {v}")
    return "USER FACTS:\n" + "\n".join(lines)

# ========== FREE, FRESH ENGINES ==========
def engine_google_news_rss(query: str, k: int = 8, locale="en-IN"):
    try:
        hl = locale
        country = (locale.split("-")[1] if "-" in locale else "US").upper()
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl={hl}&gl={country}&ceid={country}:{hl.split('-')[0]}"
        feed = feedparser.parse(url)
        out = []
        for entry in feed.entries[:k]:
            out.append({"title": entry.get("title",""), "url": entry.get("link",""),
                        "snippet": entry.get("summary",""), "engine": "google-news"})
        return out
    except Exception as e:
        print("[NaN] Google News RSS error:", e); return []

def engine_searxng(query: str, k: int = 6, time_range: str = "day", language="en"):
    if not SEARXNG_URL: return []
    try:
        params = {"q": query, "format": "json", "language": language,
                  "time_range": time_range, "safesearch": 1}
        r = requests.get(f"{SEARXNG_URL.rstrip('/')}/search", params=params, timeout=15,
                         headers={"User-Agent":"NaN/1.0"})
        r.raise_for_status()
        data = r.json().get("results", [])
        out=[]
        for res in data[:k]:
            title = res.get("title",""); url = res.get("url","")
            snippet = res.get("content","") or res.get("snippet","")
            out.append({"title": title, "url": url, "snippet": snippet, "engine": "searxng"})
        return out
    except Exception as e:
        print("[NaN] SearXNG error:", e); return []

def engine_gdelt_rss(query: str, k: int = 8):
    try:
        url = "https://api.gdeltproject.org/api/v2/search/rss"
        params = {"query": f'"{query}"', "mode": "ArtList", "maxrecords": str(k),
                  "sort": "DateDesc", "timelimit": "24"}
        r = requests.get(url, params=params, timeout=15, headers={"User-Agent":"NaN/1.0"})
        r.raise_for_status()
        feed = feedparser.parse(r.text)
        out=[]
        for entry in feed.entries[:k]:
            out.append({"title": entry.get("title",""), "url": entry.get("link",""),
                        "snippet": entry.get("summary",""), "engine": "gdelt"})
        return out
    except Exception as e:
        print("[NaN] GDELT error:", e); return []

def engine_ddg_fresh(query: str, k: int = 8, timelimit="d"):
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=k, safesearch="moderate", timelimit=timelimit)
        out=[]
        for h in hits:
            out.append({"title": h.get("title",""), "url": h.get("href",""),
                        "snippet": h.get("body",""), "engine": "ddg"})
        return out
    except Exception as e:
        print("[NaN] DDG error:", e); return []

def engine_hn(query: str, k: int = 6):
    try:
        r = requests.get("https://hn.algolia.com/api/v1/search",
                         params={"query": query, "tags": "story", "hitsPerPage": k,
                                 "restrictSearchableAttributes": "title"},
                         headers={"User-Agent":"NaN/1.0"}, timeout=12)
        r.raise_for_status()
        data = r.json().get("hits",[])
        out=[]
        for item in data[:k]:
            url = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID')}"
            out.append({"title": item.get("title",""), "url": url,
                        "snippet": f"points: {item.get('points')} • by {item.get('author')}",
                        "engine": "hackernews"})
        return out
    except Exception as e:
        print("[NaN] HN error:", e); return []

def engine_arxiv(query: str, k: int = 6):
    try:
        endpoint = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "start": 0, "max_results": k}
        r = requests.get(endpoint, params=params, timeout=15, headers={"User-Agent":"NaN/1.0"})
        r.raise_for_status()
        feed = feedparser.parse(r.text)
        out=[]
        for e in feed.entries[:k]:
            title = e.get("title","").replace("\n"," ").strip()
            link = e.get("link","")
            summary = e.get("summary","").replace("\n"," ").strip()
            out.append({"title": title, "url": link, "snippet": summary, "engine": "arxiv"})
        return out
    except Exception as e:
        print("[NaN] arXiv error:", e); return []

def engine_wikipedia_brief(query: str):
    try:
        r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/"+query.replace(" ","_"),
                         headers={"User-Agent":"NaN/1.0"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return [{
                "title": data.get("title","Wikipedia"),
                "url": data.get("content_urls",{}).get("desktop",{}).get("page",""),
                "snippet": data.get("extract",""),
                "engine": "wikipedia"
            }]
    except Exception as e:
        print("[NaN] Wikipedia error:", e)
    return []

def normalize_domain(url):
    try:
        dom = re.sub(r"^https?://(www\.)?","",url).split("/")[0]
        return dom.lower()
    except:
        return url

def dedupe_merge(items):
    seen=set(); out=[]
    for it in items:
        dom = normalize_domain(it.get("url",""))
        key = (dom, it.get("title","").strip().lower())
        if key in seen: continue
        seen.add(key); out.append(it)
    return out

def meta_search_free(query: str, k: int = 8, locale="en-IN"):
    """
    Advanced FREE meta-search:
      Google News RSS + SearXNG(day) + GDELT(24h) + DDG(day) + HN(tech) + arXiv(research) + Wikipedia fallback
    """
    bucket = []
    bucket += engine_google_news_rss(query, k=max(4, k//2), locale=locale)
    bucket += engine_gdelt_rss(query, k=max(3, k//3))
    bucket += engine_searxng(query, k=max(3, k//3), time_range="day", language="en")
    bucket += engine_ddg_fresh(query, k=max(3, k//3), timelimit="d")

    lower = query.lower()
    if any(t in lower for t in ["ai","ml","startup","github","python","javascript","react","openai",
                                "gpu","nvidia","apple","google","microsoft","meta","privacy","security",
                                "hack","linux","docker","kubernetes","cloud","aws","gcp","azure",
                                "android","ios","xcode","visual studio"]):
        bucket += engine_hn(query, k=4)

    if any(t in lower for t in ["paper","arxiv","research","study","dataset","benchmark",
                                "conference","neurips","icml","iclr","cvpr","acl"]):
        bucket += engine_arxiv(query, k=4)

    if not bucket:
        bucket += engine_wikipedia_brief(query)

    return dedupe_merge(bucket)[:k]

def format_sources(items):
    if not items: return "No sources."
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"*Sources (as of {stamp})*:"]
    for it in items:
        title = it.get("title","").strip() or "(no title)"
        url   = it.get("url","").strip()
        eng   = it.get("engine","free")
        lines.append(f"- [{title}]({url})  · {eng}")
    return "\n".join(lines)

def format_web_ctx(items):
    if not items: return ""
    lines = ["WEB RESULTS:"]
    for it in items:
        title = it.get("title","").strip()
        url   = it.get("url","").strip()
        snip  = it.get("snippet","").strip()
        eng   = it.get("engine","free")
        lines.append(f"- {title}  · {eng}\n  {url}\n  {snip}")
    return "\n".join(lines)

# ========== LLM ==========
def chat_complete(history, user_text, rag_ctx="", model_id=None, tone="friendly"):
    if not GROQ_KEY:
        return "⚠ Missing GROQ_API_KEY in .env"

    brief_summary, recent = build_context_messages()

    client = Groq(api_key=GROQ_KEY)
    sys_prompt = (
        f"You are NaN — a helpful, human-like assistant created by {CREATOR_NAME}.\n"
        f"- Adopt a {tone} tone. Use contractions. No robotic phrasing.\n"
        f"- Do NOT repeat earlier messages verbatim; avoid echoing user prompts.\n"
        f"- If WEB/RAG context is provided, integrate it and cite sources in plain text.\n"
        f"- If you're unsure, say what you'd need to check.\n"
        f"- Prefer bullets for multi-step instructions; keep answers concise."
    )

    ctx_parts = []
    if brief_summary: ctx_parts.append("CONVERSATION-SUMMARY:\n" + brief_summary)
    if rag_ctx:       ctx_parts.append(rag_ctx)
    context_blob = "\n\n".join(ctx_parts)

    messages = [{"role":"system","content":sys_prompt}]
    messages += recent
    final_user = (context_blob + "\n\n" + user_text).strip() if context_blob else user_text
    messages.append({"role":"user","content":final_user})

    try:
        r = client.chat.completions.create(
            model=(model_id or CURRENT_MODEL),
            messages=messages,
            max_tokens=350,
            temperature=0.55,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"

# ========== main pipeline ==========
def respond(text, audio, files, speak_out, use_rag, use_web, model_pick, thumbs, teach_text, tone):
    global CURRENT_MODEL
    CURRENT_MODEL = model_pick or CURRENT_MODEL

    teach_status = ""
    if teach_text and teach_text.strip():
        teach_status = teach_fact_manual(teach_text)

    if files:
        add_docs(files)

    if audio and not text:
        text = transcribe(audio)

    if not text or not str(text).strip():
        return "⚠ Say/type something.", None, "Idle.", "No sources."

    learned_bits = autolearn_scan_and_store(text)

    # Freshness heuristic
    ql = text.lower()
    fresh = any(t in ql for t in [
        "today","latest","now","news","price","score","weather",
        "who won","released","this week","breaking","live","real-time","realtime","trending"
    ])
    if fresh: use_web = True

    ctx_parts = []
    sources_md = "No sources."
    web_items = []

    if use_rag:
        docs = rag_search(text, 4)
        if docs: ctx_parts.append("RAG:\n" + "\n\n".join(docs))

    if use_web:
        web_items = meta_search_free(text, k=8, locale="en-IN")
        ctx_parts.append(format_web_ctx(web_items) if web_items else "WEB: (no results)")
        sources_md = format_sources(web_items)

    ctx = "\n\n".join(ctx_parts) if ctx_parts else ""

    reply = chat_complete(memory, text, rag_ctx=ctx, model_id=CURRENT_MODEL, tone=tone)

    memory.append({"role":"user","content":text})
    memory.append({"role":"assistant","content":reply})
    save_memory()
    update_running_summary()

    if thumbs in ("👍","👎"):
        feedback.append({"q":text, "a":reply, "rating":thumbs, "ts":time.time()})
        save_feedback()

    status = f"Model: {CURRENT_MODEL} • Tone: {tone}"
    if learned_bits: status += " • Learned: " + "; ".join(learned_bits)
    if teach_status: status += " • " + teach_status

    audio_out = speak(reply) if speak_out else None
    return reply, audio_out, status, sources_md

def clear_memory():
    global memory
    memory = []; save_memory()
    summary_state["brief"] = ""; save_summary()
    return "🧹 Memory cleared."

# ========== gradio ui ==========
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(f"## 🤖 NaN — Your Personal AI Assistant  \nby *{CREATOR_NAME}*")

    with gr.Row():
        with gr.Column(scale=3):
            out = gr.Textbox(label="Reply", lines=16, show_copy_button=True)
            voice = gr.Audio(label="Voice", autoplay=True)
            status = gr.Markdown("Model: —")
            sources = gr.Markdown("No sources.")

        with gr.Column(scale=2):
            text = gr.Textbox(label="Type here")
            mic  = gr.Audio(sources=["microphone","upload"], type="filepath", label="Speak")
            files= gr.Files(label="Upload docs (PDF/TXT/MD/DOCX)", file_count="multiple",
                            file_types=[".pdf",".txt",".docx",".md",".markdown"])
            speak_cb = gr.Checkbox(value=True, label="🔊 Speak reply")
            rag_cb   = gr.Checkbox(value=True, label="📂 Use RAG on uploads")
            web_cb   = gr.Checkbox(value=True, label="🌍 Use Web (auto for fresh queries)")
            model_pick = gr.Dropdown(
                label="Model", choices=[QUALITY_MODEL, FAST_MODEL], value=CURRENT_MODEL
            )
            tone = gr.Dropdown(
                label="Tone", choices=["friendly","professional","playful"], value="friendly"
            )
            teach_text = gr.Textbox(label="Teach NaN (persist a fact/note)")
            thumbs = gr.Radio(choices=["👍","👎","(no feedback)"], value="(no feedback)", label="Feedback")
            with gr.Row():
                go = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🧹 Clear Memory")

    def _run(text, mic, files, speak_cb, rag_cb, web_cb, model_pick, thumbs, teach_text, tone):
        return respond(text, mic, files, speak_cb, rag_cb, web_cb, model_pick, thumbs, teach_text, tone)

    go.click(_run, [text, mic, files, speak_cb, rag_cb, web_cb, model_pick, thumbs, teach_text, tone],
             [out, voice, status, sources])

    clear_btn.click(fn=lambda: clear_memory(), inputs=None, outputs=[status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)