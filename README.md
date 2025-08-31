# nan-assistant
# 🤖 NaN Assistant

NaN is a personal AI assistant built by *Shiva JSK*.  
It combines a strong LLM backend (Groq) with real-time web search and RAG (documents) to provide up-to-date, context-aware answers.

---

## ✨ Features
- Groq LLM backend  
  Uses llama-3.1-8b-instant by default (fast), or llama-3.1-70b-versatile (stronger).
- Realtime Web Search  
  Integrates:
  - SearXNG (meta-search, free & open-source)  
  - Google News RSS (fresh headlines)  
  - DuckDuckGo Today (day-limited results)  
- RAG (Retrieval-Augmented Generation)  
  Upload PDFs, DOCX, TXT, or MD files → ask questions and get contextual answers.  
- Persistent Memory  
  Remembers the last N turns of conversation (MAX_HISTORY), and optionally saves memory across sessions (PERSIST_MEMORY).  
- Text-to-Speech  
  Replies can be spoken aloud using gTTS.  
- Source References  
  Shows links when web results are used, so answers are transparent.  

---

## ⚙ Setup

### Requirements
- Python 3.10+
- Packages from requirements.txt

### Environment Variables / Secrets
---

## 🚀 Run Locally
Clone this repo and run:
Then open: http://127.0.0.1:7860

---

## 🌐 Deploy on Hugging Face Spaces
1. Create a new Space → SDK = Gradio → choose “From Repository”.  
2. Paste this GitHub repo URL.  
3. Add your secrets and variables in Settings → Variables and secrets.  
4. Hugging Face will auto-build and give you a permanent link:  
   https://<username>-nan-assistant.hf.space

---

## 👨‍💻 Creator
NaN Assistant was created by *Shiva JSK* 💙

Go to Hugging Face → Settings → Variables and secrets and add:

Repository secrets (private keys):
