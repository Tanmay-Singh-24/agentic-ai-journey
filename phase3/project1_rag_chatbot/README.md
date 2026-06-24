# 📄 PaperMind

**A conversational RAG chatbot that answers questions about 5 landmark AI papers — grounded in their actual content, with citations.**

Ask *"What is multi-head attention?"* or *"How was InstructGPT trained?"* and get an answer drawn straight from the papers, attributed to the source. PaperMind refuses to guess: if something isn't in the papers, it says so.

---

## The problem it solves

Research papers are dense, long, and hard to search. Finding *"the one paragraph that explains X"* across several 20–60 page PDFs means a lot of skimming. A plain chatbot (e.g. ChatGPT) can help, but it will happily **hallucinate** details it doesn't actually know, and it can't point you to *where* a claim came from.

PaperMind fixes both problems with **Retrieval-Augmented Generation (RAG)**: it retrieves the most relevant passages from the real papers and forces the model to answer **only** from them — citing the source paper each time.

---

## The papers

| Paper | Why it matters |
|-------|----------------|
| **Attention Is All You Need** (2017) | Introduced the Transformer — the architecture behind every modern LLM. |
| **BERT** (2018) | Showed bidirectional pre-training; kicked off the "pre-train then fine-tune" era. |
| **GPT-3** (2020) | Demonstrated that scale alone unlocks few-shot, in-context learning. |
| **InstructGPT** (2022) | Aligned LLMs to human intent via RLHF — the recipe behind ChatGPT. |
| **ReAct** (2022) | Combined reasoning + acting (tool use), the foundation of modern AI agents. |

---

## Tech stack

- **Python**
- **LangGraph** — orchestrates the retrieve → generate flow as a stateful graph, with native conversation memory
- **LangChain** — document loading, splitting, and integrations
- **ChromaDB** — persistent vector store for the embedded paper chunks
- **HuggingFace embeddings** — `all-MiniLM-L6-v2` (384-dim sentence embeddings)
- **Groq** — `llama-3.3-70b-versatile` for fast, grounded answer generation
- **Streamlit** — minimal chat UI

---

## How it works

```
                          ┌─────────────────── one-time ingest ───────────────────┐
   5 PDFs ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► HuggingFace ──► ChromaDB
              (per page)       (1000 chars, 100 overlap)         embeddings      (persisted)
                          └────────────────────────────────────────────────────────┘

   per question (LangGraph):

      question ──► [retrieve] ──► [generate] ──► grounded, cited answer
                       │              │
            top-5 chunks by      LLM answers using ONLY those chunks
            cosine similarity    + the running conversation history
```

1. **Ingest (once).** The 5 PDFs are loaded, split into ~1000-character chunks (100-char overlap so facts aren't cut across boundaries), embedded, and stored in ChromaDB. Each chunk is tagged with its **source paper** so answers can cite it. The store is **persisted to disk** — on later runs it's loaded instantly instead of re-embedding.
2. **Retrieve.** For each question, the `retrieve` node embeds it with the *same* model and pulls the **top-5** most semantically similar chunks.
3. **Generate.** The `generate` node hands those chunks (plus the conversation so far) to the LLM, with a system prompt that says: answer only from the context, admit when you don't know, and cite the paper.
4. **Remember.** LangGraph's **checkpointer** stores each conversation by `thread_id`, so follow-up questions keep their context — no manual history wiring.

---

## Setup & run

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com).

```bash
# 1. From the repo root, activate your virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r phase3/project1_rag_chatbot/requirements.txt

# 3. Make sure your Groq key is in phase1/.env
#    GROQ_API_KEY=your_key_here

# 4. Launch the app (from the project folder)
cd phase3/project1_rag_chatbot
streamlit run frontend/app.py
```

The **first launch** embeds all 5 papers (takes ~30–60s, one time only) and writes the vector store to `chroma_store/`. Every launch after that loads it instantly.

---

## Project structure

```
project1_rag_chatbot/
├── papers/              # the 5 source PDFs
├── backend/
│   ├── __init__.py
│   └── rag.py           # the whole RAG pipeline: ingest, retrieve, generate, memory
├── frontend/
│   └── app.py           # Streamlit chat UI (thin — all logic is in the backend)
├── chroma_store/        # persisted vector store (created on first run)
├── requirements.txt
└── README.md
```
