# CLAUDE.md — Tanmay's Agentic AI Learning Context
> This file is the single source of truth for my learning journey.
> Read this fully before helping me with anything. Every session starts here.
---
## Who I Am
- **Name:** Tanmay
- **Degree:** B.Tech Computer Science (AI/ML specialization)
- **Graduation:** February 2027
- **Current Status:** Preparing for campus placements — they begin in approximately 3-4 months
- **Location:** India
---
## My Goal
Land a job as an **AI Engineer** at a top-tier company.
**Dream targets:** NVIDIA, OpenAI
**Realistic targets:** Any strong AI-first company or well-funded AI startup
This is not casual learning. This is placement preparation with a hard deadline.
---
## My Current Skill Level
### What I Know
- **Python:** Comfortable. I can write clean Python code, understand OOP, work with libraries.
- **Engineering Mathematics:** Solid foundation — linear algebra, calculus, probability.
- **ML Theory (college level):** I have theoretical awareness of:
  - Deep Learning, CNNs, RNNs
  - Computer Vision concepts
  - Reinforcement Learning basics
  - Classical ML algorithms: logistic regression, decision trees, SVMs, etc.
  - I know WHAT these are but have NOT implemented most of them from scratch.
- **DSA:** Practicing daily. Still building proficiency. Doing LeetCode problems every day. Not strong yet but improving.
- **NumPy:** Completed. Covered arrays, indexing, slicing, reshaping, stats (mean, std, argmax), boolean masking, matrix multiplication, normalization, loading from files.
- **Pandas:** Completed. Covered Series, DataFrames, read_csv, loc/iloc, boolean filtering, missing data, apply/map, groupby, merge/concat, NumPy bridge.
### What I Do NOT Know Yet
- LangChain, LlamaIndex, or any LLM orchestration framework
- RAG pipelines (conceptually aware, not implemented)
- Vector databases (Chroma, Pinecone, Weaviate)
- Agentic AI workflows (CrewAI, LangGraph, AutoGen)
- Fine-tuning LLMs
- Deploying AI apps (FastAPI, Streamlit, Docker)
- Production-level AI engineering patterns
---
## The Roadmap (Built with Claude Chat)
This is a 4-phase, ~16-week plan designed for placement readiness.
### Phase 1 — Foundations (Weeks 1–2) ← CURRENT PHASE
**Goal:** Be comfortable with Python for AI, LLM APIs, and understand embeddings + vector search from scratch.
| Days | Topic | Status |
|------|-------|--------|
| 1–2 | numpy, pandas, APIs with requests | NumPy ✅ Pandas ✅ |
| 3–4 | OpenAI/Anthropic API basics — roles, tokens, temperature, streaming | ⬜ Not started |
| 5 | Async Python — asyncio, await | ⬜ Not started |
| 6–7 | Mini project: CLI tool that streams LLM responses | ⬜ Not started |
| 8–9 | Embeddings — what they are, generating them, cosine similarity | ⬜ Not started |
| 10–11 | Vector databases — ChromaDB locally, collections, similarity search | ⬜ Not started |
| 12–13 | Manual RAG pipeline (NO frameworks) — PDF → chunk → embed → store → retrieve → LLM | ⬜ Not started |
| 14 | Consolidate, clean code, push to GitHub | ⬜ Not started |
### Phase 2 — Core Stack (Weeks 3–7)
**Goal:** Master LangChain, build production-quality RAG, build first agent.
- LangChain: chains, prompts, memory, document loaders
- RAG pipeline: chunking → embedding → vector DB → retrieval (with LangChain)
- LangGraph or CrewAI: first agent with tools + memory
- Deploy via FastAPI or Streamlit
### Phase 3 — Build 2 Solid Projects (Weeks 8–13)
**Goal:** Have real, demonstrable, GitHub-hosted projects to show in interviews.
- **Project 1:** RAG-based chatbot (e.g., chat with research papers or college documents)
- **Project 2:** Multi-agent workflow (e.g., AI research agent, coding agent, or data analyst agent)
- Both must have: clean GitHub repo, proper README, live demo if possible
- **Bonus if time permits:** Fine-tune a small model using LoRA on HuggingFace
- **Key:** I must be able to explain every line of how it works — interviewers will probe deeply
### Phase 4 — Placement Prep (Weeks 14–16)
**Goal:** Be interview-ready across all rounds.
- DSA: Focus on arrays, trees, graphs, dynamic programming — LeetCode medium level
- ML fundamentals: Explain transformers, attention, RAG verbally and confidently
- System design: How to architect a production RAG system
- Resume: Lead with projects, not coursework
---
## Key Decisions & Philosophy
- **Why Agentic AI?** Highest ROI skill for a fresher in 2025-26. Companies are actively hiring for this and there aren't enough people. It differentiates me from standard CS grads.
- **ML theory vs implementation:** I do NOT need to implement backprop or train CNNs. I need to understand transformers, fine-tuning vs RAG vs prompting, embeddings, and LLM limitations — well enough to discuss in interviews.
- **Projects are everything.** At my stage, projects ARE my resume. Two solid, well-documented, demonstrable projects outweigh any certification or course.
- **Build RAG manually before using LangChain.** This is intentional — understanding the internals before using the abstraction makes me far stronger in interviews.
- **DSA runs parallel the whole time.** Not batched to the end. 1-2 problems daily.
---
## My Project Folder Structure
```
agentic-ai-journey/
├── CLAUDE.md                  ← This file
├── phase1/
│   ├── numpy_practice.ipynb   ← Done ✅
│   ├── pandas_practice.ipynb  ← Done ✅
│   ├── students.csv           ← Sample dataset for pandas practice
│   ├── api_basics/            ← Coming next (Day 3-4)
│   └── manual_rag/            ← End of Phase 1
├── phase2/
├── phase3/
│   ├── project1_rag_chatbot/
│   └── project2_multi_agent/
└── resources.md               ← Links, references
```
---
## Tools & Environment
- **Python version:** 3.13
- **Editor:** VS Code + Jupyter Notebooks
- **Libraries installed so far:** numpy, pandas, requests
- **Libraries to install next:** anthropic or openai, langchain, chromadb, sentence-transformers, fastapi, streamlit
- **Version control:** GitHub (repo: `agentic-ai-journey`)
---
## How to Help Me
- **Be direct.** Don't over-explain things I already know. I know Python. Skip the basics.
- **Give working code.** Not pseudocode. Real, runnable code I can paste and execute.
- **Tell me what to do, step by step.** I'm in execution mode, not exploration mode.
- **Call out mistakes clearly.** If my code or understanding is wrong, say so directly.
- **Don't teach what I don't need yet.** Stick to what's relevant to the current phase.
- **If I ask you to build or improve something**, do it completely — don't give me half a solution.
- **Keep the roadmap in mind.** Every task should connect back to making me placement-ready.
---
## Current Task
> **Starting Phase 1, Day 3-4**
> Pandas is done. Next: Anthropic/OpenAI API basics — roles, tokens, temperature, streaming.
> Set up API key, make first LLM call, understand request/response structure.
---
## Progress Log
| Date | What Was Completed |
|------|--------------------|
| Day 1 | NumPy — arrays, indexing, slicing, stats, boolean masking, matrix ops, normalization |
| Day 2 | Pandas — Series, DataFrames, read_csv, loc/iloc, missing data, groupby, merge, NumPy bridge |
---
*Last updated: Phase 1, Day 2 complete*
*This file should be updated after every major session or completed topic.*
