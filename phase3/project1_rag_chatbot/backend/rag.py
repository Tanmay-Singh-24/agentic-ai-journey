"""
PaperMind — RAG backend.

A conversational Retrieval-Augmented Generation pipeline over 5 landmark AI
papers, built with LangGraph. The frontend calls one function: get_response().

High-level flow (one user turn):

    question ─► [retrieve] ─► [generate] ─► answer (grounded + cited)
                    │             │
       top-5 chunks from      LLM answers using ONLY those chunks,
       ChromaDB by meaning    plus the running conversation history

Memory is handled natively by LangGraph's checkpointer (keyed by thread_id),
so follow-up questions keep their context without any manual history plumbing.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# Document loading + splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings + vector store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# LLM + message types
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ── PATHS ─────────────────────────────────────────────────────────────────────
# Everything is resolved relative to THIS file so the app runs regardless of the
# working directory it's launched from.
HERE = os.path.dirname(os.path.abspath(__file__))          # .../project1_rag_chatbot/backend
PROJECT_DIR = os.path.dirname(HERE)                        # .../project1_rag_chatbot
PAPERS_DIR = os.path.join(PROJECT_DIR, "papers")           # the 5 source PDFs
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_store")     # persisted vector store
# The shared API key lives in phase1/.env (repo root is two levels above PROJECT_DIR:
# project1_rag_chatbot -> phase3 -> repo root).
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_DIR))
load_dotenv(dotenv_path=os.path.join(REPO_ROOT, "phase1", ".env"))   # sets GROQ_API_KEY


# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"        # small, fast, 384-dim sentence embeddings
LLM_MODEL = "llama-3.3-70b-versatile"        # Groq-hosted; strong enough for grounded QA
CHUNK_SIZE = 1000                            # characters per chunk
CHUNK_OVERLAP = 100                          # shared chars between neighbours (keeps context)
TOP_K = 5                                    # how many chunks to retrieve per question
COLLECTION_NAME = "papermind"                # named so build + load always hit the same store

# Map each PDF (by its arXiv id prefix) to a human-readable title. The filenames
# are arXiv ids like "1706.03762v7.pdf", which would be useless as a citation —
# we want answers to cite "Attention Is All You Need" instead.
PAPER_TITLES = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers",
    "2005.14165": "GPT-3: Language Models are Few-Shot Learners",
    "2203.02155": "InstructGPT: Training Language Models to Follow Instructions",
    "2210.03629": "ReAct: Synergizing Reasoning and Acting in Language Models",
}


def paper_title_for(filename):
    """Turn a PDF filename into its readable paper title (for citations).

    Falls back to the raw filename if the id isn't in our map, so an unexpected
    PDF never breaks ingestion.
    """
    for arxiv_id, title in PAPER_TITLES.items():
        if filename.startswith(arxiv_id):
            return title
    return filename


# ── EMBEDDINGS (one shared instance) ──────────────────────────────────────────
# Loaded once at import. Used both to embed chunks at ingest time AND to embed
# the user's question at query time — they MUST be the same model so the vectors
# live in the same space and distances are meaningful.
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


# ── BUILD OR LOAD THE VECTOR STORE ────────────────────────────────────────────
def build_or_load_vectorstore():
    """Return a Chroma vector store, embedding the PDFs only on the first run.

    If the persisted store already exists on disk, we just open it (fast).
    Otherwise we do the one-time work: load PDFs, chunk, tag with the source
    paper, embed, and persist. Re-embedding 5 papers on every startup would be
    slow and wasteful, so we guard it behind this existence check.
    """
    if os.path.exists(CHROMA_DIR):
        # Already ingested on a previous run — just reconnect to the saved store.
        print("Loading existing vector store from disk...")
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    # First run: ingest the papers from scratch.
    print("No vector store found — building it (one-time embedding)...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for filename in sorted(os.listdir(PAPERS_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(PAPERS_DIR, filename)
        title = paper_title_for(filename)

        # PyPDFLoader yields one Document per page (with page-number metadata).
        pages = PyPDFLoader(path).load()
        # Split those pages into ~1000-char chunks.
        chunks = splitter.split_documents(pages)
        # Tag every chunk with its source paper so the LLM can cite it later.
        for chunk in chunks:
            chunk.metadata["paper"] = title
        all_chunks.extend(chunks)
        print(f"  {title}: {len(chunks)} chunks")

    # Embed all chunks and write them to disk in one shot.
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Built vector store with {len(all_chunks)} total chunks.")
    return vectorstore


vectorstore = build_or_load_vectorstore()


# ── LLM ───────────────────────────────────────────────────────────────────────
# temperature=0 keeps answers deterministic and faithful to the retrieved text —
# we don't want creativity here, we want grounded accuracy.
llm = ChatGroq(model=LLM_MODEL, temperature=0)

SYSTEM_PROMPT = (
    "You are PaperMind, an assistant that helps people understand landmark AI "
    "research papers. Answer using ONLY the provided context from the papers. "
    "If the answer isn't in the context, say so honestly rather than guessing. "
    "When you answer, mention which paper the information comes from."
)


# ── GRAPH STATE ───────────────────────────────────────────────────────────────
# The data carried through the graph for one turn:
#   messages → the running conversation (add_messages APPENDS, so history grows)
#   context  → the retrieved chunks for THIS question, formatted as text
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


# ── NODE 1: RETRIEVE ──────────────────────────────────────────────────────────
def retrieve(state: State) -> dict:
    """Embed the latest user question and fetch the top-5 relevant chunks.

    We retrieve on the most recent question (state messages end with it). Each
    chunk is prefixed with its source paper so the generate step — and thus the
    final answer — can attribute facts correctly.
    """
    question = state["messages"][-1].content          # the newest user turn
    docs = vectorstore.similarity_search(question, k=TOP_K)

    # Format chunks into one context block, labelling each with its paper.
    blocks = []
    for doc in docs:
        paper = doc.metadata.get("paper", "Unknown paper")
        blocks.append(f"[Source: {paper}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(blocks)

    return {"context": context}                        # write context into state


# ── NODE 2: GENERATE ──────────────────────────────────────────────────────────
def generate(state: State) -> dict:
    """Ask the LLM to answer using the retrieved context + conversation history.

    We prepend a SystemMessage (instructions + this turn's context) to the full
    message history. Passing the history is what makes follow-up questions work —
    the model can resolve "it" / "that paper" from earlier turns.
    """
    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context from the papers:\n{state['context']}"
    )
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}                    # append the AI reply to history


# ── BUILD + COMPILE THE GRAPH ─────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")        # every turn starts by retrieving
builder.add_edge("retrieve", "generate")   # then answers from what it found
builder.add_edge("generate", END)

# InMemorySaver persists each thread's State between calls, keyed by thread_id —
# this is the native LangGraph replacement for RunnableWithMessageHistory.
graph = builder.compile(checkpointer=InMemorySaver())


# ── PUBLIC API ────────────────────────────────────────────────────────────────
def get_response(question, thread_id):
    """Answer a question, remembering everything said under this thread_id.

    The frontend calls this once per user message. We pass only the new question;
    the checkpointer supplies the prior history automatically.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
    return result["messages"][-1].content              # the latest AI message text


# Quick manual smoke test: `python backend/rag.py`
if __name__ == "__main__":
    print("\n--- smoke test ---")
    print(get_response("What problem does the Transformer architecture solve?", "test"))
