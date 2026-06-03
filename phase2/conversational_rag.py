"""
Conversational RAG — extends phase2/langchain_rag.py with MEMORY.

langchain_rag.py answered each question in isolation: it had no idea what you
asked before. This file adds chat history so follow-up questions like
"what about in windy conditions?" work — the pipeline knows "what" refers to
the flight time you just asked about.

WHAT'S NEW vs langchain_rag.py (and vs the manual phase1 pipeline)
──────────────────────────────────────────────────────────────────
The manual pipeline (phase1/manual_rag/rag_pipeline.py) was STATELESS — ask()
took one question, retrieved, answered, forgot everything. So most of what's
below has NO manual equivalent; it's the machinery memory requires:

  load / chunk / embed / retriever      → same as langchain_rag.py (unchanged idea)
  build_prompt() + generate_answer()    → create_stuff_documents_chain
  ask() orchestration                   → create_retrieval_chain
  (nothing — manual had no memory)      → create_history_aware_retriever  [NEW]
  (nothing — manual had no memory)      → RunnableWithMessageHistory       [NEW]
  (nothing — manual had no memory)      → InMemoryChatMessageHistory       [NEW]

THE CORE PROBLEM memory creates for RAG:
A follow-up like "what about in windy conditions?" is useless to a retriever on
its own — there are no keywords to match in the vector store. So we add a FIRST
LLM step that rewrites it, using the chat history, into a standalone question
("what is the flight time of the Zephyr X1 in windy conditions?") BEFORE
retrieval. That rewrite is exactly what create_history_aware_retriever does.

Run:  python conversational_rag.py
"""

import os
from dotenv import load_dotenv

# ── IMPORTS ───────────────────────────────────────────────────────────────────
# Same base pieces as langchain_rag.py:
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
# ChatPromptTemplate as before, plus MessagesPlaceholder — a SLOT in the prompt
# where a whole list of past messages (the chat history) gets injected.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The classic "chain constructor" helpers. In LangChain 1.x these moved out of
# `langchain.chains` (which no longer exists) into the `langchain_classic` package.
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Memory plumbing:
# RunnableWithMessageHistory  — wraps a chain and auto-saves/loads history per session.
# InMemoryChatMessageHistory  — a simple in-RAM list of messages for one session.
#   (langchain_community also has ChatMessageHistory, but it's the same thing and
#    triggers the "community is sunset" warning, so we use the core version.)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


# ── PATHS + ENV ───────────────────────────────────────────────────────────────
# Paths relative to THIS file so it runs from any directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV_PATH = os.path.join(ROOT, "phase1", ".env")
DOC_PATH = os.path.join(ROOT, "phase1", "manual_rag", "knowledge.txt")
load_dotenv(dotenv_path=ENV_PATH)        # puts GROQ_API_KEY into the environment


# ── STEPS 1-4 — INDEX THE DOCUMENT (identical idea to langchain_rag.py) ────────
# Load → chunk → embed → store → retriever. Unchanged from the non-conversational
# version except chunk_size=600/overlap=50 (the size that retrieves whole facts).
documents = TextLoader(DOC_PATH).load()                                  # Step 1: load
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
chunks = splitter.split_documents(documents)                             # Step 2: chunk
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")        # Step 3a: embedder
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)  # Step 3b: store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})             # Step 4: retriever
print(f"Indexed {len(chunks)} chunks.\n")

# The LLM, shared by BOTH llm-using steps below (the rewriter and the answerer).
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


# ── STEP 5 — HISTORY-AWARE RETRIEVER (the NEW piece) ──────────────────────────
# NO manual equivalent — this exists only because we now have memory.
#
# Problem: "what about in windy conditions?" has nothing for the vector store to
# match. Solution: before retrieving, ask the LLM to rewrite the follow-up into a
# standalone question using the chat history. THEN retrieve with that rewrite.
#
# This prompt instructs the LLM to ONLY reformulate, never answer. MessagesPlaceholder
# injects the running chat history; {input} is the new (possibly vague) question.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference "
     "context in the chat history, formulate a standalone question which can be "
     "understood without the chat history. Do NOT answer the question — just "
     "reformulate it if needed, otherwise return it unchanged."),
    MessagesPlaceholder("chat_history"),   # slot: all prior user/assistant turns
    ("human", "{input}"),                  # slot: the new question
])

# create_history_aware_retriever bundles: (rewrite question with llm) → (retrieve).
# If chat_history is empty it skips the rewrite and just retrieves directly.
history_aware_retriever = create_history_aware_retriever(
    llm,                       # does the rewriting
    retriever,                 # does the actual vector search on the rewrite
    contextualize_q_prompt,    # tells the llm how to rewrite
)


# ── STEP 6 — ANSWER CHAIN ─────────────────────────────────────────────────────
# Manual equivalent: build_prompt() + generate_answer() combined.
#
# create_stuff_documents_chain "stuffs" the retrieved docs into the {context}
# slot of a prompt, then calls the LLM. ("stuff" = the strategy of dumping all
# retrieved docs straight into one prompt — fine for small k.)
#
# Note this QA prompt ALSO carries chat_history, so the final answer can use the
# conversation (e.g. resolve "that battery") not just the retrieved chunks.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context "
     "below. If the answer is not in the context, say you don't know.\n\n"
     "{context}"),                         # retrieved docs get inserted here
    MessagesPlaceholder("chat_history"),   # prior turns, for resolving references
    ("human", "{input}"),                  # the current question
])

# Builds the docs → prompt → llm → text answer sub-chain.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# ── STEP 7 — TIE THEM TOGETHER ────────────────────────────────────────────────
# Manual equivalent: the ask() orchestration (retrieve → prompt → answer), but now
# the "retrieve" half is the history-aware retriever.
#
# create_retrieval_chain wires: history_aware_retriever (gets docs) → feeds them as
# {context} into question_answer_chain (writes the answer). Input dict needs keys
# "input" and "chat_history"; output dict includes "answer" (and "context").
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ── STEP 8 — ADD AUTOMATIC SESSION MEMORY ─────────────────────────────────────
# NO manual equivalent — the manual pipeline never stored a single past message.
#
# Right now rag_chain expects us to pass chat_history by hand every call. We don't
# want to manage that manually, so we wrap it. RunnableWithMessageHistory:
#   • before each call: loads that session's history and fills the chat_history slot
#   • after each call: appends the new question + answer to that session's history
#
# It finds a session's history via get_session_history(session_id). We keep a dict
# of session_id → history object, creating one on first use. (Different session_id
# = a separate, isolated conversation.)
store = {}

def get_session_history(session_id):
    # Return the history for this session, creating an empty one if it's new.
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",        # which input key holds the user's question
    history_messages_key="chat_history",  # which prompt slot to inject history into
    output_messages_key="answer",      # which output key holds the reply to save
)


# ── STEP 9 — DEMO CONVERSATION (follow-ups that need memory) ──────────────────
def ask(question, session_id="demo"):
    # session_id routes to a specific conversation's memory. Same id = same thread.
    result = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n" + "-" * 80)

if __name__ == "__main__":
    # Each question after the first is a FOLLOW-UP that's meaningless without the
    # earlier turns — proving memory + question-rewriting are working.
    ask("What is the flight time of the Zephyr X1?")          # establishes topic: flight time
    ask("What about in windy conditions?")                    # "what" = flight time (needs history)
    ask("How long does its battery take to charge?")          # "its" = the Zephyr X1's
    ask("And how should I store it if I won't use it a while?")  # "it" = that battery
