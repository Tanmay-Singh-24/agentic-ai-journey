"""
PaperMind — Streamlit chat UI.

Thin presentation layer: it renders the chat and forwards each question to the
backend's get_response(). All RAG logic lives in backend/rag.py.

Run from the project1_rag_chatbot/ folder:
    streamlit run frontend/app.py
"""

import os
import sys

import streamlit as st

# This file lives in frontend/. Add the PROJECT root (its parent) to the import
# path so `from backend.rag import ...` resolves no matter where streamlit is
# launched from.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from backend.rag import get_response, PAPER_TITLES   # noqa: E402 (after sys.path tweak)


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PaperMind", page_icon="📄")

st.title("📄 PaperMind")
st.caption("Ask questions about 5 landmark AI papers — answers are grounded in the papers themselves, with citations.")


# ── SIDEBAR: what the user can ask about ──────────────────────────────────────
# Listing the papers sets expectations: the bot only knows these five.
with st.sidebar:
    st.header("Papers in the library")
    for title in PAPER_TITLES.values():
        st.markdown(f"- {title}")
    st.divider()
    st.caption("Answers come only from these papers. If something isn't in them, PaperMind will say so.")


# ── CHAT HISTORY (for display) ────────────────────────────────────────────────
# session_state survives Streamlit's reruns. We keep our own list purely to
# RE-RENDER the visible transcript; the backend keeps the real memory via its
# checkpointer (thread_id below).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Repaint the whole conversation on every rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── CHAT INPUT + RESPONSE ─────────────────────────────────────────────────────
# A fixed thread_id ties every turn in this session to one backend memory thread.
THREAD_ID = "user_session"

if question := st.chat_input("Ask about the Transformer, BERT, GPT-3, InstructGPT, or ReAct..."):
    # Show + store the user's message.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get + show the assistant's grounded answer.
    with st.chat_message("assistant"):
        with st.spinner("Searching the papers..."):
            answer = get_response(question, thread_id=THREAD_ID)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
