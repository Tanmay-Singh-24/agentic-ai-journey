"""
LangChain RAG — the SAME pipeline as phase1/manual_rag/rag_pipeline.py,
rebuilt with LangChain so you can see exactly what the framework does for you.

Each section below names the manual function it replaces. Read it as a
side-by-side: "we hand-wrote X before → LangChain gives us Y now."

Manual pipeline (phase1)            LangChain equivalent (this file)
─────────────────────────           ────────────────────────────────
load_document()                  →  TextLoader
chunk_text()                     →  RecursiveCharacterTextSplitter
embed() + build_store()          →  HuggingFaceEmbeddings + Chroma.from_documents()
retrieve()                       →  vectorstore.as_retriever()
build_prompt()                   →  ChatPromptTemplate
generate_answer() (raw Groq)     →  ChatGroq
ask() (manual orchestration)     →  an LCEL chain built with the | operator

Run:  python langchain_rag.py
"""

import os                                  # for building file paths
from dotenv import load_dotenv             # loads GROQ_API_KEY from the .env file

# ── IMPORTS ───────────────────────────────────────────────────────────────────
# NOTE on versions: this project is on LangChain 1.x. Your original prompt asked
# for `from langchain_community.vectorstores import Chroma`, but in 1.x that path
# is deprecated. The current, maintained import is `langchain_chroma.Chroma`.
# (TextLoader still lives in langchain_community and only triggers a "community is
# being sunset" warning — harmless for learning.)
from langchain_community.document_loaders import TextLoader        # Step 1
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Step 2
from langchain_huggingface import HuggingFaceEmbeddings            # Step 3 (embeddings)
from langchain_chroma import Chroma                                # Step 3 (vector store)
from langchain_core.prompts import ChatPromptTemplate             # Step 5
from langchain_groq import ChatGroq                               # Step 6
from langchain_core.output_parsers import StrOutputParser         # Step 7
from langchain_core.runnables import RunnablePassthrough          # Step 7


# ── PATHS ───────────────────────────────────────────────────────────────────--
# We build paths relative to THIS file (not the current working directory) so
# the script runs correctly no matter which folder you launch it from.
HERE = os.path.dirname(os.path.abspath(__file__))   # .../agentic-ai-journey/phase2
ROOT = os.path.dirname(HERE)                         # .../agentic-ai-journey
ENV_PATH = os.path.join(ROOT, "phase1", ".env")
DOC_PATH = os.path.join(ROOT, "phase1", "manual_rag", "knowledge.txt")

# load_dotenv reads the .env file and puts GROQ_API_KEY into the environment.
# ChatGroq (Step 6) picks it up automatically from there.
load_dotenv(dotenv_path=ENV_PATH)


# ── STEP 1 — LOAD THE DOCUMENT ────────────────────────────────────────────────
# Manual equivalent: load_document() — which just did open(path).read() → a str.
#
# A LangChain `Document` is NOT a plain string. It is an object with two fields:
#   .page_content  → the actual text (str)
#   .metadata      → a dict (e.g. {"source": "knowledge.txt"})
# Every step downstream (splitter, vector store, retriever) speaks "Document",
# so loaders wrap raw text in this object to carry metadata along the pipeline.
loader = TextLoader(DOC_PATH)        # create a loader pointed at our text file
documents = loader.load()            # returns a LIST of Document objects (here: 1)
print(f"Step 1 — loaded {len(documents)} document(s)")
print(f"          metadata: {documents[0].metadata}")


# ── STEP 2 — CHUNK THE DOCUMENT ───────────────────────────────────────────────
# Manual equivalent: chunk_text() — we split on whitespace into fixed word
# windows with overlap.
#
# RecursiveCharacterTextSplitter is smarter: it tries a PRIORITISED list of
# separators — paragraphs ("\n\n"), then lines ("\n"), then sentences/spaces —
# and only falls back to a cruder split if a piece is still too big. So it
# prefers to break at natural boundaries instead of mid-sentence.
#
# chunk_size=200 is measured in CHARACTERS here (our manual version counted
# WORDS — different unit, same idea). chunk_overlap=20 means each chunk repeats
# the last 20 characters of the previous one, so a fact sitting on a boundary
# isn't lost to both neighbouring chunks.
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
chunks = splitter.split_documents(documents)   # list[Document] → more, smaller Documents
print(f"\nStep 2 — split into {len(chunks)} chunks")
print(f"          first chunk: {chunks[0].page_content[:60]!r}...")


# ── STEP 3 — EMBED AND STORE IN CHROMADB ──────────────────────────────────────
# Manual equivalent: embed() (SentenceTransformer.encode) + build_store()
# (chromadb.add with our own vectors).
#
# HuggingFaceEmbeddings wraps the SAME model we used manually: all-MiniLM-L6-v2.
# We deliberately reuse it so vectors land in the same 384-dim space and the
# comparison is fair. The difference: before, WE called .encode() and passed raw
# vectors to ChromaDB. Now LangChain owns that step — Chroma.from_documents()
# calls embeddings.embed_documents() on each chunk internally and stores the
# result. The embedding still happens; LangChain just hides the wiring.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,        # the chunked Documents
    embedding=embeddings,    # the model used to turn each chunk into a vector
)                            # in-memory (no persist_directory) → rebuilt each run
print(f"\nStep 3 — stored {vectorstore._collection.count()} vectors in Chroma")


# ── STEP 4 — CREATE A RETRIEVER ───────────────────────────────────────────────
# Manual equivalent: retrieve() — embed the question, call collection.query(),
# pull the top-k documents out of the result dict.
#
# A retriever is a thin, standard interface that wraps all of that. You give it
# a question STRING; it embeds the question (same model), runs the similarity
# search, and hands back a list[Document]. k=3 = "return the 3 closest chunks",
# exactly like n_results=3 in the manual version.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("\nStep 4 — retriever ready (k=3)")


# ── STEP 5 — BUILD THE PROMPT TEMPLATE ────────────────────────────────────────
# Manual equivalent: build_prompt() — we f-string'd the context and question
# into one big text blob by hand.
#
# A prompt template is a reusable blueprint with {placeholders}. We define the
# shape ONCE; later the chain fills {context} and {question} per call. Using
# variables (not hardcoded text) means the same template serves every question,
# and the structure stays readable. from_messages also lets us cleanly separate
# the system instruction from the user's turn — the proper chat format.
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context "
     "provided. If the answer is not in the context, say you don't know."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}"),
])
print("\nStep 5 — prompt template built")


# ── STEP 6 — INITIALISE THE LLM ───────────────────────────────────────────────
# Manual equivalent: generate_answer() — a raw groq.Groq().chat.completions.create
# call where we built the messages list ourselves.
#
# ChatGroq is LangChain's wrapper around that same Groq endpoint. It reads
# GROQ_API_KEY from the environment (loaded above) and, crucially, plugs into the
# chain: it accepts the prompt template's output and returns a message object —
# no manual messages=[...] assembly needed. temperature=0.2 keeps it factual.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
print("Step 6 — Groq LLM initialised")


# ── STEP 7 — BUILD THE FULL RAG CHAIN (LCEL) ──────────────────────────────────
# Manual equivalent: the ask() function, which called retrieve() → build_prompt()
# → generate_answer() in sequence by hand.
#
# format_docs takes the retriever's list[Document] and flattens it into one
# string (the same "\n\n".join we did manually in build_prompt).
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL (LangChain Expression Language) uses the pipe operator | to wire steps
# together — output of the left feeds the input of the right, like a Unix pipe.
#
# Reading the chain top to bottom:
#
#   {"context": retriever | format_docs, "question": RunnablePassthrough()}
#       The chain is invoked with a plain question STRING. This dict builds the
#       two values the prompt template needs:
#         • "context"  → send the question into the retriever (gets top-3 docs),
#                        then pipe those docs through format_docs → one string.
#         • "question" → RunnablePassthrough() forwards the original question
#                        UNCHANGED. (Without it, we'd have nothing to put in the
#                        {question} slot — the input would be lost.)
#   | prompt            Fill {context} and {question} into the template.
#   | llm               Send the finished prompt to Groq → a message object.
#   | StrOutputParser() Pull the plain .content text out of that message object,
#                       so the chain returns a clean string instead of an object.
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Step 7 — RAG chain assembled\n")


# ── STEP 8 — RUN TEST QUESTIONS ───────────────────────────────────────────────
# Same questions as the manual pipeline so you can diff the answers directly.
# The last one is NOT in the document → the model should say it doesn't know.
questions = [
    "What is the flight time of the Zephyr X1?",
    "How long is the battery warranty?",
    "What software is used to plan flights?",
    "Does the drone work without GPS?",
    "What is the price of the Zephyr X1?",   # not in the doc → expect "don't know"
]

for q in questions:
    answer = chain.invoke(q)     # one call runs the WHOLE pipeline end to end
    print(f"Q: {q}")
    print(f"A: {answer}\n" + "-" * 80)
