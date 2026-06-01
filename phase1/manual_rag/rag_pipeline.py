"""
Manual RAG pipeline — NO frameworks (no LangChain / LlamaIndex).

Every step is explicit so you can explain it in an interview:

    text document
        │  load
        ▼
    chunk            ← split into overlapping pieces
        │  embed (SentenceTransformer)        ← WE control this step
        ▼
    store (ChromaDB) ← ChromaDB is JUST a vector store here, it never embeds
        │
        ▼
    question ──embed (SAME model)──► query vectors ──► top-k chunks
                                                            │ build prompt
                                                            ▼
                                                          Groq LLM
                                                            │
                                                            ▼
                                                          answer

KEY IDEA: retrieval works because the question and the chunks are embedded
by the SAME model into the SAME vector space. Closeness in that space ≈
closeness in meaning. If you embedded them with different models, the
distances would be meaningless.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()


# ── 0. LOAD THE MODELS ONCE ───────────────────────────────────────────────────
# Loading the embedding model is slow (it reads weights from disk), so we do it
# a single time at module level and reuse it everywhere.
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim vectors
GROQ = Groq()                                            # reads GROQ_API_KEY from .env
LLM_MODEL = "llama-3.1-8b-instant"


# ── 1. LOAD DOCUMENT ──────────────────────────────────────────────────────────
def load_document(path):
    """Read a plain-text file into a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── 2. CHUNK ──────────────────────────────────────────────────────────────────
# Why chunk at all?
#   1. Embedding models have a token limit — a whole document won't fit.
#   2. Retrieval is more precise: we want the few sentences that answer the
#      question, not the entire document stuffed into the prompt.
#
# We chunk by WORDS with OVERLAP. Overlap means consecutive chunks share some
# words, so a sentence that lands on a chunk boundary isn't split in a way that
# destroys its meaning for both chunks.
def chunk_text(text, chunk_size=120, overlap=20):
    """
    Split text into chunks of `chunk_size` words, where each chunk shares
    `overlap` words with the previous one.

    Example with chunk_size=120, overlap=20:
        chunk 0 -> words   0..120
        chunk 1 -> words 100..220   (starts 20 words before chunk 0 ended)
        chunk 2 -> words 200..320
    """
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap          # how far we advance each iteration

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step

    return chunks


# ── 3. EMBED ──────────────────────────────────────────────────────────────────
# This is the step ChromaDB would normally hide from you. We do it ourselves.
# model.encode() turns each string into a 384-number vector (a numpy array).
# .tolist() converts numpy arrays -> plain Python lists, which is the format
# ChromaDB expects.
def embed(texts):
    """Embed a list of strings into a list of 384-dim vectors."""
    vectors = EMBED_MODEL.encode(texts)
    return vectors.tolist()


# ── 4. STORE ──────────────────────────────────────────────────────────────────
# We create a ChromaDB collection and push our OWN vectors into it via
# `embeddings=`. Because we never pass `documents=` to be auto-embedded,
# ChromaDB does zero embedding — it only stores vectors and runs the
# nearest-neighbour search. It is a pure vector database here.
#
# metadata={"hnsw:space": "cosine"} tells ChromaDB to rank by COSINE distance.
# Cosine compares the *direction* of vectors (ignoring length), which is the
# right choice for sentence embeddings. (ChromaDB's default is squared L2.)
def build_store(chunks, embeddings):
    """Create a fresh in-memory collection holding the chunk vectors."""
    client = chromadb.Client()                       # in-memory: rebuilt each run
    # Start clean so re-running the script doesn't stack duplicate data.
    if "rag_chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("rag_chunks")

    collection = client.create_collection(
        name="rag_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,            # stored so we can read the text back later
        embeddings=embeddings,       # OUR vectors — ChromaDB does not re-embed
    )
    return collection


# ── 5. RETRIEVE ───────────────────────────────────────────────────────────────
# Embed the question with the SAME model, then ask ChromaDB for the closest
# chunk vectors. Lower cosine distance = more semantically similar.
def retrieve(collection, question, n_results=3):
    """Return the top-n most relevant chunks for a question."""
    question_vector = embed([question])              # list with one 384-dim vector

    results = collection.query(
        query_embeddings=question_vector,            # query by VECTOR, not text
        n_results=n_results,
    )

    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(docs, distances))


# ── 6. BUILD THE AUGMENTED PROMPT ─────────────────────────────────────────────
# This is the "A" in RAG: we Augment the prompt with retrieved context.
# We instruct the model to answer ONLY from the context — this is what stops
# it from hallucinating and is the whole point of RAG.
def build_prompt(question, retrieved):
    context = "\n\n".join(f"[{i+1}] {doc}" for i, (doc, _) in enumerate(retrieved))
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ── 7. GENERATE ───────────────────────────────────────────────────────────────
def generate_answer(prompt):
    """Send the augmented prompt to Groq and return the text answer."""
    response = GROQ.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,             # low temp = factual, sticks to the context
    )
    return response.choices[0].message.content


# ── 8. ORCHESTRATE: ask() ties every step together ────────────────────────────
def ask(collection, question, n_results=3, show_context=True):
    retrieved = retrieve(collection, question, n_results)

    if show_context:
        print("  Retrieved chunks (lower distance = closer match):")
        for doc, dist in retrieved:
            preview = doc[:70].replace("\n", " ")
            print(f"    dist={dist:.3f} | {preview}...")
        print()

    prompt = build_prompt(question, retrieved)
    return generate_answer(prompt)


# ── 9. DEMO ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ---- Index the document (steps 1-4) ----
    print("Building the index...")
    text = load_document(os.path.join(os.path.dirname(__file__), "knowledge.txt"))
    chunks = chunk_text(text)
    print(f"  Document split into {len(chunks)} chunks.")
    embeddings = embed(chunks)
    print(f"  Embedded into {len(embeddings)} vectors of {len(embeddings[0])} dims each.")
    collection = build_store(chunks, embeddings)
    print(f"  Stored {collection.count()} chunks in ChromaDB.\n")

    # ---- Ask questions (steps 5-8) ----
    questions = [
        "What is the flight time of the Zephyr X1?",
        "How long is the battery warranty?",
        "What software is used to plan flights?",
        "Does the drone work without GPS?",
        "What is the price of the Zephyr X1?",   # NOT in the document — should say it doesn't know
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = ask(collection, q)
        print(f"A: {answer}\n" + "-" * 80 + "\n")
