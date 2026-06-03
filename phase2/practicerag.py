import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv(dotenv_path="../phase1/.env")
embed_model=SentenceTransformer("all-MiniLM-L6-v2")
groq_client=Groq()

text = """Tanmay is a 21 year old CS student.
He lives in Meerut.
He is preparing for placements at NVIDIA and OpenAI.
He drinks 3 cups of coffee every day.
He has a MacBook Air."""

chunks=text.split("\n")
print(f"chunks: {chunks}")

vectors = embed_model.encode(chunks).tolist()

client = chromadb.Client()
collection = client.create_collection("Tanmay")
collection.add(
    ids=[str(i+1) for i in range(len(chunks))],
    documents=chunks,
    embeddings=vectors
)

question="where does tanmay live?"
question_vector=embed_model.encode(question).tolist()

results = collection.query(query_embeddings=question_vector,n_results=2)
retrieved_chunks=results["documents"][0]
print(f"Retrieved: {retrieved_chunks}")

context = "\n".join(retrieved_chunks)
prompt = f"Answer using only this context:\n{context}\n\nQuestion: {question}\nAnswer:"

response=groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)