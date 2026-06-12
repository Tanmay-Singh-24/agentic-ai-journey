import chromadb

client = chromadb.PersistentClient(path="chromadb_basics/chroma_store")
collection = client.get_or_create_collection(name="PhaseOne")

results = collection.query(
    query_texts=["I like sports"],
    n_results=2
)

for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}  (distance: {results['distances'][0][i]:.3f})")