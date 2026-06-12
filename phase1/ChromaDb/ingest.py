import chromadb

client = chromadb.PersistentClient(path="chromadb_basics/chroma_store")
collection = client.get_or_create_collection(name="PhaseOne")

if collection.count() == 0:
    collection.add(
        ids=["id1", "id2", "id3"],
        documents=[
            "I love playing football",
            "I enjoy watching cricket",
            "Machine learning is fascinating"
        ]
    )
    print(f"Added documents. Total: {collection.count()}")
else:
    print(f"Collection already has {collection.count()} documents. Skipping.")