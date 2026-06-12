# import chromadb

# chroma_client = chromadb.Client()

# collection = chroma_client.create_collection(name="PhaseOne")

# collection.add(
#     ids=["id1", "id2", "id3"],
#     documents=[
#         "I love playing football",
#         "I enjoy watching cricket",
#         "Machine learning is fascinating"
#     ]
# )

# results = collection.query(
#     query_texts=["I like sports"],  # ← your question
#     n_results=2                     # ← how many results to return
# )
# for i, doc in enumerate(results['documents'][0]):
#     print(f"Rank {i+1}: {doc}  (distance: {results['distances'][0][i]:.3f})")
import chromadb

# This saves data to a folder called 'chroma_store' in your current directory
chroma_client = chromadb.PersistentClient(path="chroma_store")

collection = chroma_client.get_or_create_collection(name="PhaseOne")

collection.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "I love playing football",
        "I enjoy watching cricket",
        "Machine learning is fascinating"
    ]
)

results = collection.query(
    query_texts=["I like sports"],
    n_results=2,
    # where_document={'$contains':'cricket'} #this is when you need to filter out 
)

for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}  (distance: {results['distances'][0][i]:.3f})")