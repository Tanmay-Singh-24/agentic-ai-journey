import chromadb

# ── 1. PERSISTENT CLIENT ──────────────────────────────────────────────────────
# PersistentClient saves data to disk — survives after script ends
# Regular Client() stores in memory — gone when script ends
client = chromadb.PersistentClient(path="chroma_store")


# ── 2. COLLECTION ─────────────────────────────────────────────────────────────
# A collection is a named container for your documents
# get_or_create_collection — creates if doesn't exist, opens if it does
collection = client.get_or_create_collection(name="knowledge_base")


# ── 3. UPSERT DOCUMENTS ───────────────────────────────────────────────────────
# upsert = update + insert
# - ID already exists → updates it
# - ID doesn't exist → adds it
# - Never throws an error (safer than add())
# Each document also gets metadata — extra info you can filter by later
collection.upsert(
    ids=["id1", "id2", "id3", "id4", "id5"],
    documents=[
        "I love playing football on weekends",
        "I enjoy watching cricket matches",
        "Machine learning is fascinating and powerful",
        "Deep learning uses neural networks",
        "Mango is the best tropical fruit"
    ],
    metadatas=[
        {"category": "sports",  "topic": "football"},
        {"category": "sports",  "topic": "cricket"},
        {"category": "tech",    "topic": "ml"},
        {"category": "tech",    "topic": "dl"},
        {"category": "food",    "topic": "fruits"}
    ]
)

print(f"Total documents in collection: {collection.count()}")
print()


# ── 4. BASIC QUERY ────────────────────────────────────────────────────────────
# Converts query text to a vector, finds closest documents by distance
# Lower distance = more similar
print("=== BASIC QUERY: 'I like sports' ===")
results = collection.query(
    query_texts=["I like sports"],
    n_results=2
)

for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}  (distance: {results['distances'][0][i]:.3f})")
print()


# ── 5. QUERY WITH METADATA FILTER ────────────────────────────────────────────
# where= filters BEFORE the similarity search
# Only searches within documents that match the filter
print("=== QUERY WITH FILTER: only 'tech' category ===")
results_filtered = collection.query(
    query_texts=["I want to learn AI"],
    n_results=2,
    where={"category": "tech"}   # only search within tech documents
)

for i, doc in enumerate(results_filtered['documents'][0]):
    print(f"Rank {i+1}: {doc}  (distance: {results_filtered['distances'][0][i]:.3f})")
print()


# ── 6. FETCH A SPECIFIC DOCUMENT BY ID ───────────────────────────────────────
# Use .get() when you know exactly which document you want
print("=== FETCH BY ID ===")
doc = collection.get(ids=["id1"])
print(f"Document: {doc['documents'][0]}")
print(f"Metadata: {doc['metadatas'][0]}")
print()


# ── 7. DELETE A DOCUMENT ─────────────────────────────────────────────────────
# Remove a document by its ID
collection.delete(ids=["id5"])
print(f"After deleting id5, total documents: {collection.count()}")
print()
u

# ── 8. UPSERT TO UPDATE EXISTING DATA ────────────────────────────────────────
# Update an existing document — same ID, new content
collection.upsert(
    ids=["id1"],
    documents=["I love playing football and basketball"],
    metadatas=[{"category": "sports", "topic": "football"}]
)

updated = collection.get(ids=["id1"])
print(f"=== UPDATED DOCUMENT ===")
print(f"id1 is now: {updated['documents'][0]}")
print()


# ── 9. METADATA FILTERS IN DEPTH ─────────────────────────────────────────────
# Create a separate collection for movies to demonstrate filters cleanly
movies = client.get_or_create_collection(name="movies")

movies.upsert(
    ids=["m1", "m2", "m3", "m4", "m5", "m6"],
    documents=[
        "A boy discovers he is a wizard and goes to magic school",
        "A wizard destroys a powerful ring to save the world",
        "Two scientists create a dinosaur theme park",
        "A shark terrorizes a small beach town",
        "Toys come to life when humans are not around",
        "A lion cub grows up to reclaim his kingdom"
    ],
    metadatas=[
        {"genre": "fantasy",   "year": 2001, "rating": 8},
        {"genre": "fantasy",   "year": 2003, "rating": 9},
        {"genre": "scifi",     "year": 1993, "rating": 8},
        {"genre": "thriller",  "year": 1975, "rating": 8},
        {"genre": "animation", "year": 1995, "rating": 9},
        {"genre": "animation", "year": 1994, "rating": 9}
    ]
)

# NO FILTER — searches all 6 movies
print("=== NO FILTER ===")
results = movies.query(query_texts=["magic and adventure"], n_results=2)
for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}")
print()

# EXACT MATCH FILTER — only fantasy movies
print("=== ONLY FANTASY ===")
results = movies.query(
    query_texts=["magic and adventure"],
    n_results=2,
    where={"genre": "fantasy"}
)
for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}")
print()

# NUMERIC EXACT MATCH — rating exactly 9
print("=== RATING EXACTLY 9 ===")
results = movies.query(
    query_texts=["magic and adventure"],
    n_results=2,
    where={"rating": 9}
)
for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}")
print()

# OPERATOR FILTER — year >= 2000
# $gte = greater than or equal
# $lte = less than or equal
# $ne  = not equal
# $gt  = greater than
# $lt  = less than
print("=== YEAR AFTER 2000 ($gte) ===")
results = movies.query(
    query_texts=["magic and adventure"],
    n_results=2,
    where={"year": {"$gte": 2000}}
)
for i, doc in enumerate(results['documents'][0]):
    print(f"Rank {i+1}: {doc}")
