from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat",
    "A dog rested on the rug",
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "I love pizza"
]

embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")
print(f"Each sentence becomes a vector of {embeddings.shape[1]} numbers")
print(f"\nFirst embedding (first 5 numbers):\n{embeddings[0][:5]}")
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print("\nSimilarity matrix:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            score = similarities[i][j]
            print(f"{score:.2f} | {sent1[:30]} <-> {sent2[:30]}")