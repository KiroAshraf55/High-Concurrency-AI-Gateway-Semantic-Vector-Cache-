import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# User A: The first question ever asked (to our cache)
text_a = ["How do I reset my password?"]
vector_a = model.encode(text_a)

# impotant maths rule for faiss:
# vectors must be normalized for the cosine similarity search to work properly
faiss.normalize_L2(vector_a)

# build the faiss vector database
dimension = vector_a.shape[1]  # dimension of the vector
index = faiss.IndexFlatIP(dimension)  # Using Inner Product (IP) for cosine similarity
index.add(vector_a)  # Add the vector to the database

# User B: A new question that is similar to User A's question
new_users_text = [
    "I forgot my password, how can I change it?",
    "What is the Capital of France?"
]

new_users_vectors = model.encode(new_users_text)
faiss.normalize_L2(new_users_vectors)

# Search for the most similar question in the cache
k = 1  # number of nearest neighbors to retrieve
distances, indices = index.search(new_users_vectors, k)

# Print the results
print("\nSearch results:")

# Results for User B
print(f"\nNew Question (User B): '{new_users_text[0]}'")
print(f"Closest match in Cache: '{text_a[indices[0][0]]}'")
print(f"Similarity Score: {distances[0][0]:.4f} (1.0 is a perfect match)")

# Results for User C
print(f"\nNew Question (User C): '{new_users_text[1]}'")
print(f"Closest match in Cache: '{text_a[indices[1][0]]}'")
print(f"Similarity Score: {distances[1][0]:.4f} (1.0 is a perfect match)")
