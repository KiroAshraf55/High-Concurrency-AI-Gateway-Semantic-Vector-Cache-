import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# Load the model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Select the ID-Mapped Index and the dictionary
dimension = 384
base_index = faiss.IndexFlatIP(dimension)  # Using Inner Product (IP) for cosine similarity
index = faiss.IndexIDMap(base_index)  # Create an ID-mapped index
cache_payload = {}  # Dictionary to store the mapping of IDs to questions

texts = ["I forgot my password", "where is my refund?"]
vectors = model.encode(texts)
faiss.normalize_L2(vectors)

# Add the vectors to the index with unique IDs
dynamic_ids = []  # Unique IDs for each question
for i in range(len(texts)):
    # Generate a unique ID for each question (e.g., using a timestamp)
    # + i to ensure uniqueness even if the timestamp is the same
    unique_id = int(time.time() * 1000) + i
    dynamic_ids.append(unique_id)

# convert the list of dynamic IDs to a numpy array of type int64
ids_array = np.array(dynamic_ids, dtype=np.int64)

# Add the vectors to the index with their corresponding IDs
index.add_with_ids(vectors, ids_array)

cache_payload[dynamic_ids[0]] = texts[0]
cache_payload[dynamic_ids[1]] = texts[1]

# Print the contents of the cache
print("\n------ Insertion Complete ------")
print(f"total items in cache: {index.ntotal}")
print("Dictionary Contents:")
for id, value in cache_payload.items():
    print(f"ID: {id}, Text: '{value}'")

# delete the first question from the cache
target_ID = dynamic_ids[0]  # The ID of the first question
print(f"\nAttempting to delete the text with ID: {target_ID}")

index.remove_ids(np.array([target_ID], dtype=np.int64))  # Remove the vector from the index
del cache_payload[target_ID]  # Remove the entry from the dictionary

print("After deletion:")
print(f"total items in cache: {index.ntotal}")
print(f"Remaining dictionary contents: {cache_payload}")