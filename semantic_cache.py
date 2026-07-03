import os
import time
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, threshold=0.8, max_size=1000, storage_dir: str = "cache_storage"):
        """
        Initializes the production-grade local semantic cache
        """
        print("Initializing Semantic Cache...")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.dimension = 384  # Dimension of the embedding vector
        
        self.threshold = threshold  # Similarity threshold for cache hits
        self.max_size = max_size  # Maximum size of the cache

        self.base_index = faiss.IndexFlatIP(self.dimension)  # Using Inner Product (IP) for cosine similarity
        self.index = faiss.IndexIDMap(self.base_index)  # Create an ID-mapped index
        self.cache_payload = {}  # Dictionary to store the mapping of IDs to questions

        # Set-up paths for hard drive storage
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")
        self.payload_path = os.path.join(storage_dir, "payloads.pkl")

        # Ensure the storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load existing cache from disk if available
        if os.path.exists(self.index_path) and os.path.exists(self.payload_path):
            self.load_from_disk()
        else:
            print("No existing cache found. Starting fresh.")
            self.base_index = faiss.IndexFlatIP(self.dimension)  # Using Inner Product (IP) for cosine similarity
            self.index = faiss.IndexIDMap(self.base_index)  # Create an ID-mapped index
            self.cache_payload = {}  # Dictionary to store the mapping of IDs to questions


    def _get_embedding(self, text: str | list[str]) -> np.ndarray:
        """
            Internal helper method to get the embedding of a text.
        """
        # if it is a string, convert it to a list of one string
        if isinstance(text, str):
            text = [text]
        vectors = self.model.encode(text)
        faiss.normalize_L2(vectors)

        return vectors
    
    def query(self, prompt:str) -> dict|None:
        """
        Checks the cache for a similar question.
        Returns the cached payload dict if found (Cache Hit), or None if not (Cache Miss).
        """
        if self.index.ntotal == 0:
            return None  # Cache is empty, so it's a miss
        
        vector = self._get_embedding(prompt)
        k = 1  # number of nearest neighbors to retrieve

        distances, indicies = self.index.search(vector, k)

        score = distances[0][0]
        matched_id = indicies[0][0]

        if score >= self.threshold:
            print(f"[cache hit] Found a similar question in the cache with score: {score:.4f}")
            return self.cache_payload[matched_id]
        
        print(f"[cache miss] No similar question found in the cache. Highest score: {score:.4f}")
        return None
    
    def insert(self, prompt:str, response:str):
        """Insert a prompt and its response into the cache."""
        if len(self.cache_payload) >= self.max_size:
            self.evict_oldest()
        
        vector = self._get_embedding(prompt)
        
        # Generate a unique ID for the new entry (using a timestamp)
        unique_id = time.time_ns()  # Use nanoseconds for better uniqueness
        while unique_id in self.cache_payload:
            unique_id += 1  # Increment to ensure uniqueness
        
        ids_array = np.array([unique_id], dtype=np.int64)
        self.index.add_with_ids(vector, ids_array)

        self.cache_payload[unique_id] = {"prompt": prompt, "response": response, "timestamp": unique_id}
        print(f"[cache insert] Added new entry to cache with ID: {unique_id}")

        self.save_to_disk()  # Save the updated cache to disk after insertion

    def evict_oldest(self):
        if not self.cache_payload:
            return
        
        # next(iter()) instantly grabs the very first key in the dictionary (The oldest!)
        # This is an O(1) operation. No looping required.
        oldest_id = next(iter(self.cache_payload))

        print(f"[cache eviction] Evicting oldest entry with ID: {oldest_id}")
        self.index.remove_ids(np.array([oldest_id], dtype=np.int64))
        del self.cache_payload[oldest_id]

    def save_to_disk(self):
        """Saves the current state of the cache to the hard drive."""
        try:
            # Save the FAISS maths matrix
            faiss.write_index(self.index, self.index_path)

            # Save the payload dictionary
            with open(self.payload_path, 'wb') as f:
                pickle.dump(self.cache_payload, f)
            print(f"[cache save] Cache saved to disk at {self.storage_dir}")
        except Exception as e:
            print(f"[cache save error] Failed to save cache to disk: {e}")

    def load_from_disk(self):
        """Loads the cache state from the hard drive."""
        try:
            print(f"[cache load] Loading cache from disk at {self.storage_dir}...")
            # Load the FAISS maths matrix
            self.index = faiss.read_index(self.index_path)

            # Load the payload dictionary
            with open(self.payload_path, 'rb') as f:
                self.cache_payload = pickle.load(f)
            print(f"[cache load] Success! Cache loaded from disk at {self.storage_dir}")
        except Exception as e:
            print(f"[cache load error] Failed to load cache from disk: {e}")
