import os
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class SemanticCache:
    def __init__(self, threshold=0.8, max_size=1000):
        """
        Initializes the distributed Qdrant vector database semantic cache
        """
        print("Initializing Distributed Qdrant Semantic Cache...")
        self.threshold = threshold
        self.max_size = max_size
        self.dimension = 384  
        
        # Connect to the Qdrant container over the secure Docker internal network
        self.client = QdrantClient(host="qdrant", port=6333)
        self.collection_name = "ai_cache"
        
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        try:
            if not self.client.collection_exists(self.collection_name):
                print(f"Starting fresh. Creating new Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
            else:
                print(f"Successfully connected to existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"[cache init error] Failed to handle Qdrant collection: {e}")

    def _get_embedding(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def query(self, prompt: str) -> dict | None:
        """Checks Qdrant for a semantically similar question"""
        try:
            vector = self._get_embedding(prompt)
            
            # The new modern Qdrant API syntax
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=1
            )
            
            # Extract the actual list of matches from the response object
            search_result = response.points
            
            if not search_result:
                return None  
                
            match = search_result[0]
            score = match.score
            
            if score >= self.threshold:
                print(f"[cache hit] Found a similar question in Qdrant with score: {score:.4f}")
                return match.payload  
            
            print(f"[cache miss] No similar question found in Qdrant. Highest score: {score:.4f}")
            return None
        except Exception as e:
            print(f"[cache query error] Failed to query Qdrant: {e}")
            return None
    
    def insert(self, prompt: str, response: str):
        """Inserts a prompt, its response, and its vector math into Qdrant"""
        try:
            vector = self._get_embedding(prompt)
            point_id = time.time_ns()  
            
            payload = {
                "prompt": prompt,
                "response": response,
                "timestamp": point_id
            }
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            print(f"[cache insert] Successfully saved vector embedding to Qdrant!")
        except Exception as e:
            print(f"[cache insert error] Failed to save to Qdrant: {e}")