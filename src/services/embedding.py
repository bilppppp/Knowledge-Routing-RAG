# src/services/embedding.py
"""
Embedding Service Client.
Interacts with local Docker Infinity embedding endpoint (google/embeddinggemma-300m, 768 dim).
Includes caching for query embeddings to accelerate repeated queries.
"""

import os
import httpx
from typing import List, Dict

class EmbeddingService:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
        self.client = httpx.Client(timeout=30.0)
        self._cache: Dict[str, List[float]] = {}

    def get_embedding(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        
        resp = self.client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": [text]}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding request failed: {resp.status_code} {resp.text}")
            
        data = resp.json()
        emb = data["data"][0]["embedding"]
        self._cache[text] = emb
        return emb

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # Check cache
        needed_indices = []
        needed_texts = []
        results = [None] * len(texts)
        
        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                needed_indices.append(i)
                needed_texts.append(t)
                
        if needed_texts:
            resp = self.client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": needed_texts}
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Batch embedding request failed: {resp.status_code} {resp.text}")
            data = resp.json()
            for idx, item in zip(needed_indices, data["data"]):
                emb = item["embedding"]
                self._cache[texts[idx]] = emb
                results[idx] = emb
                
        return results

    def close(self):
        self.client.close()
