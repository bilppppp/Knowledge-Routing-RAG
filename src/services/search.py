# src/services/search.py
"""
Search Service providing:
1. Pure Vector Retrieval (Qdrant with corpus filter)
2. Pure FTS5 BM25 Retrieval (SQLite FTS5 with jieba tokenization and corpus filter)
3. Hybrid Vector + FTS5 Retrieval with Reciprocal Rank Fusion (RRF)
"""

import os
import sqlite3
import jieba
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.common.models import EvidenceItem
from src.services.embedding import EmbeddingService

class SearchService:
    def __init__(
        self,
        sqlite_path: str = "data/knowledge_lsdb.sqlite",
        qdrant_url: str = "http://localhost:55001",
        collection_name: str = "knowledge_routing_exp",
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.sqlite_path = sqlite_path
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()

    def vector_search(
        self,
        query: str,
        corpus: str = "D20",
        top_k: int = 5
    ) -> List[EvidenceItem]:
        """
        Executes vector search filtered strictly by corpus membership.
        """
        q_vec = self.embedding_service.get_embedding(query)
        corpus_field = f"in_{corpus.lower()}"
        
        corpus_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=corpus_field,
                    match=qmodels.MatchValue(value=True)
                )
            ]
        )
        
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            query_filter=corpus_filter,
            limit=top_k
        )
        
        items = []
        for h in hits:
            p = h.payload
            items.append(EvidenceItem(
                chunk_id=p["chunk_id"],
                doc_id=p["doc_id"],
                title=p["title"],
                heading_path=p["heading_path"],
                text=p["text"],
                score=float(h.score),
                source_method="vector"
            ))
        return items

    def fts_search(
        self,
        query: str,
        corpus: str = "D20",
        top_k: int = 5
    ) -> List[EvidenceItem]:
        """
        Executes BM25 full-text search against SQLite FTS5 filtered by corpus.
        """
        tokens = " ".join([w for w in jieba.cut(query) if len(w.strip()) > 0])
        if not tokens.strip():
            tokens = query
            
        corpus_col = f"c.in_{corpus.lower()}"
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        
        # SQLite FTS5 BM25 search joined with chunks table for corpus filtering
        sql = f"""
            SELECT 
                c.chunk_id, c.doc_id, c.title, c.heading_path, c.text, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
            WHERE chunks_fts.tokenized_text MATCH ?
              AND {corpus_col} = 1
            ORDER BY score ASC
            LIMIT ?;
        """
        try:
            cur.execute(sql, (tokens, top_k))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Fallback for special syntax tokens
            safe_tokens = " OR ".join([f'"{w}"' for w in jieba.cut(query) if len(w.strip()) > 1])
            cur.execute(sql, (safe_tokens, top_k))
            rows = cur.fetchall()
        finally:
            conn.close()

        items = []
        for r in rows:
            items.append(EvidenceItem(
                chunk_id=r[0],
                doc_id=r[1],
                title=r[2],
                heading_path=r[3],
                text=r[4],
                score=float(-r[5]),  # BM25 is lower is better, invert for ranking
                source_method="fts"
            ))
        return items

    def hybrid_search(
        self,
        query: str,
        corpus: str = "D20",
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[EvidenceItem]:
        """
        Hybrid Vector + FTS5 search using Reciprocal Rank Fusion (RRF).
        """
        # Retrieve candidate pools
        pool_size = max(top_k * 2, 10)
        vec_items = self.vector_search(query, corpus=corpus, top_k=pool_size)
        fts_items = self.fts_search(query, corpus=corpus, top_k=pool_size)

        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, EvidenceItem] = {}

        for rank, item in enumerate(vec_items):
            cid = item.chunk_id
            item_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, item in enumerate(fts_items):
            cid = item.chunk_id
            if cid not in item_map:
                item_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by combined RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        results = []
        for cid in sorted_cids[:top_k]:
            it = item_map[cid]
            it.score = rrf_scores[cid]
            it.source_method = "hybrid_rrf"
            results.append(it)
            
        return results
