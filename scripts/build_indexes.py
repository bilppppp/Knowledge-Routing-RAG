#!/usr/bin/env python3
"""
Index Construction & Knowledge LSDB Builder (Phase 1)
Reference: 实验方案.md Section 9, 10, 11 & 准备清单.md Section 17, 18, 19, 20

Tasks:
1. Build SQLite Knowledge LSDB (data/knowledge_lsdb.sqlite):
   - documents table
   - chunks table
   - chunks_fts table (FTS5 with jieba whitespace tokenization)
   - graph nodes & edges tables (populating approved routing graph & structural edges)
2. Build Qdrant Vector Index (collection: knowledge_routing_exp):
   - Model: google/embeddinggemma-300m (768 dim, Cosine)
   - Batch embed all 2,862 chunks via local Docker endpoint
   - Upload vectors with payload filters: chunk_id, doc_id, in_d20, in_d50, in_d100
3. Verification & smoke testing of both vector & FTS5 filtered searches
"""

import os
import sys
import json
import sqlite3
import time
import httpx
import jieba
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
SQLITE_PATH = DATA_DIR / "knowledge_lsdb.sqlite"

EMBEDDING_URL = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 55001))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_routing_exp")

# 1. Load documents and chunks
with open(DATA_DIR / "manifests" / "corpus_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

chunks = []
with open(DATA_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Loaded {len(manifest)} documents and {len(chunks)} chunks.")

# 2. Build SQLite Knowledge LSDB
def build_sqlite_lsdb():
    print("\n--- 1. Building SQLite Knowledge LSDB ---")
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    # Documents table
    cur.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            char_count INTEGER,
            in_d20 BOOLEAN NOT NULL,
            in_d50 BOOLEAN NOT NULL,
            in_d100 BOOLEAN NOT NULL
        );
    """)
    for d in manifest:
        cur.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (d["doc_id"], d["title"], d["relative_path"], d.get("char_count", 0), d["in_d20"], d["in_d50"], d["in_d100"])
        )

    # Chunks table
    cur.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            section_id TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            order_num INTEGER NOT NULL,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            in_d20 BOOLEAN NOT NULL,
            in_d50 BOOLEAN NOT NULL,
            in_d100 BOOLEAN NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
        );
    """)
    for c in chunks:
        cur.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c["chunk_id"], c["doc_id"], c["title"], c["section_id"], c["heading_path"], c["order"], c["text"], c["char_count"], c["in_d20"], c["in_d50"], c["in_d100"])
        )

    # FTS5 virtual table
    cur.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            title,
            heading_path,
            text,
            tokenized_text
        );
    """)
    
    print("Tokenizing and indexing chunks into FTS5...")
    fts_rows = []
    for c in chunks:
        # Segment text and title with jieba for accurate Chinese full-text search
        words = jieba.cut(c["text"])
        tokenized_text = " ".join(words)
        tokenized_title = " ".join(jieba.cut(c["title"]))
        tokenized_heading = " ".join(jieba.cut(c["heading_path"]))
        fts_rows.append((c["chunk_id"], c["doc_id"], tokenized_title, tokenized_heading, c["text"], tokenized_text))
        
    cur.executemany(
        "INSERT INTO chunks_fts (chunk_id, doc_id, title, heading_path, text, tokenized_text) VALUES (?, ?, ?, ?, ?, ?)",
        fts_rows
    )

    # Graph tables: nodes and edges
    cur.execute("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            title TEXT,
            doc_id TEXT,
            in_d20 BOOLEAN,
            in_d50 BOOLEAN,
            in_d100 BOOLEAN
        );
    """)
    
    cur.execute("""
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            confidence REAL DEFAULT 1.0,
            provenance TEXT,
            is_routing BOOLEAN NOT NULL
        );
    """)

    # Populate nodes and edges from preflight graph logic
    from scripts.preflight_analysis import build_graphs
    G_full, G_routing, extracted_edges, stats = build_graphs(manifest, chunks)

    # Insert nodes
    node_rows = []
    for node, attrs in G_full.nodes(data=True):
        ntype = attrs.get("type", "UNKNOWN")
        title = attrs.get("title", "")
        doc_id = attrs.get("doc_id", "")
        in_d20 = attrs.get("in_d20", False)
        in_d50 = attrs.get("in_d50", False)
        in_d100 = attrs.get("in_d100", False)
        node_rows.append((node, ntype, title, doc_id, in_d20, in_d50, in_d100))
    cur.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", node_rows)

    # Insert edges
    edge_rows = []
    for u, v, attrs in G_full.edges(data=True):
        rel = attrs.get("relation", "RELATED_TO")
        weight = attrs.get("weight", 1.0)
        conf = attrs.get("confidence", 1.0)
        prov = attrs.get("provenance", "")
        is_routing = (rel not in ["HAS_CHILD", "PART_OF", "NEXT", "PREVIOUS"])
        edge_rows.append((u, v, rel, weight, conf, prov, is_routing))
    cur.executemany(
        "INSERT INTO edges (source, target, relation, weight, confidence, provenance, is_routing) VALUES (?, ?, ?, ?, ?, ?, ?)",
        edge_rows
    )

    # Create indexes
    cur.execute("CREATE INDEX idx_edges_source ON edges(source);")
    cur.execute("CREATE INDEX idx_edges_target ON edges(target);")
    cur.execute("CREATE INDEX idx_edges_rel ON edges(relation);")
    cur.execute("CREATE INDEX idx_edges_routing ON edges(is_routing);")
    cur.execute("CREATE INDEX idx_chunks_doc ON chunks(doc_id);")

    conn.commit()
    conn.close()
    print(f"SQLite Knowledge LSDB built: {SQLITE_PATH} ({len(node_rows)} nodes, {len(edge_rows)} edges)")

# 3. Build Qdrant Vector Collection
def build_qdrant_index():
    print("\n--- 2. Building Qdrant Vector Index ---")
    client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
    
    # Check if collection exists
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        print(f"Recreating existing collection '{QDRANT_COLLECTION}'...")
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=768,
            distance=qmodels.Distance.COSINE
        )
    )
    print(f"Created collection '{QDRANT_COLLECTION}' (dim=768, metric=Cosine).")

    # Embedding chunks in batches
    batch_size = 32
    http_client = httpx.Client(timeout=60.0)
    
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    print(f"Embedding {total_chunks} chunks in {total_batches} batches...")

    t0 = time.time()
    points = []
    
    for b_idx in range(total_batches):
        batch = chunks[b_idx * batch_size : (b_idx + 1) * batch_size]
        texts = [f"{c['title']} > {c['heading_path']}\n{c['text']}" for c in batch]
        
        resp = http_client.post(
            f"{EMBEDDING_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "input": texts}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding batch {b_idx} failed: {resp.text}")
            
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        
        for c, emb in zip(batch, embeddings):
            # point id: integer hash or sequential index
            point_id = int(c["chunk_id"].replace("doc", "").replace("#c", ""))
            payload = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "heading_path": c["heading_path"],
                "text": c["text"],
                "char_count": c["char_count"],
                "in_d20": c["in_d20"],
                "in_d50": c["in_d50"],
                "in_d100": c["in_d100"]
            }
            points.append(qmodels.PointStruct(id=point_id, vector=emb, payload=payload))

        if (b_idx + 1) % 10 == 0 or b_idx == total_batches - 1:
            # Upload batch to Qdrant
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            print(f"  Processed {min((b_idx + 1) * batch_size, total_chunks)} / {total_chunks} chunks...")
            points.clear()

    # Create payload indexes for fast filtered searches
    print("Creating Qdrant payload indexes...")
    client.create_payload_index(QDRANT_COLLECTION, field_name="in_d20", field_schema=qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(QDRANT_COLLECTION, field_name="in_d50", field_schema=qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(QDRANT_COLLECTION, field_name="in_d100", field_schema=qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(QDRANT_COLLECTION, field_name="doc_id", field_schema=qmodels.PayloadSchemaType.KEYWORD)

    dur = time.time() - t0
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"Qdrant indexing complete in {dur:.2f}s! Total points: {info.points_count}")

# 4. Smoke Verification of Vector and FTS5 Queries
def verify_searches():
    print("\n--- 3. Verifying Vector & FTS5 Search Filtering ---")
    # Test query
    test_q = "执业医师 考核周期"
    
    # 1. Test FTS5
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    tokens = " ".join(jieba.cut(test_q))
    cur.execute("""
        SELECT chunk_id, title, heading_path, bm25(chunks_fts) as score
        FROM chunks_fts
        WHERE tokenized_text MATCH ?
        ORDER BY score
        LIMIT 3;
    """, (tokens,))
    fts_results = cur.fetchall()
    print("FTS5 Top 3 results:")
    for r in fts_results:
        print(f"  [{r[0]}] {r[1]} - {r[2]} (bm25: {r[3]:.4f})")
    assert len(fts_results) > 0, "FTS5 search returned 0 results!"
    conn.close()

    # 2. Test Qdrant Vector Search
    http_client = httpx.Client(timeout=30.0)
    resp = http_client.post(f"{EMBEDDING_URL}/embeddings", json={"model": EMBEDDING_MODEL, "input": [test_q]})
    q_vec = resp.json()["data"][0]["embedding"]
    
    client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
    
    for corpus in ["D20", "D50", "D100"]:
        corpus_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key=f"in_{corpus.lower()}", match=qmodels.MatchValue(value=True))]
        )
        vec_hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            query_filter=corpus_filter,
            limit=3
        )
        print(f"Vector Top 3 in {corpus}:")
        for h in vec_hits:
            print(f"  [{h.payload['chunk_id']}] {h.payload['title']} (score: {h.score:.4f})")
        assert len(vec_hits) > 0, f"Vector search returned 0 hits in {corpus}!"

    print("\n Phase 1 Index Construction & Knowledge LSDB COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    build_sqlite_lsdb()
    build_qdrant_index()
    verify_searches()
