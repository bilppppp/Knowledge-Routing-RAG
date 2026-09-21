#!/usr/bin/env python3
"""
Knowledge Routing RAG - Preflight Smoke Test (Phase -1)
Reference: 准备清单.md Section 26 & Section 27
Verifies:
  1. Embedding API connectivity, batch output & dimension
  2. Qdrant connectivity, collection lifecycle & doc_id filtering
  3. SQLite FTS5 extension & jieba Chinese search
  4. DeepSeek API connectivity & structured JSON output
  5. NetworkX Graph structure & Hub degree calculation
"""

import os
import sys
import json
import sqlite3
import httpx
import networkx as nx
from dotenv import load_dotenv

# Load environment
load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(title: str):
    print(f"\n{BLUE}=== {title} ==={RESET}")

def print_pass(msg: str):
    print(f"  {GREEN}[PASS]{RESET} {msg}")

def print_fail(msg: str):
    print(f"  {RED}[FAIL]{RESET} {msg}")

def test_embedding_service() -> bool:
    print_header("1. Testing Embedding Service")
    base_url = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
    expected_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    
    url = f"{base_url.rstrip('/')}/embeddings"
    payload = {
        "input": ["知识路由检索方案验证", "三级中医医院管理规定"],
        "model": model
    }
    
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                print_fail(f"HTTP Status {resp.status_code}: {resp.text}")
                return False
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            dim0 = len(embeddings[0])
            dim1 = len(embeddings[1])
            if dim0 == expected_dim and dim1 == expected_dim:
                print_pass(f"Endpoint accessible ({url})")
                print_pass(f"Model '{model}' generated {len(embeddings)} vectors")
                print_pass(f"Vector dimension strictly matches: {dim0}d")
                return True
            else:
                print_fail(f"Dimension mismatch: expected {expected_dim}, got {dim0}")
                return False
    except Exception as e:
        print_fail(f"Embedding connection failed: {e}")
        return False

def test_qdrant_service() -> bool:
    print_header("2. Testing Qdrant Vector Database")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "55001"))
    expected_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    test_collection = "smoke_test_verify"
    
    try:
        client = QdrantClient(host=host, port=port, timeout=10.0)
        # Check connection
        collections = client.get_collections()
        print_pass(f"Connected to Qdrant at {host}:{port}")
        
        # Recreate test collection
        if client.collection_exists(test_collection):
            client.delete_collection(test_collection)
            
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE)
        )
        print_pass(f"Created temporary collection '{test_collection}' (dim={expected_dim})")
        
        # Insert sample points with doc_id payloads (D20 vs D50)
        dummy_vec = [0.01] * expected_dim
        points = [
            PointStruct(id=1, vector=dummy_vec, payload={"doc_id": "doc001", "text": "D20 item"}),
            PointStruct(id=2, vector=dummy_vec, payload={"doc_id": "doc035", "text": "D50 distractor"})
        ]
        client.upsert(collection_name=test_collection, points=points)
        print_pass("Upserted points with metadata payload")
        
        # Test filtered search (doc_id == doc001)
        search_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value="doc001"))]
        )
        results = client.search(
            collection_name=test_collection,
            query_vector=dummy_vec,
            query_filter=search_filter,
            limit=5
        )
        if len(results) == 1 and results[0].payload["doc_id"] == "doc001":
            print_pass("Metadata filtering (doc_id filter for D20/D50/D100) works accurately")
        else:
            print_fail(f"Filter search returned unexpected count: {len(results)}")
            return False
            
        # Clean up
        client.delete_collection(test_collection)
        print_pass(f"Cleaned up temporary collection '{test_collection}'")
        return True
    except Exception as e:
        print_fail(f"Qdrant test failed: {e}")
        return False

def test_sqlite_fts5() -> bool:
    print_header("3. Testing SQLite FTS5 & jieba Chinese Search")
    import jieba
    
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        
        # Test FTS5 creation
        cur.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(chunk_id, doc_id, content);")
        print_pass("SQLite FTS5 virtual table created successfully")
        
        # Sample documents
        docs = [
            ("c1", "doc001", "根据三级中医医院评审标准，人员编制应符合国家规定。"),
            ("c2", "doc002", "医疗机构执业许可证应当在有效期内换发。")
        ]
        
        for cid, did, text in docs:
            # tokenize with jieba
            tokens = " ".join(jieba.cut_for_search(text))
            cur.execute("INSERT INTO documents_fts (chunk_id, doc_id, content) VALUES (?, ?, ?)", (cid, did, tokens))
            
        # Search query
        query_words = " ".join(jieba.cut_for_search("中医医院 评审标准"))
        cur.execute("SELECT chunk_id, doc_id FROM documents_fts WHERE content MATCH ?", (query_words,))
        rows = cur.fetchall()
        
        if len(rows) == 1 and rows[0][1] == "doc001":
            print_pass("jieba segmentation + SQLite FTS5 full-text search matched correctly")
            return True
        else:
            print_fail(f"FTS5 match returned unexpected results: {rows}")
            return False
    except Exception as e:
        print_fail(f"SQLite FTS5 test failed: {e}")
        return False

def test_deepseek_api() -> bool:
    print_header("4. Testing DeepSeek API & Structured Output")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    proxy = os.getenv("HTTPS_PROXY")
    
    if not api_key:
        print_fail("DEEPSEEK_API_KEY is not set in .env")
        return False
        
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = (
        "你是一个RAG评估控制器。请以严格的JSON格式输出以下字段："
        '{"status": "ready", "system": "Knowledge-Routing-RAG", "features": ["RIFT", "EIGRP", "SRv6"]}。'
        "只输出JSON，不要附加任何其他说明。"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        # First try direct, fallback to proxy if needed
        client_kwargs = {"timeout": 30.0}
        resp = None
        try:
            with httpx.Client(**client_kwargs) as client:
                resp = client.post(url, headers=headers, json=payload)
        except Exception:
            if proxy:
                client_kwargs["proxy"] = proxy
                with httpx.Client(**client_kwargs) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    
        if resp and resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if parsed.get("status") == "ready":
                print_pass(f"DeepSeek model '{model}' called successfully")
                print_pass(f"Structured JSON output verified: {parsed.get('system')}")
                return True
            else:
                print_fail(f"Parsed JSON content unexpected: {content}")
                return False
        else:
            status = resp.status_code if resp else "No Response"
            body = resp.text if resp else ""
            print_fail(f"DeepSeek call failed (Status {status}): {body}")
            return False
    except Exception as e:
        print_fail(f"DeepSeek API exception: {e}")
        return False

def test_networkx_graph() -> bool:
    print_header("5. Testing NetworkX Graph & Hub Threshold Logic")
    try:
        G = nx.DiGraph()
        
        # Add hierarchy nodes
        G.add_node("doc001", type="DOCUMENT")
        G.add_node("doc001#sec1", type="SECTION")
        G.add_node("doc001#chunk1", type="CHUNK")
        
        G.add_edge("doc001", "doc001#sec1", relation="HAS_CHILD")
        G.add_edge("doc001#sec1", "doc001#chunk1", relation="HAS_CHILD")
        
        # Add Hub concept node
        hub_node = "concept:医疗机构"
        G.add_node(hub_node, type="ROUTE_PREFIX")
        for i in range(10):
            target = f"doc{i+1:03d}#sec1"
            G.add_node(target, type="SECTION")
            G.add_edge(target, hub_node, relation="REFERENCES")
            
        degree = G.degree(hub_node)
        hub_threshold = max(6, 8)
        is_hub = degree >= hub_threshold
        
        print_pass("Hierarchy and relational DiGraph constructed")
        print_pass(f"Hub node '{hub_node}' degree = {degree}, threshold = {hub_threshold}")
        if is_hub:
            print_pass("Hub isolation triggered correctly (degree >= threshold)")
            return True
        else:
            print_fail("Hub detection failed")
            return False
    except Exception as e:
        print_fail(f"Graph test failed: {e}")
        return False

def main():
    print(f"\n{BLUE}======================================================{RESET}")
    print(f"{BLUE} Knowledge Routing RAG - Preflight Smoke Test {RESET}")
    print(f"{BLUE}======================================================{RESET}")
    
    results = [
        test_embedding_service(),
        test_qdrant_service(),
        test_sqlite_fts5(),
        test_deepseek_api(),
        test_networkx_graph()
    ]
    
    print_header("Summary")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"\n{GREEN}All {total} smoke tests passed successfully! Phase -1 environment is READY.{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}{total - passed} out of {total} smoke tests failed. Please review the errors above.{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
