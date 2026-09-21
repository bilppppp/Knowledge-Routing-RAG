# src/graph/lsdb.py
"""
Knowledge LSDB (Link-State Database) Graph Engine.
Reference: 实验方案.md Section 10, 11, 12, 13

Features:
- Immutable shared routing graph loaded directly from SQLite LSDB
- Strictly separates Structural Edges (HAS_CHILD, PART_OF, NEXT, PREVIOUS) from Routing Edges (SUPERSEDES, AMENDS, BASED_ON, REFERENCES)
- Precise Hub Detection using frozen thresholds:
    H_fanout = 11 (Out-degree Hubs: fan-out explosion prevention)
    H_in = 6      (In-degree Hubs: fan-in aggregation nodes)
    H_total = 13  (Total-degree Hubs)
- Scope narrowing and typed expansion to prevent graph flooding
"""

import sqlite3
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path

class KnowledgeLSDB:
    def __init__(
        self,
        sqlite_path: str = "data/knowledge_lsdb.sqlite",
        h_fanout: int = 11,
        h_in: int = 6,
        h_total: int = 13
    ):
        self.sqlite_path = sqlite_path
        self.h_fanout = h_fanout
        self.h_in = h_in
        self.h_total = h_total
        
        self.G_full = nx.DiGraph()
        self.G_routing = nx.DiGraph()
        self.chunk_meta: Dict[str, Dict[str, Any]] = {}
        self.doc_meta: Dict[str, Dict[str, Any]] = {}
        
        self._load_from_sqlite()

    def _load_from_sqlite(self):
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()

        # Load Documents
        cur.execute("SELECT doc_id, title, relative_path, in_d20, in_d50, in_d100 FROM documents;")
        for r in cur.fetchall():
            self.doc_meta[r[0]] = {
                "doc_id": r[0], "title": r[1], "relative_path": r[2],
                "in_d20": bool(r[3]), "in_d50": bool(r[4]), "in_d100": bool(r[5])
            }

        # Load Chunks
        cur.execute("SELECT chunk_id, doc_id, title, section_id, heading_path, order_num, text, char_count, in_d20, in_d50, in_d100 FROM chunks;")
        for r in cur.fetchall():
            self.chunk_meta[r[0]] = {
                "chunk_id": r[0], "doc_id": r[1], "title": r[2], "section_id": r[3],
                "heading_path": r[4], "order_num": r[5], "text": r[6], "char_count": r[7],
                "in_d20": bool(r[8]), "in_d50": bool(r[9]), "in_d100": bool(r[10])
            }

        # Load Nodes
        cur.execute("SELECT node_id, node_type, title, doc_id, in_d20, in_d50, in_d100 FROM nodes;")
        for r in cur.fetchall():
            node_id, ntype, title, doc_id, d20, d50, d100 = r
            attrs = {
                "node_type": ntype, "title": title, "doc_id": doc_id,
                "in_d20": bool(d20), "in_d50": bool(d50), "in_d100": bool(d100)
            }
            self.G_full.add_node(node_id, **attrs)
            if ntype in ["DOCUMENT", "CHUNK", "ROUTE_PREFIX"]:
                self.G_routing.add_node(node_id, **attrs)

        # Load Edges
        cur.execute("SELECT source, target, relation, weight, confidence, provenance, is_routing FROM edges;")
        for r in cur.fetchall():
            src, tgt, rel, weight, conf, prov, is_routing = r
            edge_attrs = {
                "relation": rel, "weight": weight, "confidence": conf,
                "provenance": prov, "is_routing": bool(is_routing)
            }
            self.G_full.add_edge(src, tgt, **edge_attrs)
            if bool(is_routing):
                self.G_routing.add_edge(src, tgt, **edge_attrs)

        conn.close()

    def is_hub(self, node: str) -> bool:
        """
        Hub determination based on non-leaf routing degrees.
        A node is a routing hub if its out-degree >= 11 or total-degree >= 13.
        """
        if not self.G_routing.has_node(node):
            return False
        out_deg = self.G_routing.out_degree(node)
        tot_deg = self.G_routing.degree(node)
        return (out_deg >= self.h_fanout) or (tot_deg >= self.h_total)

    def is_in_degree_hub(self, node: str) -> bool:
        if not self.G_routing.has_node(node):
            return False
        return self.G_routing.in_degree(node) >= self.h_in

    def get_routing_neighbors(
        self,
        node: str,
        corpus: str = "D20",
        allowed_relations: Optional[Set[str]] = None,
        prevent_hub_explosion: bool = True
    ) -> List[Tuple[str, str, float]]:
        """
        Returns list of (target_node, relation, cost).
        If prevent_hub_explosion is True and node is a Hub, suppresses full neighbors(*) expansion.
        """
        if not self.G_routing.has_node(node):
            return []

        # Enforce Hub safety constraint
        if prevent_hub_explosion and self.is_hub(node):
            # Prohibit unconstrained neighbors(*) expansion
            return []

        corpus_flag = f"in_{corpus.lower()}"
        neighbors = []
        for target in self.G_routing.successors(node):
            edge_data = self.G_routing.get_edge_data(node, target)
            rel = edge_data.get("relation", "")
            
            if allowed_relations and rel not in allowed_relations:
                continue

            # Check corpus boundary for DOCUMENT and CHUNK nodes
            target_data = self.G_routing.nodes[target]
            ntype = target_data.get("node_type", "")
            if ntype in ["DOCUMENT", "CHUNK"]:
                if not target_data.get(corpus_flag, False):
                    continue

            # Standard cost lookup
            cost = 1.0
            if rel in ["SUPERSEDES", "AMENDS"]:
                cost = 1.0
            elif rel == "BASED_ON":
                cost = 1.0
            elif rel == "REFERENCES":
                cost = 1.2
            else:
                cost = 1.5

            neighbors.append((target, rel, cost))

        return neighbors

    def get_scope_prefix(self, node: str) -> Optional[str]:
        """
        Returns the Route Prefix associated with a node, or its Document ID.
        """
        if node in self.doc_meta:
            return node
        if node in self.chunk_meta:
            return self.chunk_meta[node]["doc_id"]
        return None

    def get_chunk_evidence(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.chunk_meta.get(chunk_id)

    def get_document_chunks(self, doc_id: str, corpus: str = "D20") -> List[str]:
        corpus_flag = f"in_{corpus.lower()}"
        return [
            cid for cid, c in self.chunk_meta.items()
            if c["doc_id"] == doc_id and c.get(corpus_flag, False)
        ]
