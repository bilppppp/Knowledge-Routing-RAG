#!/usr/bin/env python3
"""
Import top 100 largest text documents from source archive into Knowledge-Routing-RAG.
Creates:
  - data/documents/doc001.txt ... doc100.txt
  - data/manifests/corpus_manifest.json
  - data/manifests/d20.json
  - data/manifests/d50.json
  - data/manifests/d100.json
"""

import os
import re
import shutil
import os
import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(os.getenv("SRC_DIR", str(BASE_DIR / "data" / "raw_documents")))
DOCS_DIR = BASE_DIR / "data" / "documents"
MANIFESTS_DIR = BASE_DIR / "data" / "manifests"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def clean_filename_title(fn: str) -> str:
    """Derive canonical legal title from source filename."""
    s = fn.replace(".txt", "").strip()
    s = re.sub(r"^[《“](.*?)[》”]$", r"\1", s)
    s = re.sub(r"^批复[︱|]\s*(.*?)[：:]", "", s)
    s = re.sub(r"^文件[︱|]\s*(.*?)[：:]", "", s)
    return s.strip()

def main():
    all_files = list(SRC_DIR.glob("*.txt"))
    all_files_with_size = [(f, f.stat().st_size) for f in all_files]
    all_files_with_size.sort(key=lambda x: x[1], reverse=True)
    
    top_100 = all_files_with_size[:100]
    manifest_entries = []
    d20_ids = []
    d50_ids = []
    d100_ids = []
    
    for idx, (src_file, size_bytes) in enumerate(top_100, start=1):
        doc_id = f"doc{idx:03d}"
        dest_filename = f"{doc_id}.txt"
        dest_file = DOCS_DIR / dest_filename
        
        text = src_file.read_text(encoding="utf-8")
        title = clean_filename_title(src_file.name)
        shutil.copy2(src_file, dest_file)
        
        in_d20 = idx <= 20
        in_d50 = idx <= 50
        in_d100 = True
        
        if in_d20:
            d20_ids.append(doc_id)
        if in_d50:
            d50_ids.append(doc_id)
        d100_ids.append(doc_id)
        
        manifest_entries.append({
            "doc_id": doc_id,
            "title": title,
            "original_filename": src_file.name,
            "relative_path": f"data/documents/{dest_filename}",
            "size_bytes": size_bytes,
            "char_count": len(text),
            "line_count": len(text.splitlines()),
            "md5": compute_md5(text),
            "in_d20": in_d20,
            "in_d50": in_d50,
            "in_d100": in_d100
        })
        
    assert len(d20_ids) == 20
    assert len(d50_ids) == 50
    assert len(d100_ids) == 100
    assert set(d20_ids).issubset(set(d50_ids))
    assert set(d50_ids).issubset(set(d100_ids))
    
    (MANIFESTS_DIR / "corpus_manifest.json").write_text(json.dumps(manifest_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    (MANIFESTS_DIR / "d20.json").write_text(json.dumps({"corpus": "D20", "count": 20, "doc_ids": d20_ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    (MANIFESTS_DIR / "d50.json").write_text(json.dumps({"corpus": "D50", "count": 50, "doc_ids": d50_ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    (MANIFESTS_DIR / "d100.json").write_text(json.dumps({"corpus": "D100", "count": 100, "doc_ids": d100_ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated manifests with clean titles for {len(top_100)} documents.")

if __name__ == "__main__":
    main()
