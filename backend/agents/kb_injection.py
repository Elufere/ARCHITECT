"""
Knowledge Base Injection utilities for Agents.
Bridges the chunking pipeline with LLM agents.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)

# Paths relative to the backend folder
KB_ROOT = Path(__file__).resolve().parent.parent / "knowledge_base"
RULES_DIR = KB_ROOT / "rules"
CHROMA_DIR = KB_ROOT / "chroma_store"


def load_static_rules(categories: list[str] = None) -> str:
    """
    Reads static rule .md files to inject directly into the system prompt.
    If no categories are provided, it loads all rules.
    """
    if categories is None:
        categories = [d.name for d in RULES_DIR.iterdir() if d.is_dir()]
        
    rules_text = []
    for cat in categories:
        cat_dir = RULES_DIR / cat
        if not cat_dir.exists():
            continue
            
        for md_file in cat_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            # Strip the header line to save tokens
            lines = content.split("\n")
            clean_content = "\n".join(lines[1:]).strip()
            rules_text.append(f"### Rule: {md_file.stem}\n{clean_content}")
            
    if not rules_text:
        return "No static rules defined."
        
    logger.info(f"Loaded {len(rules_text)} static rules from categories: {categories}")
    return "\n\n---\n\n".join(rules_text)


def search_kb_patterns(query: str, category: Optional[str] = None, n_results: int = 3) -> str:
    """
    Dynamically queries ChromaDB for situational patterns.
    Used when the user mentions a specific domain (e.g., 'payments', 'auth').
    """
    if not CHROMA_DIR.exists():
        return ""
        
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name="engineering_patterns")
        
        where_filter = {"category": category} if category else None
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        if not results["documents"][0]:
            return "No relevant patterns found in knowledge base."
            
        patterns_text = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            patterns_text.append(f"**Pattern from {meta['source_slug']} ({meta['header']}):**\n{doc[:500]}...")
            
        return "\n\n".join(patterns_text)
        
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return ""