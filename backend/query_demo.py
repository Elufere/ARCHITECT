"""
Demo: Filtered Similarity Search against the Pattern KB.
"""

import chromadb
from pathlib import Path

CHROMA_ROOT = Path(__file__).resolve().parent / "knowledge_base" / "chroma_store"

def search_patterns(query: str, category: str | None = None, n_results: int = 3):
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))
    collection = client.get_collection(name="engineering_patterns")
    
    where_filter = {"category": category} if category else None
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    
    print(f"\n{'='*70}")
    print(f"QUERY: '{query}'")
    if category:
        print(f"FILTER: category = '{category}'")
    print(f"{'='*70}\n")
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    )):
        print(f"--- Result {i+1} (Distance: {dist:.4f}) ---")
        print(f"Source: {meta['source_slug']} | Section: {meta['header']}")
        print(f"Category: {meta['category']}")
        # Print first 300 chars of the chunk
        print(doc[:300] + "...\n")

if __name__ == "__main__":
    print("Demo 1: Unfiltered search (across all categories)")
    search_patterns("How do I handle pagination in a REST API?")
    
    print("\n\n")
    
    print("Demo 2: Filtered search (ONLY security)")
    search_patterns("How do I prevent users from accessing other users data?", category="security")
    
    print("\n\n")
    
    print("Demo 3: Filtered search (ONLY standards)")
    search_patterns("When should I use a partial index in postgres?", category="standards")