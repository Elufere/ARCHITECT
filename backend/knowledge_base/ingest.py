"""
Knowledge Base Ingestion Pipeline (Phase 2)

1. Reads scraped markdown files from .okf/
2. Chunks them by H2 (##) headings
3. Classifies each chunk as 'rule' or 'pattern'
4. Saves 'rule' chunks to static files (for system prompt injection)
5. Embeds 'pattern' chunks into ChromaDB (for RAG similarity search)
"""

from __future__ import annotations

import re
import logging
import yaml
from pathlib import Path
from typing import TypedDict

import chromadb

logger = logging.getLogger(__name__)

# Paths
OKF_ROOT = Path(__file__).resolve().parent / ".okf"
RULES_ROOT = Path(__file__).resolve().parent / "rules"
CHROMA_ROOT = Path(__file__).resolve().parent / "chroma_store"

# --- Heuristic Classifier ---

RULE_KEYWORDS = [
    "must", "shall", "required", "must not", "shall not", "mandatory",
    "syntax", "ddl", "alter table",
    "openapi", "json schema", "rfc 2119", "standard format",
    "constraint", "unique", "not null", "foreign key",
]

PATTERN_KEYWORDS = [
    "consider", "best practice", "strategy", "pattern", "recommend",
    "approach", "typical", "common", "often", "usually", "tend to",
    "good way", "better to", "avoid doing", "pitfall", "trade-off",
    "example", "scenario", "use case",
]

CODE_FENCE_RULE_MARKERS = [
    "create table", "alter table", "openapi:", "swagger:",
    "paths:", "components:", "type: object", "not null",
    "foreign key", "primary key", "unique constraint",
]


def classify_chunk(text: str) -> str:
    """
    Heuristic classification of a text chunk.
    Returns 'rule' or 'pattern'.
    """
    lower_text = text.lower()

    # Extract fenced code blocks separately — strict syntax usually lives here
    code_blocks = re.findall(r"```.*?```", text, flags=re.DOTALL)
    code_text = " ".join(code_blocks).lower()

    # Strong signal: structural syntax markers inside actual code blocks
    if any(marker in code_text for marker in CODE_FENCE_RULE_MARKERS):
        return "rule"

    rule_score = sum(1 for kw in RULE_KEYWORDS if kw in lower_text)
    pattern_score = sum(1 for kw in PATTERN_KEYWORDS if kw in lower_text)

    # With the code-fence override catching DDL/OpenAPI, we can safely
    # keep the prose threshold strict to avoid locking conversational advice.
    if rule_score > pattern_score:
        return "rule"

    return "pattern"

# --- Chunking Logic ---

class Chunk(TypedDict):
    text: str
    header: str
    chunk_id: str
    source_slug: str
    category: str
    source_url: str
    classification: str


def chunk_by_headers(markdown_text: str, source_slug: str, category: str, source_url: str) -> list[Chunk]:
    """
    Split markdown text by H2 (##) headings.
    """
    # Split by H2 headers
    parts = re.split(r"(?=^## )", markdown_text, flags=re.MULTILINE)
    
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        # Extract header
        header_match = re.match(r"^## (.+)", part)
        header = header_match.group(1).strip() if header_match else "Introduction"
        
        # Clean header for ID generation
        clean_header = re.sub(r"[^a-z0-9]+", "-", header.lower()).strip("-")
        chunk_id = f"{source_slug}--{clean_header}--{i}" if clean_header else f"{source_slug}--intro--{i}"

        header_is_toc = "table of contents" in header.lower()
        lines = [l.strip() for l in part.split("\n") if l.strip()]
        if lines:
            link_lines = sum(1 for l in lines if l.startswith("[") and "](" in l)
            content_is_toc = (link_lines / len(lines)) > 0.3
        else:
            content_is_toc = False
            
        if header_is_toc or content_is_toc:
            continue

        chunks.append({
            "text": part,
            "header": header,
            "chunk_id": chunk_id,
            "source_slug": source_slug,
            "category": category,
            "source_url": source_url,
            "classification": classify_chunk(part)
        })
        
    return chunks


def extract_frontmatter(markdown_text: str) -> dict:
    """Extract YAML frontmatter as a dict."""
    match = re.match(r"^---\s*\n(.*?)\n---", markdown_text, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = yaml.safe_load(match.group(1))
        return parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse frontmatter: {e}")
        return {}


# --- Storage Logic ---

def save_rule_chunk(chunk: Chunk) -> None:
    """Save a rule chunk to a static markdown file."""
    category_dir = RULES_ROOT / chunk["category"]
    category_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = category_dir / f"{chunk['chunk_id']}.md"
    
    content = f"# {chunk['header']}\n\n"
    content += f"> Source: [{chunk['source_slug']}]({chunk['source_url']})\n\n"
    content += chunk["text"]
    
    filepath.write_text(content, encoding="utf-8")
    logger.debug(f"Saved rule: {filepath.name}")


def embed_pattern_chunks(chunks: list[Chunk]) -> None:
    """Embed pattern chunks into ChromaDB."""
    CHROMA_ROOT.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))
    
    # Get or create collection. 
    # sentence-transformers: all-MiniLM-L6-v2
    collection = client.get_or_create_collection(
        name="engineering_patterns",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare batch data
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "header": c["header"],
        "source_slug": c["source_slug"],
        "category": c["category"],
        "source_url": c["source_url"],
    } for c in chunks]
    
    # ChromaDB handles batching, but let's do it in slices of 100 to be safe on memory
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
        
    logger.info(f"Embedded {len(ids)} pattern chunks into ChromaDB.")


# --- Main Pipeline ---

def run_ingestion() -> None:
    logger.info("Starting ingestion pipeline...")
    
    md_files = list(OKF_ROOT.rglob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in .okf/")
    
    all_pattern_chunks = []
    stats = {"total_chunks": 0, "rules": 0, "patterns": 0}
    
    for md_file in md_files:
        # Determine category from parent folder (e.g., .okf/standards/)
        category = md_file.parent.name
        
        # Read and strip frontmatter for chunking, but keep it for metadata
        raw_text = md_file.read_text(encoding="utf-8")
        frontmatter = extract_frontmatter(raw_text)
        source_url = frontmatter.get("source_url", "")
        slug = frontmatter.get("slug", md_file.stem)
        
        # Remove frontmatter before chunking
        clean_text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw_text, flags=re.DOTALL)
        
        # Chunk
        chunks = chunk_by_headers(clean_text, slug, category, source_url)
        
        for chunk in chunks:
            stats["total_chunks"] += 1
            if chunk["classification"] == "rule":
                stats["rules"] += 1
                save_rule_chunk(chunk)
            else:
                stats["patterns"] += 1
                all_pattern_chunks.append(chunk)
                
    # Embed all patterns at once
    if all_pattern_chunks:
        embed_pattern_chunks(all_pattern_chunks)
        
    logger.info("--- Ingestion Complete ---")
    logger.info(f"Total Chunks: {stats['total_chunks']}")
    logger.info(f"Rules (Static Files): {stats['rules']}")
    logger.info(f"Patterns (ChromaDB): {stats['patterns']}")
    
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    print(f"Total Chunks Processed: {stats['total_chunks']}")
    print(f"Classified as RULES (saved to /rules): {stats['rules']}")
    print(f"Classified as PATTERNS (embedded in ChromaDB): {stats['patterns']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_ingestion()