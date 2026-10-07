"""
Stage 1: Document chunking.

Why we chunk:
LLMs and embedding models have a limited context window, and retrieval
quality drops if you embed an entire document as a single vector (too much
averaged-out meaning). So we split documents into smaller, semantically
coherent pieces ("chunks"), embed each one separately, and retrieve only
the chunks relevant to a given query.

We use a *recursive character splitter*: try to split on paragraph breaks
first, then sentences, then words -- only falling back to a hard character
cut if nothing else fits. This keeps chunks from breaking mid-sentence
whenever possible, so each chunk stays meaningful on its own.

CHUNK_OVERLAP exists so information near a chunk boundary isn't lost -- if
a fact spans the end of chunk N and the start of chunk N+1, overlap means
at least one of the two chunks still contains the full fact.
"""

import os
from typing import List, Dict
from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text.strip(), separators, chunk_size, overlap)


def _recursive_split(text: str, separators: List[str], chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep:
        pieces = text.split(sep)
    else:
        pieces = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks, current = [], ""
    for piece in pieces:
        candidate = current + (sep if current else "") + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) > chunk_size and remaining_seps:
                chunks.extend(_recursive_split(piece, remaining_seps, chunk_size, overlap))
                current = ""
            else:
                current = piece

    if current:
        chunks.append(current)

    # prepend the tail of the previous chunk to create overlap
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0 or overlap == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-overlap:]
            overlapped.append(tail + " " + c)
    return overlapped


def load_and_chunk_documents(docs_dir: str = DOCS_DIR) -> List[Dict]:
    """Reads every .txt file in docs_dir and returns a list of chunk records:
    {"id": "<file>_<chunk_idx>", "text": "...", "source": "<file>"}
    """
    records = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, c in enumerate(chunks):
            records.append({"id": f"{fname}_{i}", "text": c, "source": fname})
    return records


if __name__ == "__main__":
    recs = load_and_chunk_documents()
    print(f"Loaded {len(recs)} chunks from {DOCS_DIR}")
    if recs:
        print("--- sample chunk ---")
        print(recs[0]["text"][:300])
