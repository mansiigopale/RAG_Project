"""
Stage 2: Embeddings + FAISS indexing.

Why embeddings:
An embedding model turns text into a dense vector such that semantically
similar text ends up close together in vector space. This gives us
"search by meaning" instead of keyword search -- a query like "how do I
reset my password" can retrieve a chunk that says "to recover account
access, go to Settings" even though no words overlap.

We use `all-MiniLM-L6-v2` (sentence-transformers): 384-dim, small, fast,
runs on CPU, no API cost -- good for a portfolio project since you don't
need external API keys just to build the index.

Why FAISS:
FAISS (Facebook AI Similarity Search) does fast nearest-neighbor search
over dense vectors. A linear scan would work for a tiny dataset too, but
FAISS is the standard tool this project is meant to demonstrate, and the
same API scales to millions of vectors.

We use IndexFlatIP (inner product) over L2-normalized vectors, which is
mathematically equivalent to cosine similarity -- the standard choice for
sentence-embedding search.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, INDEX_PATH, METADATA_PATH, EMBED_MODEL_NAME
from chunking import load_and_chunk_documents

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")


def build_index(docs_dir: str = None):
    records = load_and_chunk_documents(docs_dir) if docs_dir else load_and_chunk_documents()
    if not records:
        raise ValueError("No chunks found -- check your docs directory.")

    texts = [r["text"] for r in records]
    vectors = embed_texts(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine similarity via normalized inner product
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(records)} chunks -> {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
