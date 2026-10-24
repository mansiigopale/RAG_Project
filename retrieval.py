"""
Stage 3: Retrieval.

Given a query, embed it with the SAME model used for the document chunks
(embeddings from different models aren't comparable), then ask FAISS for
the top_k nearest chunk vectors. We return chunk text + source + similarity
score -- the score is used later by the validation stage.
"""

import json
import faiss

from config import INDEX_PATH, METADATA_PATH, TOP_K
from embed_index import embed_texts


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve(query: str, top_k: int = TOP_K, index=None, metadata=None):
    if index is None or metadata is None:
        index, metadata = load_index()

    q_vec = embed_texts([query])
    scores, idxs = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        rec = metadata[idx]
        results.append({
            "text": rec["text"],
            "source": rec["source"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    q = "example question about your documents"
    for r in retrieve(q):
        print(f"[{r['score']:.3f}] ({r['source']}) {r['text'][:120]}...")
