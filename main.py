"""
End-to-end pipeline + CLI.

Usage:
    python main.py index                   # build the FAISS index from sample_docs/
    python main.py query "your question"   # ask a question against the index
"""

import sys
import json

from embed_index import build_index
from retrieval import load_index, retrieve
from generate import generate_answer
from validation import check_grounding, track_latency


def run_query(question: str):
    timings = {}

    with track_latency("retrieval", timings):
        index, metadata = load_index()
        chunks = retrieve(question, index=index, metadata=metadata)

    with track_latency("generation", timings):
        answer = generate_answer(question, chunks)

    with track_latency("validation", timings):
        grounding = check_grounding(answer, chunks)

    result = {
        "question": question,
        "answer": answer,
        "retrieved_chunks": [{"source": c["source"], "score": c["score"]} for c in chunks],
        "grounding": grounding,
        "latency_seconds": timings,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index":
        build_index()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print('Usage: python main.py query "your question"')
            sys.exit(1)
        run_query(sys.argv[2])
    else:
        print(__doc__)
