# RAG Pipeline with Automated Output Validation

A RAG (Retrieval-Augmented Generation) pipeline built from scratch, matching
the resume line: document chunking → embeddings → FAISS retrieval →
response generation → automated grounding validation + latency tracking.

## How the pieces fit together

```
sample_docs/*.txt
      │
      ▼
 chunking.py      → splits documents into overlapping text chunks
      │
      ▼
 embed_index.py   → embeds chunks (sentence-transformers) + builds a FAISS index
      │
      ▼
 retrieval.py     → embeds a query, finds the top-k nearest chunks in FAISS
      │
      ▼
 generate.py      → sends the question + retrieved chunks to an LLM
      │
      ▼
 validation.py    → checks each answer sentence is grounded in the retrieved
                     chunks, flags unsupported ones, and main.py times every stage
```

Every file's docstring explains the "why," not just the "what" — read them
in the order above; that's also a natural walkthrough for an interview.

## 1. Why each stage exists (quick reference)

- **Chunking**: embedding a whole document as one vector loses too much
  meaning; splitting into smaller pieces lets retrieval find just the
  relevant part. Overlap between chunks stops facts from being cut in half
  at a chunk boundary.
- **Embeddings**: turn text into vectors so "search by meaning" works —
  a query and a chunk can match even with zero words in common.
- **FAISS**: the standard library for fast nearest-neighbor search over
  those vectors; here it's a small in-memory index, but the same API
  scales to millions of vectors.
- **Generation**: the LLM is instructed to answer *only* from retrieved
  context — that constraint is what makes "grounding" a well-defined,
  checkable thing afterward.
- **Validation**: catches hallucination automatically — sentences in the
  answer that aren't actually backed by the retrieved chunks — and reports
  how long retrieval/generation/validation each took.

## 2. Setup

```bash
cd rag_pipeline
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or set the equivalent for your provider
```

(First run will download the `all-MiniLM-L6-v2` embedding model, ~80MB, once.)

## 3. Run it

```bash
# Step 1: build the FAISS index from sample_docs/
python main.py index

# Step 2: ask a question
python main.py query "What is a common failure mode in RAG systems?"
```

Output is JSON: the answer, which chunks were retrieved and their scores,
the grounding check (which sentences were flagged as unsupported), and how
long each pipeline stage took.

## 4. Use it on your own documents

Drop `.txt` files into `sample_docs/` (or point `DOCS_DIR` in `config.py`
at another folder), then re-run `python main.py index`.

## 5. Extending it (good next steps / talking points)

- Swap the cosine-similarity grounding check for a proper NLI
  (entailment/contradiction) model for stricter validation.
- Add PDF/DOCX loaders to `chunking.py` (currently `.txt` only).
- Swap `IndexFlatIP` for `IndexIVFFlat` or HNSW once the corpus grows large
  enough that a linear scan gets slow — explain the accuracy/speed
  trade-off in an interview.
- Log latency + grounding scores over time to spot regressions.

## 6. How to talk about this on your resume / in interviews

- "Built a RAG pipeline using document chunking, embeddings, and FAISS for
  context retrieval" → you can explain *why* each stage exists (see
  section 1), not just that you used the tools.
- "Added response validation to check answer grounding, flag unsupported
  outputs, and track latency" → be ready to explain the cosine-similarity
  grounding method, its threshold, and its known limitation (it's a
  semantic-similarity proxy, not true entailment) — and what you'd improve
  with more time (NLI-based grounding).
