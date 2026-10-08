import os

# --- Directories ---
DOCS_DIR = "sample_docs"
INDEX_DIR = "index_store"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# --- Chunking ---
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # characters of overlap between consecutive chunks

# --- Embedding model ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, runs locally, no API cost

# --- Retrieval ---
TOP_K = 4

# --- Validation ---
GROUNDING_THRESHOLD = 0.55   # cosine similarity below this => unsupported claim

# --- LLM (generation) ---
LLM_MODEL = "claude-sonnet-4-5"   # swap for any provider you have a key for
