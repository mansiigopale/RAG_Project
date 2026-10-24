"""
Stage 4: Generation.

We take the user's question plus the retrieved chunks and build a prompt
that instructs the LLM to answer ONLY from the provided context, and to
say so explicitly when the context doesn't contain the answer. This
instruction is what makes grounding-validation meaningful -- without it,
the model free-associates from its own training data and "grounding"
stops being measurable against your documents.

Swap in any provider you have a key for (OpenAI, local Ollama, etc.) --
the pipeline doesn't care which LLM sits here, as long as it returns text.
"""

import os
from anthropic import Anthropic
from config import LLM_MODEL

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are a RAG assistant. Answer the user's question using ONLY the "
    "context provided below. If the context does not contain enough "
    "information to answer, say so explicitly instead of guessing. "
    "Do not use outside knowledge."
)


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(question: str, chunks: list) -> str:
    prompt = build_prompt(question, chunks)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
