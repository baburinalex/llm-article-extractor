"""Step 2c: hand-rolled semantic search over data/*.txt (no vector DB yet)."""
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "nomic-embed-text"


def load_chunks(data_dir: str = "data") -> list[dict]:
    """Each paragraph of each file becomes one chunk."""
    chunks = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, para in enumerate(p.strip() for p in text.split("\n\n")):
            if para:
                chunks.append({"source": path.name, "para": i, "text": para})
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """Texts -> matrix of embedding vectors (one row per text)."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data])
    # normalize rows so that dot product == cosine similarity
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "ring resonator quality factor"

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from data/")

    chunk_vecs = embed([c["text"] for c in chunks])   # (N, 768)
    query_vec = embed([query])[0]                     # (768,)

    scores = chunk_vecs @ query_vec                   # cosine similarities
    top = np.argsort(scores)[::-1][:3]

    print(f"\nQuery: {query}\n")
    for rank, idx in enumerate(top, 1):
        c = chunks[idx]
        print(f"{rank}. [{scores[idx]:.3f}] {c['source']} (para {c['para']})")
        print(f"   {c['text'][:150]}...\n")


if __name__ == "__main__":
    main()
