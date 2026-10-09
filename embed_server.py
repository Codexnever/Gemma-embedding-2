"""
Local EmbeddingGemma 2 server for Rerank-Debugger.

Why this exists: google/embeddinggemma-2 isn't deployed on any HF
serverless Inference Provider yet, so the Next.js routes can't call
HF's router API for it. This runs the model locally via
sentence-transformers and exposes one endpoint that mirrors the
{ inputs: string | string[] } -> vector(s) shape the HF Inference API
used, so the Next.js side only needs a different URL, not different
parsing logic.

Setup:
    pip install fastapi uvicorn sentence-transformers

Run (keep this running in its own terminal alongside `npm run dev`):
    python embed_server.py

First run downloads the model weights (~1.2 GB) — that can take a
few minutes depending on your connection. After that it loads from
the local HF cache.

Request:
    POST /embed
    { "inputs": "some text" }                 -> [0.1, 0.2, ...]            (flat vector)
    { "inputs": ["text a", "text b"] }         -> [[0.1, ...], [0.2, ...]]  (list of vectors)

This matches the shape the existing route code already parses
(`Array.isArray(data[0]) ? data : [data]` / `Array.isArray(data[0]) ? data[0] : data`),
so no changes are needed there beyond pointing at this URL.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Union, List
import uvicorn
import sys

MODEL_NAME = "google/embeddinggemma-2"

print(f"Loading {MODEL_NAME} — this can take a while on first run...", file=sys.stderr)
model = SentenceTransformer(MODEL_NAME)
print("Model loaded. Ready to serve on http://localhost:8000", file=sys.stderr)

app = FastAPI()


class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed")
def embed(payload: EmbedRequest):
    try:
        is_single = isinstance(payload.inputs, str)
        texts = [payload.inputs] if is_single else payload.inputs

        if not texts:
            raise HTTPException(status_code=400, detail="inputs is empty.")

        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        if is_single:
            return vectors[0].tolist()

        return vectors.tolist()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)