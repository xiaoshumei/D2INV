"""
Simple Vector Embedder for D2INV Agent — Phase 3.

Uses a lightweight character-level n-gram hashing approach to generate
fixed-dimension embeddings without requiring external models (no sentence-transformers,
no OpenAI embeddings API dependency).

Embeddings are sparse float vectors suitable for cosine-similarity matching
in the vector store.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List


class SimpleEmbedder:
    """
    Generate fixed-size embeddings from text using character n-gram hashing.

    This is a fast, dependency-free alternative to heavy embedding models.
    Embeds text into a D-dimensional unit-normalised vector.

    Parameters:
        dim: Embedding dimension. Higher = more granular, lower = faster.
             Recommended: 128–512.
        ngram_range: Tuple of (min_n, max_n) character n-grams to hash.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple = (3, 6)):
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim
        self.ngram_range = ngram_range

    # ------------------------------------------------------------------
    def embed(self, text: str) -> List[float]:
        """Return a normalised embedding vector (list of floats)."""
        vec = self._sparse_vector(text)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_multiple(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    # ------------------------------------------------------------------
    def _sparse_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        if not text:
            return vec

        text_lower = text.lower().strip()
        for min_n, max_n in [self.ngram_range]:
            for n in range(min_n, max_n + 1):
                for i in range(len(text_lower) - n + 1):
                    ngram = text_lower[i : i + n]
                    h = _hash_to_int(ngram)
                    idx = h % self.dim
                    vec[idx] += 1.0
        # Apply sqrt scaling similar to TF-IDF smoothing
        vec = [math.sqrt(v) for v in vec]
        return vec


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _hash_to_int(s: str) -> int:
    """Deterministic integer hash of a string using SHA-256 truncated."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)