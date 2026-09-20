"""
Vector Store for D2INV Agent — Phase 3.

A lightweight, numpy-backed vector store that supports:
 - insert / upsert vectors with metadata
 - cosine-similarity search (top-k)
 - persistence to disk (JSON)

No external database or vector-library dependency.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VectorEntry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """In-memory vector store with cosine-similarity search and JSON persistence."""

    def __init__(self, persist_path: Optional[str] = None, dim: Optional[int] = None):
        self._entries: Dict[str, VectorEntry] = {}
        self._matrix: Optional[np.ndarray] = None  # [N, dim], lazily updated
        self._dirty = False
        self._persist_path = persist_path
        self._dim = dim

        if persist_path and os.path.exists(persist_path):
            self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, entry_id: str, vector: List[float],
               metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._dim is None:
            self._dim = len(vector)
        else:
            assert len(vector) == self._dim, "Vector dimension mismatch"

        entry = VectorEntry(
            id=entry_id,
            vector=list(vector),
            metadata=metadata or {},
        )
        self._entries[entry_id] = entry
        self._dirty = True

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._dirty = True
            return True
        return False

    def get(self, entry_id: str) -> Optional[VectorEntry]:
        return self._entries.get(entry_id)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._matrix = None
        self._dirty = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: List[float], top_k: int = 5,
               min_score: float = 0.0) -> List[Tuple[VectorEntry, float]]:
        """
        Return top_k entries ranked by cosine similarity (descending).

        Args:
            query_vector: Query embedding.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity threshold (0.0 to 1.0).

        Returns:
            List of (VectorEntry, similarity_score) tuples.
        """
        if not self._entries:
            return []

        matrix, ids = self._get_matrix()
        if matrix is None:
            return []

        query = np.array(query_vector, dtype=np.float32)

        # Cosine similarity = dot(a,b) / (norm(a) * norm(b))
        # All stored embeddings are already normalised.
        scores = np.dot(matrix, query)

        # Filter and rank
        best_indices = np.argsort(-scores)
        results = []
        for idx in best_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            entry_id = ids[idx]
            results.append((self._entries[entry_id], score))
            if len(results) >= top_k:
                break

        return results

    def search_by_metadata(self, metadata_query: Dict[str, Any]) -> List[VectorEntry]:
        """Simple exact-match metadata filter (no vector search)."""
        results = []
        for entry in self._entries.values():
            if all(entry.metadata.get(k) == v for k, v in metadata_query.items()):
                results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = path or self._persist_path
        if not target:
            raise ValueError("No persist path configured")
        data = {
            "entries": {
                eid: {
                    "id": e.id,
                    "vector": e.vector,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for eid, e in self._entries.items()
            }
        }
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def _load(self) -> None:
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        for eid, raw in data.get("entries", {}).items():
            self._entries[eid] = VectorEntry(
                id=raw["id"],
                vector=raw["vector"],
                metadata=raw.get("metadata", {}),
                timestamp=raw.get("timestamp", 0),
            )
        self._dim = len(next(iter(self._entries.values())).vector) if self._entries else self._dim
        self._dirty = False

    def flush(self) -> None:
        if self._dirty and self._persist_path:
            self.save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_matrix(self) -> Tuple[Optional[np.ndarray], List[str]]:
        if self._dirty or self._matrix is None:
            if not self._entries:
                self._matrix = None
                return None, []

            ids = sorted(self._entries.keys())
            vecs = [self._entries[eid].vector for eid in ids]
            matrix = np.array(vecs, dtype=np.float32)

            # Normalise all rows so cosine-sim = dot product
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            matrix = matrix / norms

            self._matrix = matrix
            self._ids = ids
            self._dirty = False

        return self._matrix, self._ids