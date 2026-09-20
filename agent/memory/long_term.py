"""
Long-Term Memory Manager for D2INV Agent — Phase 3.

Integrates the SimpleEmbedder and VectorStore to provide:
  - Semantic storage and retrieval of conversations, tool results, and facts
  - Cross-session persistence via disk-based VectorStore
  - Automatic saving and loading
  - Memory trimming (forget old / low-value entries)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.memory.embedder import SimpleEmbedder
from agent.memory.vector_store import VectorStore, VectorEntry


class LongTermMemory:
    """
    Manages long-term agent memory across sessions.

    Stores:
      - Conversation snippets (user messages, agent responses)
      - Tool execution results that were particularly insightful
      - Explicit facts the user has stated (preferences, domain knowledge)
      - Error resolutions (how an error was fixed)

    Memory entries are embedded and stored in a VectorStore for semantic retrieval.
    """

    DEFAULT_PATH = "./agent_storage/long_term_memory.json"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedder: Optional[SimpleEmbedder] = None,
        max_entries: int = 1000,
        embed_dim: int = 256,
    ):
        self._storage_path = storage_path or self.DEFAULT_PATH
        self._max_entries = max_entries
        self._embedder = embedder or SimpleEmbedder(dim=embed_dim)
        self._store = VectorStore(persist_path=self._storage_path, dim=embed_dim)
        self._entry_count = 0

    # ----- Public API -------------------------------------------------------

    def remember(
        self,
        content: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new memory entry.

        Args:
            content: The text to remember.
            memory_type: Category ('conversation','tool_result','fact','error_resolution').
            metadata: Additional key-value tags.

        Returns:
            The auto-generated memory ID.
        """
        meta = metadata or {}
        meta["type"] = memory_type
        meta["timestamp"] = time.time()

        vec = self._embedder.embed(content)
        mem_id = _make_id(content, meta)

        self._store.upsert(mem_id, vec, metadata=meta)
        self._entry_count += 1

        self._maybe_prune()
        self._store.flush()
        return mem_id

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_score: float = 0.15,
        max_age_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memory entries semantically relevant to the query.

        Returns a list of dicts with keys: id, content, metadata, score.

        Args:
            query: Natural language query string.
            top_k: Max results to return.
            memory_type: Optional filter by memory type.
            min_score: Minimum cosine similarity score.
            max_age_seconds: Ignore entries older than this.
        """
        query_vec = self._embedder.embed(query)
        results = self._store.search(query_vec, top_k=top_k * 2, min_score=min_score)

        output = []
        now = time.time()
        for entry, score in results:
            if memory_type and entry.metadata.get("type") != memory_type:
                continue
            if max_age_seconds is not None:
                age = now - entry.metadata.get("timestamp", 0)
                if age > max_age_seconds:
                    continue
            output.append({
                "id": entry.id,
                "content": entry.metadata.get("content", entry.id[:50]),
                "metadata": entry.metadata,
                "score": score,
            })
            if len(output) >= top_k:
                break
        return output

    def recall_by_fact(self, key: str) -> Optional[str]:
        """Retrieve a specific fact by exact metadata key."""
        results = self._store.search_by_metadata({"key": key, "type": "fact"})
        if results:
            return results[0].metadata.get("value")
        return None

    def forget(self, memory_id: str) -> None:
        """Remove a specific memory entry."""
        self._store.delete(memory_id)
        self._store.flush()

    def forget_by_type(self, memory_type: str) -> int:
        """Purge all memories of a given type. Returns count removed."""
        count = 0
        to_delete = []
        for eid, entry in self._store._entries.items():
            if entry.metadata.get("type") == memory_type:
                to_delete.append(eid)
        for eid in to_delete:
            self._store.delete(eid)
            count += 1
        if count:
            self._store.flush()
        return count

    def remember_fact(self, key: str, value: str) -> str:
        """Remember a key-value fact. Overwrites existing facts with the same key."""
        # Forget old fact with same key
        old = self._store.search_by_metadata({"key": key, "type": "fact"})
        for entry in old:
            self._store.delete(entry.id)
        return self.remember(
            content=f"{key}: {value}",
            memory_type="fact",
            metadata={"key": key, "value": value},
        )

    def stats(self) -> Dict[str, int]:
        """Return counts per memory type."""
        stats: Dict[str, int] = {}
        for entry in self._store._entries.values():
            t = entry.metadata.get("type", "unknown")
            stats[t] = stats.get(t, 0) + 1
        stats["total"] = sum(stats.values())
        return stats

    # ----- Internal ---------------------------------------------------------

    def _maybe_prune(self):
        if len(self._store._entries) <= self._max_entries:
            return
        # Remove oldest entries until under threshold
        sorted_entries = sorted(
            self._store._entries.values(), key=lambda e: e.timestamp
        )
        to_remove = len(sorted_entries) - self._max_entries
        for entry in sorted_entries[:to_remove]:
            self._store.delete(entry.id)


def _make_id(content: str, meta: Dict[str, Any]) -> str:
    """Generate a deterministic unique id from content + timestamp."""
    import hashlib

    raw = f"{content}|{meta.get('type','')}|{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]