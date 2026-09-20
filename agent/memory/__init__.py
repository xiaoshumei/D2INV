"""
D2INV Agent Long-term Memory package.

Exports:
  - SimpleEmbedder: dependency-free text embedding
  - VectorStore: numpy-backed cosine-similarity vector store
  - LongTermMemory: cross-session semantic memory manager
"""

from agent.memory.embedder import SimpleEmbedder
from agent.memory.vector_store import VectorStore, VectorEntry
from agent.memory.long_term import LongTermMemory

__all__ = ["SimpleEmbedder", "VectorStore", "VectorEntry", "LongTermMemory"]