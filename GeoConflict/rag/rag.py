"""
RAG module for geographic conflict concept retrieval.

Responsibilities:
- Load structured geographic knowledge from JSON
- Build FAISS indices per geographic category
- Retrieve top-K semantically distant (conflicting) concepts
- Return human-readable candidates for LLM prompt construction

This module:
- Does NOT depend on LLMs
- Does NOT construct prompts
- Does NOT decide which candidate is finally used
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import faiss
import numpy as np


class GeographyRAG:
    """
    Retrieval-Augmented Generator (RAG) for geographic conflict concepts.

    Design:
    - One FAISS index per category (architecture / climate / vegetation / landmark)
    - Use normalized embeddings + inner product
    - Farthest retrieval implemented via querying with negative vectors
    """

    def __init__(
        self,
        knowledge_path: str,
        embed_fn,
    ):
        """
        Args:
            knowledge_path:
                Path to geography_knowledge.json

            embed_fn:
                A callable that maps List[str] -> np.ndarray
                Shape: (N, D)
                Example:
                    embed_fn(["dense tropical rainforest", "arid desert climate"])
        """
        self.knowledge_path = Path(knowledge_path)
        self.embed_fn = embed_fn

        # Loaded raw knowledge
        self.knowledge: Dict[str, List[Dict[str, Any]]] = {}

        # FAISS indices per category
        self.indices: Dict[str, faiss.Index] = {}

        # Mapping from FAISS id -> knowledge entry
        self.id_maps: Dict[str, List[Dict[str, Any]]] = {}

        self._load_knowledge()
        self._build_indices()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_knowledge(self):
        """
        Load geographic knowledge from JSON file.
        """
        if not self.knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {self.knowledge_path}")

        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            self.knowledge = json.load(f)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_indices(self):
        """
        Build a FAISS index for each geographic category.
        """
        for category, entries in self.knowledge.items():
            if len(entries) == 0:
                continue

            # Use description field for embeddings
            texts = [e["description"] for e in entries]

            # Compute embeddings
            embeddings = self.embed_fn(texts)  # (N, D)

            if not isinstance(embeddings, np.ndarray):
                raise TypeError("embed_fn must return a numpy array")

            # Normalize embeddings for cosine similarity via inner product
            embeddings = self._normalize(embeddings)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            self.indices[category] = index
            self.id_maps[category] = entries

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve_conflicting_concepts(
        self,
        category: str,
        query_text: str,
        top_k: int = 5,
        require_visualizable: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K semantically distant concepts within the same category.

        Args:
            category:
                One of the geography categories, e.g. "vegetation"

            query_text:
                The text span to be replaced (used as embedding query)

            top_k:
                Number of candidates to retrieve

            require_visualizable:
                If True, filter out entries with visualizable == False

        Returns:
            A list of knowledge entries (dicts)
        """
        if category not in self.indices:
            raise ValueError(f"Unknown or empty category: {category}")

        # Embed query
        q = self.embed_fn([query_text])  # (1, D)
        q = self._normalize(q)

        # Search with negative vector to get farthest neighbors
        scores, indices = self.indices[category].search(-q, top_k)

        candidates = []
        for idx in indices[0]:
            entry = self.id_maps[category][idx]
            if require_visualizable and not entry.get("visualizable", True):
                continue
            candidates.append(entry)

        return candidates

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors along the last dimension.
        """
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-8, None)

    # ------------------------------------------------------------------
    # Formatting helpers (optional but useful)
    # ------------------------------------------------------------------

    @staticmethod
    def format_candidates_for_prompt(
        candidates: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved candidates into a human-readable string
        suitable for LLM prompts.
        """
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. {c['name']}\n"
                f"   - {c['description']}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    def embed_fn(texts):
        # placeholder: replace with OpenAI / sentence-transformers / etc.
        return np.random.randn(len(texts), 768)

    rag = GeographyRAG(
        knowledge_path="prompts/geography_knowledge.json",
        embed_fn=embed_fn
    )

    candidates = rag.retrieve_conflicting_concepts(
        category="vegetation",
        query_text="deciduous trees and grass lawns",
        top_k=5
    )

    print(rag.format_candidates_for_prompt(candidates))
