"""
FAISS-based voice embedding service.

Provides vector storage and similarity search for voice detection results
using Facebook AI Similarity Search (FAISS).
"""

import os
import json
import numpy as np
import faiss
from dataclasses import dataclass, field, asdict
from typing import Optional


VECTOR_DIM = 128
INDEX_FILE = "voice_index.faiss"
METADATA_FILE = "voice_metadata.json"


@dataclass
class VoiceRecord:
    id: int
    classification: str        # "AI_GENERATED" | "HUMAN"
    confidence_score: float    # 0.0 – 1.0
    language: str
    explanation: str
    filename: str = ""
    extra: dict = field(default_factory=dict)


class FAISSVoiceService:
    """
    Manages a FAISS flat-L2 index for voice-embedding vectors.

    Each entry stores a 128-dimensional float32 feature vector alongside
    metadata (classification, confidence score, language, …).
    """

    def __init__(
        self,
        dim: int = VECTOR_DIM,
        index_path: str = INDEX_FILE,
        meta_path: str = METADATA_FILE,
    ) -> None:
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.metadata: list[VoiceRecord] = []

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()
        else:
            self.index = faiss.IndexFlatL2(dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        vector: np.ndarray,
        classification: str,
        confidence_score: float,
        language: str,
        explanation: str,
        filename: str = "",
        extra: Optional[dict] = None,
    ) -> int:
        """
        Add a voice embedding vector and its metadata to the index.

        Returns the assigned record ID.
        """
        vec = self._validate_vector(vector)
        record_id = len(self.metadata)
        self.index.add(vec)
        self.metadata.append(
            VoiceRecord(
                id=record_id,
                classification=classification,
                confidence_score=float(confidence_score),
                language=language,
                explanation=explanation,
                filename=filename,
                extra=extra or {},
            )
        )
        self._save()
        return record_id

    def search(
        self, vector: np.ndarray, top_k: int = 5
    ) -> list[dict]:
        """
        Find the *top_k* most similar stored vectors.

        Returns a list of dicts with ``record`` (metadata) and
        ``distance`` (L2 distance; lower = more similar).
        """
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        vec = self._validate_vector(vector)
        distances, indices = self.index.search(vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            rec = asdict(self.metadata[idx])
            results.append({"record": rec, "distance": float(dist)})
        return results

    def total(self) -> int:
        """Return the number of vectors stored in the index."""
        return self.index.ntotal

    def reset(self) -> None:
        """Clear the index and all metadata."""
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []
        for path in (self.index_path, self.meta_path):
            if os.path.exists(path):
                os.remove(path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in self.metadata], fh, indent=2)

    def _load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.metadata = [VoiceRecord(**r) for r in raw]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_vector(vector: np.ndarray) -> np.ndarray:
        """Ensure the vector is a (1, dim) float32 array."""
        vec = np.asarray(vector, dtype=np.float32).flatten()
        return vec.reshape(1, -1)
