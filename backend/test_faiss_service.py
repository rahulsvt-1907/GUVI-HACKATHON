"""
Unit tests for faiss_service.py and audio_features.py
"""

import io
import os
import sys
import tempfile
import unittest

import numpy as np

# ── make sure the backend package is importable ───────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from faiss_service import FAISSVoiceService


class TestFAISSVoiceService(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.svc = FAISSVoiceService(
            dim=128,
            index_path=os.path.join(self.tmpdir, "test.faiss"),
            meta_path=os.path.join(self.tmpdir, "test_meta.json"),
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _random_vec(self) -> np.ndarray:
        v = np.random.rand(128).astype(np.float32)
        return v / np.linalg.norm(v)

    # ── tests ─────────────────────────────────────────────────────────────

    def test_initial_state(self):
        self.assertEqual(self.svc.total(), 0)

    def test_add_and_total(self):
        self.svc.add(
            vector=self._random_vec(),
            classification="HUMAN",
            confidence_score=0.9,
            language="Tamil",
            explanation="Natural pitch variability detected.",
        )
        self.assertEqual(self.svc.total(), 1)

    def test_search_returns_results(self):
        vec = self._random_vec()
        self.svc.add(
            vector=vec,
            classification="AI_GENERATED",
            confidence_score=0.85,
            language="English",
            explanation="Robotic artifacts detected.",
        )
        results = self.svc.search(vec, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("record", results[0])
        self.assertIn("distance", results[0])
        self.assertAlmostEqual(results[0]["distance"], 0.0, places=4)

    def test_search_empty_index(self):
        results = self.svc.search(self._random_vec(), top_k=5)
        self.assertEqual(results, [])

    def test_search_top_k_capped(self):
        for _ in range(3):
            self.svc.add(
                vector=self._random_vec(),
                classification="HUMAN",
                confidence_score=0.7,
                language="Hindi",
                explanation="",
            )
        results = self.svc.search(self._random_vec(), top_k=10)
        self.assertEqual(len(results), 3)  # capped at index size

    def test_persistence(self):
        vec = self._random_vec()
        self.svc.add(
            vector=vec,
            classification="HUMAN",
            confidence_score=0.6,
            language="Malayalam",
            explanation="Test persistence.",
        )
        # Reload from disk
        svc2 = FAISSVoiceService(
            dim=128,
            index_path=os.path.join(self.tmpdir, "test.faiss"),
            meta_path=os.path.join(self.tmpdir, "test_meta.json"),
        )
        self.assertEqual(svc2.total(), 1)
        results = svc2.search(vec, top_k=1)
        self.assertEqual(results[0]["record"]["language"], "Malayalam")

    def test_reset(self):
        self.svc.add(
            vector=self._random_vec(),
            classification="HUMAN",
            confidence_score=0.5,
            language="Telugu",
            explanation="",
        )
        self.svc.reset()
        self.assertEqual(self.svc.total(), 0)
        self.assertEqual(self.svc.search(self._random_vec()), [])

    def test_metadata_fields(self):
        vec = self._random_vec()
        self.svc.add(
            vector=vec,
            classification="AI_GENERATED",
            confidence_score=0.95,
            language="Tamil",
            explanation="Perfect monotonic delivery.",
            filename="sample.mp3",
            extra={"source": "test"},
        )
        results = self.svc.search(vec, top_k=1)
        rec = results[0]["record"]
        self.assertEqual(rec["classification"], "AI_GENERATED")
        self.assertAlmostEqual(rec["confidence_score"], 0.95, places=5)
        self.assertEqual(rec["language"], "Tamil")
        self.assertEqual(rec["filename"], "sample.mp3")
        self.assertEqual(rec["extra"]["source"], "test")

    def test_vector_dimension_validation(self):
        """_validate_vector should accept any 1-D array and reshape to (1, dim)."""
        vec_list = list(np.random.rand(128).astype(np.float32))
        validated = FAISSVoiceService._validate_vector(np.array(vec_list))
        self.assertEqual(validated.shape, (1, 128))


if __name__ == "__main__":
    unittest.main()
