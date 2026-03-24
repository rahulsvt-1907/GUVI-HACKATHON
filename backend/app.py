"""
Flask REST API that exposes FAISS-powered voice similarity search.

Endpoints
---------
GET  /health           – liveness probe
POST /store            – extract features from an audio file and store in FAISS
POST /search           – find the most similar past detections
POST /store-vector     – store a pre-computed feature vector (128 floats)
POST /search-vector    – search with a pre-computed feature vector
DELETE /reset          – wipe the entire FAISS index (dev / testing only)
GET  /stats            – total number of vectors in the index
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from faiss_service import FAISSVoiceService
from audio_features import extract_features

app = Flask(__name__)
CORS(app)

_svc = FAISSVoiceService()


# ── helpers ────────────────────────────────────────────────────────────────

def _meta_from_form() -> dict:
    return {
        "classification": request.form.get("classification", "UNKNOWN"),
        "confidence_score": float(request.form.get("confidence_score", 0.0)),
        "language": request.form.get("language", "Unknown"),
        "explanation": request.form.get("explanation", ""),
        "filename": "",
        "extra": {},
    }


def _meta_from_json() -> dict:
    data = request.get_json(force=True) or {}
    return {
        "classification": data.get("classification", "UNKNOWN"),
        "confidence_score": float(data.get("confidence_score", 0.0)),
        "language": data.get("language", "Unknown"),
        "explanation": data.get("explanation", ""),
        "filename": data.get("filename", ""),
        "extra": data.get("extra", {}),
    }


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "total_vectors": _svc.total()})


@app.get("/stats")
def stats():
    return jsonify({"total_vectors": _svc.total()})


@app.post("/store")
def store():
    """
    Multipart form upload:
      - file          (required) – audio file (MP3, WAV, …)
      - classification (optional) – "AI_GENERATED" | "HUMAN"
      - confidence_score (optional) – float 0–1
      - language       (optional) – e.g. "Tamil"
      - explanation    (optional) – free-text reason
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided (field: 'file')"}), 400

    audio_bytes = request.files["file"].read()
    filename = request.files["file"].filename or ""

    try:
        vector = extract_features(audio_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Feature extraction failed: {exc}"}), 500

    meta = _meta_from_form()
    meta["filename"] = filename

    record_id = _svc.add(vector=vector, **meta)
    return jsonify({"id": record_id, "message": "Stored successfully"}), 201


@app.post("/search")
def search():
    """
    Multipart form upload:
      - file   (required) – audio file to query
      - top_k  (optional, default 5) – number of neighbours
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided (field: 'file')"}), 400

    audio_bytes = request.files["file"].read()
    top_k = int(request.form.get("top_k", 5))

    try:
        vector = extract_features(audio_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Feature extraction failed: {exc}"}), 500

    results = _svc.search(vector, top_k=top_k)
    return jsonify({"results": results, "total_searched": _svc.total()})


@app.post("/store-vector")
def store_vector():
    """
    JSON body:
      - vector          (required) – list of 128 floats
      - classification  (optional)
      - confidence_score (optional)
      - language        (optional)
      - explanation     (optional)
      - filename        (optional)
      - extra           (optional)
    """
    data = request.get_json(force=True) or {}
    if "vector" not in data:
        return jsonify({"error": "Missing 'vector' field"}), 400

    vector = np.array(data["vector"], dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != 128:
        return jsonify({"error": "'vector' must be a list of exactly 128 floats"}), 422

    meta = _meta_from_json()
    record_id = _svc.add(vector=vector, **meta)
    return jsonify({"id": record_id, "message": "Stored successfully"}), 201


@app.post("/search-vector")
def search_vector():
    """
    JSON body:
      - vector  (required) – list of 128 floats
      - top_k   (optional, default 5)
    """
    data = request.get_json(force=True) or {}
    if "vector" not in data:
        return jsonify({"error": "Missing 'vector' field"}), 400

    vector = np.array(data["vector"], dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != 128:
        return jsonify({"error": "'vector' must be a list of exactly 128 floats"}), 422

    top_k = int(data.get("top_k", 5))
    results = _svc.search(vector, top_k=top_k)
    return jsonify({"results": results, "total_searched": _svc.total()})


@app.delete("/reset")
def reset():
    _svc.reset()
    return jsonify({"message": "Index cleared"})


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
