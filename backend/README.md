# VoxCheck AI – FAISS Voice Similarity Backend

A lightweight Python/Flask service that uses **[FAISS](https://github.com/facebookresearch/faiss)**
(Facebook AI Similarity Search) to store and retrieve voice embedding vectors, enabling fast
similarity search across past voice-detection results.

---

## What is FAISS?

FAISS is a library developed by Facebook AI Research for efficient similarity search and
clustering of dense vectors. It is ideal for finding the *k* nearest neighbours in
high-dimensional embedding spaces—exactly what is needed when comparing acoustic feature
vectors extracted from audio clips.

---

## How it fits into VoxCheck AI

```
Frontend (React/TypeScript)
        │  POST /store  (audio file + Gemini result metadata)
        │  POST /search (query audio file)
        ▼
Flask REST API  (app.py)
        │
        ▼
audio_features.py  ──►  128-dim L2-normalised feature vector
        │                (MFCCs, spectral centroid, chroma, …)
        ▼
faiss_service.py  ──►  FAISS IndexFlatL2
        │               (persisted to voice_index.faiss + voice_metadata.json)
        ▼
Similarity results  ──►  Top-K neighbours with distance + metadata
```

Each voice sample analysed by the Gemini AI front-end can optionally be stored in the FAISS
index. Future queries can then surface *similar* past detections, giving users contextual
information such as:

- "5 similar audio clips were previously classified as **AI_GENERATED** with ~90% confidence"
- "This clip's spectral profile matches 3 previous **HUMAN** samples in Tamil"

---

## Project Structure

```
backend/
├── app.py               # Flask REST API
├── faiss_service.py     # FAISS index wrapper (storage + search)
├── audio_features.py    # Librosa-based 128-dim feature extractor
├── requirements.txt     # Python dependencies
└── test_faiss_service.py  # Unit tests
```

---

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the server

```bash
python app.py
# Server starts on http://localhost:5000
```

Environment variables:

| Variable     | Default | Description             |
|--------------|---------|-------------------------|
| `PORT`       | `5000`  | TCP port to listen on   |
| `FLASK_DEBUG`| `false` | Enable debug / reloader |

---

## API Reference

### `GET /health`
Liveness probe. Returns the number of vectors currently in the index.

```json
{ "status": "ok", "total_vectors": 42 }
```

---

### `GET /stats`
Returns the total number of stored vectors.

```json
{ "total_vectors": 42 }
```

---

### `POST /store`
Upload an audio file. The backend extracts a 128-dim feature vector and stores it in the
FAISS index together with the supplied metadata.

**Content-Type:** `multipart/form-data`

| Field              | Required | Description                          |
|--------------------|----------|--------------------------------------|
| `file`             | ✅       | Audio file (MP3, WAV, OGG, …)        |
| `classification`   |          | `"AI_GENERATED"` or `"HUMAN"`        |
| `confidence_score` |          | Float 0–1 (from Gemini analysis)     |
| `language`         |          | e.g. `"Tamil"`, `"English"`          |
| `explanation`      |          | Free-text reason from Gemini         |

```json
{ "id": 7, "message": "Stored successfully" }
```

---

### `POST /search`
Upload a query audio file. Returns the *top-k* most similar stored vectors.

**Content-Type:** `multipart/form-data`

| Field  | Required | Description                             |
|--------|----------|-----------------------------------------|
| `file` | ✅       | Query audio file                        |
| `top_k`|          | Number of neighbours to return (default 5) |

```json
{
  "results": [
    {
      "record": {
        "id": 3,
        "classification": "AI_GENERATED",
        "confidence_score": 0.91,
        "language": "Tamil",
        "explanation": "Robotic artifacts detected.",
        "filename": "sample1.mp3",
        "extra": {}
      },
      "distance": 0.0412
    }
  ],
  "total_searched": 10
}
```

---

### `POST /store-vector`
Store a pre-computed 128-dim feature vector (JSON body).

```json
{
  "vector": [0.12, -0.04, ...],   // 128 floats
  "classification": "HUMAN",
  "confidence_score": 0.87,
  "language": "English",
  "explanation": "Natural variability detected."
}
```

---

### `POST /search-vector`
Search with a pre-computed 128-dim feature vector (JSON body).

```json
{
  "vector": [0.12, -0.04, ...],   // 128 floats
  "top_k": 5
}
```

---

### `DELETE /reset`  *(development only)*
Clear the entire FAISS index and all metadata.

```json
{ "message": "Index cleared" }
```

---

## Running Tests

```bash
python -m pytest test_faiss_service.py -v
```

Expected output:

```
test_faiss_service.py::TestFAISSVoiceService::test_add_and_total         PASSED
test_faiss_service.py::TestFAISSVoiceService::test_initial_state         PASSED
test_faiss_service.py::TestFAISSVoiceService::test_metadata_fields       PASSED
test_faiss_service.py::TestFAISSVoiceService::test_persistence           PASSED
test_faiss_service.py::TestFAISSVoiceService::test_reset                 PASSED
test_faiss_service.py::TestFAISSVoiceService::test_search_empty_index    PASSED
test_faiss_service.py::TestFAISSVoiceService::test_search_returns_results PASSED
test_faiss_service.py::TestFAISSVoiceService::test_search_top_k_capped   PASSED
test_faiss_service.py::TestFAISSVoiceService::test_vector_dimension_validation PASSED
9 passed in 0.19s
```
