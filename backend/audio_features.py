"""
audio_features.py – Extract a 128-dim feature vector from an audio file.

The vector is built from standard spectro-temporal descriptors via librosa:
  - 40 MFCC means + 40 MFCC stds   = 80 dims
  - Spectral centroid mean / std     =  2 dims
  - Spectral rolloff  mean / std     =  2 dims
  - Zero-crossing rate mean / std    =  2 dims
  - RMS energy        mean / std     =  2 dims
  - Chroma mean (12 bins)            = 12 dims
  - Spectral contrast mean (7 bands) =  7 dims
  - Tempo                            =  1 dim
  ------------------------------------------
  Total                              = 108 dims  (zero-padded to 128)
"""

import io
import numpy as np
import librosa


TARGET_DIM = 128
SR = 22_050       # resample target (Hz)
N_MFCC = 40


def extract_features(audio_bytes: bytes) -> np.ndarray:
    """
    Return a (128,) float32 feature vector for *audio_bytes* (any format
    supported by librosa / soundfile, e.g. MP3, WAV, OGG).

    Raises ``ValueError`` if the audio cannot be decoded or is too short.
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

    if len(y) < SR * 0.5:          # less than 0.5 s → reject
        raise ValueError("Audio clip is too short (< 0.5 s).")

    feats: list[float] = []

    # ── MFCCs ────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats.extend(mfcc.mean(axis=1).tolist())
    feats.extend(mfcc.std(axis=1).tolist())

    # ── Spectral centroid ────────────────────────────────────────────
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats += [float(cent.mean()), float(cent.std())]

    # ── Spectral rolloff ─────────────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats += [float(rolloff.mean()), float(rolloff.std())]

    # ── Zero-crossing rate ───────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)
    feats += [float(zcr.mean()), float(zcr.std())]

    # ── RMS energy ───────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)
    feats += [float(rms.mean()), float(rms.std())]

    # ── Chroma ───────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(chroma.mean(axis=1).tolist())

    # ── Spectral contrast ─────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend(contrast.mean(axis=1).tolist())

    # ── Tempo ────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats.append(float(tempo))

    # ── Pad / truncate to TARGET_DIM ─────────────────────────────────
    arr = np.array(feats, dtype=np.float32)
    if len(arr) < TARGET_DIM:
        arr = np.pad(arr, (0, TARGET_DIM - len(arr)))
    else:
        arr = arr[:TARGET_DIM]

    # ── L2 normalise so cosine ≈ L2 in FAISS ─────────────────────────
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm

    return arr
