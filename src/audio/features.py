"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Feature Extraction
  88 clinically-grounded acoustic features per audio file
═══════════════════════════════════════════════════════════════

Feature groups:
  MFCC (26)          – timbre / vocal-tract shape
  Delta MFCC (13)    – temporal dynamics of timbre
  Pitch (5)          – fundamental frequency (Parselmouth)
  Voice Quality (7)  – jitter, shimmer, HNR (clinical gold)
  Formants (5)       – vowel space / articulation
  Energy (5)         – loudness & intensity
  Spectral (10)      – frequency-domain descriptors
  Speech Patterns (5)– rate, pauses, ZCR
  Chroma (12)        – tonal content
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import librosa
import parselmouth
from parselmouth.praat import call

from . import config as C


# ═════════════════════════════════════════════════════════════
#  Individual Feature Extractors
# ═════════════════════════════════════════════════════════════

def _safe(fn, default=0.0):
    """Run fn(); return default on any exception."""
    try:
        val = fn()
        return default if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
    except Exception:
        return default


# ── 1. MFCC + Delta ────────────────────────────────────────

def _mfcc_features(y: np.ndarray, sr: int) -> dict:
    """26 MFCC (mean+std per coeff) + 13 delta-MFCC means."""
    n_fft = min(C.N_FFT, max(512, len(y) // 4))
    n_fft = int(2 ** int(np.log2(n_fft)))

    mfccs  = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr,
                                   n_mfcc=C.N_MFCC, n_fft=n_fft)
    deltas = librosa.feature.delta(mfccs)

    feats = {}
    for i in range(C.N_MFCC):
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_{i}_std"]  = float(np.std(mfccs[i]))
    for i in range(C.N_MFCC):
        feats[f"delta_mfcc_{i}_mean"] = float(np.mean(deltas[i]))
    return feats                                              # 39 features


# ── 2. Pitch ───────────────────────────────────────────────

def _pitch_features(snd: parselmouth.Sound) -> dict:
    """Pitch (F0) statistics via Parselmouth."""
    pitch  = snd.to_pitch(pitch_floor=C.PITCH_FLOOR, pitch_ceiling=C.PITCH_CEILING)
    f0     = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    if len(voiced) == 0:
        return dict.fromkeys(
            ["pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range"], 0.0
        )
    return {
        "pitch_mean":  float(np.mean(voiced)),
        "pitch_std":   float(np.std(voiced)),
        "pitch_min":   float(np.min(voiced)),
        "pitch_max":   float(np.max(voiced)),
        "pitch_range": float(np.ptp(voiced)),
    }                                                         # 5 features


# ── 3. Voice Quality  (jitter / shimmer / HNR) ────────────

def _voice_quality_features(snd: parselmouth.Sound) -> dict:
    """Clinically validated voice quality markers."""
    pitch = call(snd, "To Pitch", 0.0, C.PITCH_FLOOR, C.PITCH_CEILING)
    pp    = call(snd, "To PointProcess (periodic, cc)",
                 C.PITCH_FLOOR, C.PITCH_CEILING)

    jit_args = (0, 0, 0.0001, 0.02, 1.3)
    shim_args = (0, 0, 0.0001, 0.02, 1.3, 1.6)

    feats = {
        "jitter_local":   _safe(lambda: call(pp, "Get jitter (local)", *jit_args)),
        "jitter_rap":     _safe(lambda: call(pp, "Get jitter (rap)", *jit_args)),
        "jitter_ppq5":    _safe(lambda: call(pp, "Get jitter (ppq5)", *jit_args)),
        "shimmer_local":  _safe(lambda: call([snd, pp], "Get shimmer (local)", *shim_args)),
        "shimmer_apq3":   _safe(lambda: call([snd, pp], "Get shimmer (apq3)", *shim_args)),
        "shimmer_apq5":   _safe(lambda: call([snd, pp], "Get shimmer (apq5)", *shim_args)),
    }

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, C.PITCH_FLOOR, 0.1, 1.0)
    feats["hnr_mean"] = _safe(lambda: call(harmonicity, "Get mean", 0, 0))

    return feats                                              # 7 features


# ── 4. Formants ────────────────────────────────────────────

def _formant_features(snd: parselmouth.Sound) -> dict:
    """
    First three formant frequencies.
    Vowel-space reduction = strong depression marker.
    """
    fmt = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    return {
        "f1_mean": _safe(lambda: call(fmt, "Get mean", 1, 0, 0, "hertz")),
        "f2_mean": _safe(lambda: call(fmt, "Get mean", 2, 0, 0, "hertz")),
        "f3_mean": _safe(lambda: call(fmt, "Get mean", 3, 0, 0, "hertz")),
        "f1_std":  _safe(lambda: call(fmt, "Get standard deviation", 1, 0, 0, "hertz")),
        "f2_std":  _safe(lambda: call(fmt, "Get standard deviation", 2, 0, 0, "hertz")),
    }                                                         # 5 features


# ── 5. Energy / Loudness ──────────────────────────────────

def _energy_features(y: np.ndarray) -> dict:
    """RMS energy and intensity descriptors."""
    rms = librosa.feature.rms(y=y.astype(np.float32))[0]
    return {
        "energy_mean": float(np.mean(rms)),
        "energy_std":  float(np.std(rms)),
        "energy_max":  float(np.max(rms)),
        "rms_mean":    float(np.sqrt(np.mean(y ** 2))),
        "rms_std":     float(np.std(rms)),
    }                                                         # 5 features


# ── 6. Spectral ───────────────────────────────────────────

def _spectral_features(y: np.ndarray, sr: int) -> dict:
    """Frequency-domain shape descriptors."""
    y32 = y.astype(np.float32)
    cent  = librosa.feature.spectral_centroid(y=y32, sr=sr)[0]
    bw    = librosa.feature.spectral_bandwidth(y=y32, sr=sr)[0]
    con   = librosa.feature.spectral_contrast(y=y32, sr=sr)
    flat  = librosa.feature.spectral_flatness(y=y32)[0]
    roll  = librosa.feature.spectral_rolloff(y=y32, sr=sr)[0]
    return {
        "spec_centroid_mean":  float(np.mean(cent)),
        "spec_centroid_std":   float(np.std(cent)),
        "spec_bandwidth_mean": float(np.mean(bw)),
        "spec_bandwidth_std":  float(np.std(bw)),
        "spec_contrast_mean":  float(np.mean(con)),
        "spec_contrast_std":   float(np.std(con)),
        "spec_flatness_mean":  float(np.mean(flat)),
        "spec_flatness_std":   float(np.std(flat)),
        "spec_rolloff_mean":   float(np.mean(roll)),
        "spec_rolloff_std":    float(np.std(roll)),
    }                                                         # 10 features


# ── 7. Speech Patterns ───────────────────────────────────

def _speech_pattern_features(y: np.ndarray, sr: int) -> dict:
    """Temporal speech characteristics."""
    y32 = y.astype(np.float32)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y32)[0]

    # Pause detection  (silence < 10 % peak)
    peak = np.max(np.abs(y))
    thr  = 0.10 * peak if peak > 0 else 1e-6
    silent = np.abs(y) < thr
    bounds = np.where(np.diff(silent.astype(int)))[0] + 1
    segs   = np.split(silent, bounds)
    pauses = [len(s) / sr for s in segs
              if len(s) > 0 and bool(s[0]) and (len(s) / sr) > 0.05]

    # Speech rate (energy-peak counting)
    rms   = librosa.feature.rms(y=y32)[0]
    peaks = []
    if len(rms) > 0:
        peaks, _ = scipy.signal.find_peaks(rms, distance=max(1, len(rms) // 10))

    dur = max(len(y) / sr, 1e-6)

    return {
        "zcr_mean":           float(np.mean(zcr)),
        "zcr_std":            float(np.std(zcr)),
        "pause_frequency":    len(pauses),
        "pause_duration_mean": float(np.mean(pauses)) if pauses else 0.0,
        "speech_rate":        float(len(peaks) / dur),
    }                                                         # 5 features


# ── 8. Chroma ─────────────────────────────────────────────

def _chroma_features(y: np.ndarray, sr: int) -> dict:
    """12 chroma-bin means (tonal content)."""
    chroma = librosa.feature.chroma_stft(y=y.astype(np.float32), sr=sr)
    return {f"chroma_{i}_mean": float(np.mean(chroma[i])) for i in range(12)}
                                                              # 12 features


# ═════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════

def extract_features(audio: np.ndarray, sr: int = C.SAMPLE_RATE) -> dict | None:
    """
    Extract ALL 88 features from a cleaned audio signal.

    Parameters
    ----------
    audio : 1-D float64 array of cleaned audio
    sr    : sample rate (default 16 kHz)

    Returns
    -------
    dict  with 88 feature names → float values, or None if empty.
    """
    if audio is None or len(audio) == 0:
        return None

    audio = audio.astype(np.float64)
    snd   = parselmouth.Sound(audio, sampling_frequency=sr)

    feats: dict[str, float] = {}

    # ── safe extraction per group ──
    for name, fn in [
        ("MFCC",            lambda: _mfcc_features(audio, sr)),
        ("Pitch",           lambda: _pitch_features(snd)),
        ("Voice Quality",   lambda: _voice_quality_features(snd)),
        ("Formants",        lambda: _formant_features(snd)),
        ("Energy",          lambda: _energy_features(audio)),
        ("Spectral",        lambda: _spectral_features(audio, sr)),
        ("Speech Patterns", lambda: _speech_pattern_features(audio, sr)),
        ("Chroma",          lambda: _chroma_features(audio, sr)),
    ]:
        try:
            feats.update(fn())
        except Exception as e:
            print(f"  ⚠ {name} extraction failed: {e}")

    return feats if feats else None


# ═════════════════════════════════════════════════════════════
#  Quick Smoke Test
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 55)
    print("  SMOKE TEST — 3-second random audio")
    print("═" * 55)

    dummy = np.random.randn(C.SAMPLE_RATE * 3)
    result = extract_features(dummy, C.SAMPLE_RATE)

    if result:
        print(f"\n  Total features: {len(result)}\n")
        for k, v in result.items():
            print(f"    {k:<30} {v:>12.6f}")
    else:
        print("  Extraction returned None.")
