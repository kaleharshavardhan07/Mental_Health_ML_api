"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Preprocessing
  Load → Denoise → VAD → Normalise → Segment
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import webrtcvad
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from . import config as C


# ══════════════════════════════════════════════════════════════
#   Core Cleaning Functions
# ══════════════════════════════════════════════════════════════

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio → 16 kHz mono float32."""
    y, sr = librosa.load(str(path), sr=C.SAMPLE_RATE, mono=True)
    return y, sr


def denoise(audio: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Spectral-gated noise reduction."""
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=C.NOISE_REDUCE_PROP)


def voice_activity_detection(
    audio: np.ndarray,
    sr: int = C.SAMPLE_RATE,
    aggressiveness: int = C.VAD_AGGRESSIVENESS,
) -> np.ndarray:
    """Remove silence using WebRTC VAD. Returns only voiced frames."""
    frame_ms   = 30
    frame_size = int(sr * frame_ms / 1000)
    vad        = webrtcvad.Vad(aggressiveness)

    # float → int16
    pcm = (audio * 32767).astype(np.int16)
    pad = frame_size - (len(pcm) % frame_size)
    if pad < frame_size:
        pcm = np.pad(pcm, (0, pad), mode="constant")

    voiced = [
        pcm[i : i + frame_size]
        for i in range(0, len(pcm), frame_size)
        if vad.is_speech(pcm[i : i + frame_size].tobytes(), sr)
    ]

    if not voiced:
        return audio                       # nothing detected → keep original

    return np.concatenate(voiced).astype(np.float32) / 32767.0


def normalise_rms(audio: np.ndarray, target: float = C.TARGET_RMS) -> np.ndarray:
    """RMS amplitude normalisation."""
    rms = np.sqrt(np.mean(audio ** 2))
    return audio * (target / rms) if rms > 0 else audio


def preemphasis(audio: np.ndarray, coeff: float = C.PRE_EMPHASIS) -> np.ndarray:
    """High-frequency emphasis: y[t] = x[t] − α·x[t−1]."""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def clean_audio(path: str | Path) -> tuple[np.ndarray | None, int]:
    """
    Full cleaning pipeline for a single WAV file:
        Load → Denoise → VAD → RMS Norm → Pre-emphasis
    Returns (cleaned_array, sample_rate) or (None, sr) on failure.
    """
    try:
        y, sr = load_audio(path)
    except Exception as e:
        print(f"  ✗ load error: {e}")
        return None, C.SAMPLE_RATE

    if len(y) == 0:
        return None, sr

    y = denoise(y, sr)
    y = voice_activity_detection(y, sr)
    y = normalise_rms(y)
    y = preemphasis(y)
    return y, sr


# ══════════════════════════════════════════════════════════════
#   Audio Segmentation  (handle variable lengths)
# ══════════════════════════════════════════════════════════════

def _segment_single(
    audio: np.ndarray,
    sr: int,
    out_dir: Path,
    stem: str,
    max_sec: int   = C.SEGMENT_MAX,
    pad_sec: int   = C.SEGMENT_PAD,
    overlap: float = C.SEGMENT_OVERLAP,
    max_segments_per_file: int | None = C.SEGMENT_MAX_PER_FILE,
) -> tuple[int, str]:
    """
    Pad / keep / segment one audio array.
    Returns (n_segments, status_string).
    """
    dur = len(audio) / sr
    out_dir.mkdir(parents=True, exist_ok=True)

    if dur < pad_sec:
        # ── PAD short audio ──
        target = int(pad_sec * sr)
        padded = np.pad(audio, (0, target - len(audio)), mode="constant")
        sf.write(str(out_dir / f"{stem}.wav"), padded, sr)
        return 1, f"PAD  {dur:.1f}s → {pad_sec}s"

    if dur <= max_sec:
        # ── KEEP medium audio ──
        sf.write(str(out_dir / f"{stem}.wav"), audio, sr)
        return 1, f"KEEP {dur:.1f}s"

    # ── SEGMENT long audio ──
    seg_samples = int(max_sec * sr)
    stride      = int(max_sec * (1 - overlap) * sr)
    idx = 0
    for start in range(0, len(audio) - seg_samples + 1, stride):
        if max_segments_per_file is not None and idx >= max_segments_per_file:
            break
        seg = audio[start : start + seg_samples]
        if len(seg) == seg_samples:
            sf.write(str(out_dir / f"{stem}_seg{idx}.wav"), seg, sr)
            idx += 1

    capped = max_segments_per_file is not None and idx >= max_segments_per_file
    if capped:
        return idx, f"SPLIT {dur:.1f}s → {idx} segments (capped)"
    return idx, f"SPLIT {dur:.1f}s → {idx} segments"


def segment_directory(
    input_dir:  str | Path = C.RAW_AUDIO_DIR,
    output_dir: str | Path = C.SEGMENTED_DIR,
) -> dict:
    """
    Walk  input_dir/{condition}/*.wav  →  output_dir/{condition}/*.wav
    Returns per-condition statistics dict.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    stats: dict[str, dict] = defaultdict(lambda: {"pad": 0, "keep": 0, "split": 0, "segs": 0})

    wavs = list(input_dir.rglob("*.wav"))
    if not wavs:
        print(f"  ⚠  No WAV files in {input_dir}")
        return dict(stats)

    print(f"\n  Found {len(wavs)} files. Segmenting …\n")

    for wav in tqdm(wavs, desc="  Segment", unit="file"):
        rel   = wav.relative_to(input_dir)
        cond  = rel.parts[0]
        o_dir = output_dir / rel.parent

        y, sr = librosa.load(str(wav), sr=C.SAMPLE_RATE, mono=True)
        n, msg = _segment_single(y, sr, o_dir, wav.stem)

        if "PAD"  in msg: stats[cond]["pad"]  += 1
        elif "KEEP" in msg: stats[cond]["keep"] += 1
        else:
            stats[cond]["split"] += 1
            stats[cond]["segs"]  += n

    # ── Summary ──
    print(f"\n  {'Condition':<14} {'Padded':>7} {'Kept':>7} {'Split':>7} {'Segments':>9}")
    print("  " + "─" * 50)
    for cond in sorted(stats):
        s = stats[cond]
        print(f"  {cond:<14} {s['pad']:>7} {s['keep']:>7} {s['split']:>7} {s['segs']:>9}")

    return dict(stats)
