"""
ML Pipeline Scheduler
=====================
Polls MongoDB every POLL_INTERVAL_SECONDS for Test documents with
mlExecutionStatus = 'in_progress', then:

  1. MCQ model   — called once with the 15 mcqAnswers
  2. Audio model — called once per video (up to 8), probabilities averaged
  3. Text model  — called once per video (via speech-to-text), probabilities averaged

Final mlResults structure saved back to MongoDB:
{
  "mcq":   { "depression": 72.5, "anxiety": 18.0, "ocd": 5.0, "adhd": 4.5 },
  "audio": { "Normal": 0.05, "Depression": 0.78, ... },   # averaged probabilities
  "text":  { "disease": { "probability": 0.78, "has_disorder": True }, ... },
  "meta":  { "videoCount": 8, "audioSuccess": 6, "textSuccess": 7 }
}
"""

import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient

# ── Configuration ─────────────────────────────────────────────────────────────
MONGODB_URI="mongodb+srv://hashiman004:DMDh13da1fTDbyBY@clusterproject.ijackeq.mongodb.net/mental_health_db"
DB_NAME            = "mental_health_db"
COLLECTION_NAME    = "tests"
POLL_INTERVAL_SECS = 60          # how often to poll MongoDB
VIDEO_DOWNLOAD_TIMEOUT = 120     # seconds per video download


# ── MongoDB helpers ────────────────────────────────────────────────────────────

def _get_collection():
    """Return (MongoClient, Collection) — caller must close the client."""
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
    return client, client[DB_NAME][COLLECTION_NAME]


# ── Video download ─────────────────────────────────────────────────────────────

def _download_video(url: str, dest_path: str) -> bool:
    """Download a public Firebase Storage URL to dest_path. Returns True on success."""
    try:
        resp = requests.get(url, timeout=VIDEO_DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return True
    except Exception as exc:
        print(f"  [Scheduler] ✗ Download failed ({url[:60]}...): {exc}")
        return False


# ── Mean helpers ───────────────────────────────────────────────────────────────

def _mean_audio_probs(preds: list) -> dict:
    """
    Average a list of audio probability dicts.
    Each item: { "Normal": 0.05, "Depression": 0.87, ... }
    Returns:   { "Normal": 0.06, "Depression": 0.83, ... }
    """
    valid = [p for p in preds if p and isinstance(p, dict)]
    if not valid:
        return {}
    all_keys = set(k for p in valid for k in p)
    return {
        k: round(float(np.mean([p.get(k, 0.0) for p in valid])), 4)
        for k in all_keys
    }


def _mean_text_preds(preds: list) -> dict:
    """
    Average a list of text prediction dicts.
    Each item: { "disease": { "probability": float, "has_disorder": bool } }
    Returns the same structure with averaged probabilities.
    """
    valid = [p for p in preds if p and isinstance(p, dict)]
    if not valid:
        return {}
    all_diseases = set(d for p in valid for d in p)
    result = {}
    for disease in all_diseases:
        probs = [
            float(p[disease]["probability"])
            for p in valid
            if disease in p and "probability" in p.get(disease, {})
        ]
        if probs:
            mean_prob = round(float(np.mean(probs)), 4)
            result[disease] = {
                "probability": mean_prob,
                "has_disorder": mean_prob >= 0.5,
            }
    return result


def _mean_video_preds(preds: list) -> dict:
    """
    Average a list of video prediction dicts.
    Each item from predict_all_diseases()["predictions"]:
      { "depression": {"probability": float, "has_disorder": bool}, ... }
    Returns same structure with averaged probabilities.
    """
    valid = [p for p in preds if p and isinstance(p, dict)]
    if not valid:
        return {}
    all_diseases = set(d for p in valid for d in p)
    result = {}
    for disease in all_diseases:
        probs = [
            float(p[disease]["probability"])
            for p in valid
            if disease in p and p[disease].get("success", True)
            and "probability" in p.get(disease, {})
        ]
        if probs:
            mean_prob = round(float(np.mean(probs)), 4)
            result[disease] = {
                "probability": mean_prob,
                "has_disorder": mean_prob >= 0.5,
            }
    return result


# ── Single test processor ──────────────────────────────────────────────────────

def _process_test(test: dict, col) -> None:
    """Run all ML models for one test document and write results back to MongoDB."""
    test_id   = test["_id"]
    test_type = test.get("testType", "unknown")
    print(f"\n[Scheduler] ── Processing test {test_id} (type={test_type}) ──")

    # Temp directory for this test's videos
    temp_dir = Path(tempfile.mkdtemp(prefix=f"ml_{test_id}_"))

    try:
        ml_results = {}

        # ── 1. MCQ MODEL ──────────────────────────────────────────────────────
        try:
            from src.mcq.Final_Preprocessing.mcq_inference import predict as mcq_predict

            mcq_answers = test.get("mcqAnswers", [])
            if mcq_answers:
                mcq_result = mcq_predict(mcq_answers)
                ml_results["mcq"] = mcq_result
                print(f"  [MCQ] ✓ Result: {mcq_result}")
            else:
                print("  [MCQ] ⚠ No MCQ answers found — skipping.")
                ml_results["mcq"] = None
        except Exception as exc:
            print(f"  [MCQ] ✗ Error: {exc}")
            ml_results["mcq"] = None

        # ── 2. Load audio + text predictors (lazy, singleton-safe) ────────────
        audio_predictor = None
        text_predict    = None
        extract_text    = None

        try:
            from src.audio.predict import AudioPredictor
            from src.audio import config as C
            audio_predictor = AudioPredictor(model_dir=C.MODEL_DIR)
            print("  [Audio] ✓ Predictor loaded")
        except Exception as exc:
            print(f"  [Audio] ✗ Could not load predictor: {exc}")

        try:
            from src.text.predict import predict as _text_predict
            from src.text.video_to_text import extract_text_from_video as _extract_text
            text_predict = _text_predict
            extract_text = _extract_text
            print("  [Text]  ✓ Predictor loaded")
        except Exception as exc:
            print(f"  [Text]  ✗ Could not load predictor: {exc}")

        video_predictor = None
        try:
            from src.video.video_predictor import VideoPredictor
            video_predictor = VideoPredictor(model_dir="models/video_model")
            print("  [Video] ✓ Predictor loaded")
        except Exception as exc:
            print(f"  [Video] ✗ Could not load predictor: {exc}")

        # ── 3. Per-video inference ────────────────────────────────────────────
        interview_questions = test.get("interviewQuestions", [])
        audio_preds = []
        text_preds  = []
        video_preds = []

        for idx, iq in enumerate(interview_questions):
            video_url = iq.get("videoUrl", "")
            q_id      = iq.get("questionId", idx + 1)

            # Skip local fallback paths (Firebase upload may have failed earlier)
            if not video_url or not video_url.startswith("http"):
                print(f"  [Video] ⚠ Q{q_id}: No valid Firebase URL, skipping.")
                continue

            # Detect extension from URL
            ext = ".mp4"
            for candidate in (".webm", ".mov", ".mp4", ".avi", ".mkv"):
                if candidate in video_url:
                    ext = candidate
                    break
            video_path = str(temp_dir / f"q{q_id}{ext}")

            print(f"  [Video] Downloading Q{q_id}...")
            if not _download_video(video_url, video_path):
                continue

            # ── Audio model call ──────────────────────────────────────────────
            if audio_predictor:
                try:
                    audio_raw = audio_predictor.predict(video_path)
                    if audio_raw and "probabilities" in audio_raw:
                        audio_preds.append(audio_raw["probabilities"])
                        print(f"  [Audio] ✓ Q{q_id}: condition={audio_raw.get('condition')} "
                              f"conf={audio_raw.get('confidence', 0):.2%}")
                    else:
                        print(f"  [Audio] ✗ Q{q_id}: No prediction returned")
                except Exception as exc:
                    print(f"  [Audio] ✗ Q{q_id} error: {exc}")

            # ── Text model call (speech-to-text first) ────────────────────────
            if text_predict and extract_text:
                try:
                    txt_result = extract_text(video_path)
                    if txt_result.get("success") and txt_result.get("text", "").strip():
                        pred = text_predict(txt_result["text"])
                        if pred.get("success"):
                            text_preds.append(pred["predictions"])
                            print(f"  [Text]  ✓ Q{q_id}: extracted {len(txt_result['text'])} chars")
                        else:
                            print(f"  [Text]  ✗ Q{q_id}: model prediction failed")
                    else:
                        print(f"  [Text]  ⚠ Q{q_id}: No speech detected in video")
                except Exception as exc:
                    print(f"  [Text]  ✗ Q{q_id} error: {exc}")

            # ── Video (facial/LSTM) model ─────────────────────────────────
            if video_predictor:
                try:
                    vid_raw = video_predictor.predict_all_diseases(video_path)
                    if vid_raw.get("success") and "predictions" in vid_raw:
                        video_preds.append(vid_raw["predictions"])
                        print(f"  [Video] ✓ Q{q_id}: predictions for "
                              f"{list(vid_raw['predictions'].keys())}")
                    else:
                        print(f"  [Video] ✗ Q{q_id}: prediction failed")
                except Exception as exc:
                    print(f"  [Video] ✗ Q{q_id} error: {exc}")

            # Clean up downloaded video immediately to save disk space
            try:
                Path(video_path).unlink(missing_ok=True)
                # Also remove extracted wav if audio model created one
                wav_path = Path(video_path).with_suffix(".wav")
                if wav_path.exists():
                    wav_path.unlink()
            except Exception:
                pass

        # ── 4. Compute means ──────────────────────────────────────────────────
        audio_result = _mean_audio_probs(audio_preds)
        text_result  = _mean_text_preds(text_preds)
        video_result = _mean_video_preds(video_preds)

        print(f"\n  [Summary] Audio: {len(audio_preds)}/{len(interview_questions)} videos succeeded")
        print(f"  [Summary] Text:  {len(text_preds)}/{len(interview_questions)} videos succeeded")
        print(f"  [Summary] Video: {len(video_preds)}/{len(interview_questions)} videos succeeded")

        ml_results["audio"] = audio_result
        ml_results["text"]  = text_result
        ml_results["video"] = video_result
        ml_results["meta"]  = {
            "videoCount":    len(interview_questions),
            "audioSuccess":  len(audio_preds),
            "textSuccess":   len(text_preds),
            "videoSuccess":  len(video_preds),
        }

        # ── 5. Write back to MongoDB ──────────────────────────────────────────
        col.update_one(
            {"_id": test_id},
            {"$set": {
                "mlResults":        ml_results,
                "mlExecutionStatus": "success",
            }}
        )
        print(f"  [Scheduler] ✅ Test {test_id} → mlExecutionStatus = success")

    except Exception as exc:
        print(f"  [Scheduler] ❌ Fatal error for test {test_id}: {exc}")
        traceback.print_exc()
        try:
            col.update_one(
                {"_id": test_id},
                {"$set": {"mlExecutionStatus": "failed"}}
            )
        except Exception:
            pass

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Scheduler job ──────────────────────────────────────────────────────────────

def run_scheduler_job():
    """APScheduler entry point — called every POLL_INTERVAL_SECS."""
    print(f"\n[Scheduler] Polling MongoDB ({DB_NAME}.{COLLECTION_NAME}) "
          f"for in_progress tests...")

    if not MONGODB_URI:
        print("[Scheduler] ⚠ MONGODB_URI env var not set — skipping poll.")
        return

    client = None
    try:
        client, col = _get_collection()
        pending = list(col.find({"mlExecutionStatus": "in_progress"}))
        print(f"[Scheduler] Found {len(pending)} test(s) to process.")

        for test in pending:
            _process_test(test, col)

    except Exception as exc:
        print(f"[Scheduler] ✗ MongoDB error: {exc}")
        traceback.print_exc()
    finally:
        if client:
            client.close()


# ── Public API ─────────────────────────────────────────────────────────────────

def start_scheduler() -> BackgroundScheduler:
    """
    Create and start the APScheduler BackgroundScheduler.
    Call this once at FastAPI startup (via lifespan).
    Returns the scheduler instance so it can be shut down on exit.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_scheduler_job,
        trigger="interval",
        seconds=POLL_INTERVAL_SECS,
        id="ml_pipeline_job",
        replace_existing=True,
        max_instances=1,       # never run two jobs in parallel
        coalesce=True,         # merge missed runs into one
    )
    scheduler.start()
    print(f"[Scheduler] ✅ Started — polling every {POLL_INTERVAL_SECS}s.")
    return scheduler
