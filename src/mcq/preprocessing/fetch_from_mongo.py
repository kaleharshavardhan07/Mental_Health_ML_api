from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pymongo import MongoClient


DEFAULT_MONGO_HOST = "clusterproject.ijackeq.mongodb.net"
DEFAULT_DB_NAME = "mental_health_db"
DEFAULT_COLLECTION = "fake_aditya_tests"


def connect_mongo(mongo_uri: str | None = None) -> MongoClient:
    """Create a MongoDB client.

    For Atlas clusters, prefer providing the full SRV URI via env var:
      MONGODB_URI="mongodb+srv://<user>:<pass>@clusterproject.ijackeq.mongodb.net/?retryWrites=true&w=majority"
    """
    uri = mongo_uri or os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError(
            "MongoDB connection URI not provided. Set MONGODB_URI env var or pass --mongo-uri.\n"
            f"Expected Atlas host: {DEFAULT_MONGO_HOST}"
        )
    return MongoClient(uri)


def _load_disease_question_ids(config_path: Path, disease: str) -> list[int]:
    """Load disease-specific question IDs from a config JSON.

    Supports two common shapes:
    1) {"depression": [1,2,3], "anxiety": [16,17]}
    2) {"depression": [{"id": 1, ...}, {"id": 2, ...}], ...}
       (this matches your current `configs/mcq_questions.json`)
    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if disease not in data:
        raise KeyError(f"Disease '{disease}' not found in {config_path.as_posix()}")

    disease_entry = data[disease]
    if isinstance(disease_entry, list) and (not disease_entry or isinstance(disease_entry[0], int)):
        ids = [int(x) for x in disease_entry]
    elif isinstance(disease_entry, list) and disease_entry and isinstance(disease_entry[0], dict):
        ids = [int(q["id"]) for q in disease_entry if "id" in q]
    else:
        raise ValueError(
            f"Unsupported questions config format for disease '{disease}'. "
            "Expected list[int] or list[dict] containing 'id'."
        )

    ids = sorted(set(ids))
    if not ids:
        raise ValueError(f"No question IDs found for disease '{disease}' in {config_path.as_posix()}")
    return ids


def fetch_mcq_data(
    client: MongoClient,
    disease: str,
    db_name: str = DEFAULT_DB_NAME,
    collection_name: str = DEFAULT_COLLECTION,
) -> Iterable[dict[str, Any]]:
    """Fetch MCQ-completed documents for a selected disease."""
    coll = client[db_name][collection_name]
    query = {"testType": disease, "mcqCompleted": True}
    projection = {
        "_id": 0,
        "testType": 1,
        "mcqAnswers": 1,
        "isRealPatientData": 1
    }
    return coll.find(query, projection=projection)


def build_feature_vector(mcq_answers: list[dict[str, Any]], question_ids: list[int]) -> list[int]:
    """Convert mcqAnswers into a fixed-length vector ordered by question_id.

    Uses the existing `score` field directly (no text encoding).
    Missing questions are filled with -1.
    """
    score_by_qid: dict[int, int] = {}
    for a in mcq_answers or []:
        if a is None:
            continue
        qid = a.get("questionId")
        score = a.get("score")
        if qid is None or score is None:
            continue
        try:
            score_by_qid[int(qid)] = int(score)
        except (TypeError, ValueError):
            continue

    return [score_by_qid.get(int(qid), -1) for qid in question_ids]


def save_dataset(
    rows: list[list[int]],
    disease: str,
    question_ids: list[int],
    output_path: Path,
) -> Path:
    """Save per-disease dataset (legacy, used when --disease is specified)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"q{qid}" for qid in question_ids] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    return output_path


def save_unified_dataset(
    rows: list[tuple[list[int], str]],
    question_ids: list[int],
    disease_names: list[str],
    output_path: Path,
    include_label: bool = True,
) -> Path:
    """Save unified CSV with q1..qN + one-hot disease columns (+ optional label).

    Each row: (feature_vector, disease_name) or (feature_vector, disease_name, label).
    One-hot cols: 1 if row's disease matches, else 0.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"q{qid}" for qid in question_ids] + list(disease_names)
    if include_label and rows and len(rows[0]) == 3:
        columns = columns + ["label"]

    data: list[list[int | str]] = []
    for row_tuple in rows:
        if len(row_tuple) == 3:
            vec, disease, label = row_tuple
        else:
            vec, disease = row_tuple
            label = None
        one_hot = [1 if d == disease else 0 for d in disease_names]
        row = vec + one_hot
        if include_label and label is not None:
            row.append(label)
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    return output_path


def _get_disease_names(config_path: Path) -> list[str]:
    """Return disease names from config (keys that are lists of questions)."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return [k for k in data if isinstance(data[k], list)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and preprocess MCQ data from MongoDB into CSV.")
    p.add_argument(
        "--disease",
        default=None,
        help="If set, fetch only this disease and save per-disease CSV. Default: fetch all and save single CSV with one-hot.",
    )
    p.add_argument(
        "--questions-config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs" / "mcq_questions.json",
        help="JSON mapping disease -> questions (ids).",
    )
    p.add_argument("--mongo-uri", default=None, help="MongoDB URI. If omitted, uses MONGODB_URI env var.")
    p.add_argument("--db", default=DEFAULT_DB_NAME, help="Database name.")
    p.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: data/processed/mcq/all_diseases_dataset.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    client = connect_mongo(args.mongo_uri)

    if args.disease:
        # Legacy: single disease, per-disease CSV
        disease = args.disease.strip().lower()
        question_ids = _load_disease_question_ids(args.questions_config, disease)
        docs = fetch_mcq_data(client, disease=disease, db_name=args.db, collection_name=args.collection)
        rows: list[list[int]] = []
        for doc in docs:
            vec = build_feature_vector(doc.get("mcqAnswers", []), question_ids)
            label = 1 if doc.get("isRealPatientData") else 0
            rows.append(vec + [label])
        output_path = args.output or project_root / "data" / "processed" / "mcq" / f"{disease}_dataset.csv"
        saved = save_dataset(rows=rows, disease=disease, question_ids=question_ids, output_path=output_path)
        total = len(rows)
    else:
        # Unified: all diseases, single CSV with one-hot columns
        disease_names = _get_disease_names(args.questions_config)
        unified_rows: list[tuple[list[int], str, int]] = []

        for disease in disease_names:
            question_ids = _load_disease_question_ids(args.questions_config, disease)
            docs = fetch_mcq_data(client, disease=disease, db_name=args.db, collection_name=args.collection)
            for doc in docs:
                vec = build_feature_vector(doc.get("mcqAnswers", []), question_ids)
                label = 1 if doc.get("isRealPatientData") else 0
                unified_rows.append((vec, disease, label))

        # All diseases use same q1..q15 layout
        question_ids = _load_disease_question_ids(args.questions_config, disease_names[0])
        output_path = args.output or project_root / "data" / "processed" / "mcq" / "all_diseases_dataset.csv"
        saved = save_unified_dataset(
            rows=unified_rows,
            question_ids=question_ids,
            disease_names=disease_names,
            output_path=output_path,
            include_label=True,
        )
        total = len(unified_rows)

    print(f"Saved {total} rows to {saved.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

