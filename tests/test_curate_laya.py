"""Unit tests for Laya curation pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.clean_threads import CLEANED_THREAD_SCHEMA
from scripts.curate_laya import format_state, run_curate_laya


def test_format_state():
    row = {
        "title": "Optimizing RAG retrieval",
        "post_body": "This is a detailed post about HNSW indexes." * 20,
        "best_answer_body": "Here is the recommended configuration." * 20,
    }
    state = format_state(row, max_text_chars=50)
    assert state["title"] == "Optimizing RAG retrieval"
    assert len(state["post_body"]) <= 50
    assert len(state["best_answer"]) <= 50


def test_run_curate_laya_mocked(tmp_path: Path):
    input_parquet = tmp_path / "cleaned.parquet"
    output_parquet = tmp_path / "curated.parquet"
    report_path = tmp_path / "report.json"
    rejected_parquet = tmp_path / "rejected.parquet"

    # Create dummy cleaned thread data
    data = [
        # Thread 1: High quality RAG
        ("t1", "rag", "How to tune chunk size?", "dev1", "2024-01-01", 10, "Post text", 3, "c1", "ans1", 15, "Answer text", 4, "[]", "", 100, 500),
        # Thread 2: Non-RAG / Low quality
        ("t2", "rag", "Random career advice", "dev2", "2024-01-02", 2, "General rant", 1, "c2", "ans2", 1, "Not RAG", 2, "[]", "", 50, 200),
    ]
    batch_dict = {col: [row[i] for row in data] for i, col in enumerate(CLEANED_THREAD_SCHEMA.names)}
    table = pa.Table.from_pydict(batch_dict, schema=CLEANED_THREAD_SCHEMA)
    pq.write_table(table, str(input_parquet))

    # Mock Laya agent
    mock_agent = MagicMock()

    def mock_predict(state, questions):
        title = state.get("title", "")
        if "chunk size" in title:
            return {
                "answers": {
                    "is_rag_related": {"noul": 0.95},
                    "technical_quality": {"score": 1.7},
                    "taxonomy_topic": {"choice": "indexing", "confidence": 0.92},
                }
            }
        else:
            return {
                "answers": {
                    "is_rag_related": {"noul": 0.15},
                    "technical_quality": {"score": 0.2},
                    "taxonomy_topic": {"choice": "other", "confidence": 0.88},
                }
            }

    mock_agent.predict.side_effect = mock_predict

    with patch("laya.load", return_value=mock_agent):
        report = run_curate_laya(
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            report_path=report_path,
            rejected_parquet=rejected_parquet,
            rag_threshold=0.60,
            min_quality_score=0.50,
        )

    assert report["total_evaluated"] == 2
    assert report["curated_passed"] == 1
    assert report["curated_rejected"] == 1
    assert report["taxonomy_distribution"]["indexing"] == 1
    assert report["taxonomy_distribution"]["other"] == 1

    out_table = pq.read_table(output_parquet)
    assert out_table.num_rows == 1
    row = out_table.to_pylist()[0]
    assert row["thread_id"] == "t1"
    assert row["taxonomy_topic"] == "indexing"
    assert row["curation_passed"] is True
