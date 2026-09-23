"""Unit tests for data ingestion pipeline."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.ingest import normalize_comment, normalize_post, run_ingestion


def test_normalize_post_valid():
    raw_post = {
        "id": "abc123",
        "title": "How to optimize RAG chunking?",
        "selftext": "I am experimenting with sentence transformers.",
        "author": "rag_dev",
        "created_utc": 1700000000,
        "score": 15,
        "community": "rag",
    }
    normalized = normalize_post(raw_post)
    assert normalized is not None
    assert normalized["id"] == "abc123"
    assert normalized["post_id"] == "abc123"
    assert normalized["parent_id"] == ""
    assert normalized["title"] == "How to optimize RAG chunking?"
    assert normalized["body"] == "I am experimenting with sentence transformers."
    assert normalized["author"] == "rag_dev"
    assert normalized["score"] == 15
    assert "2023" in normalized["created_at"]
    assert normalized["source"] == "community_forum"
    assert normalized["community"] == "rag"


def test_normalize_post_missing_id():
    raw_post = {"title": "Missing ID", "selftext": "No ID field"}
    assert normalize_post(raw_post) is None


def test_normalize_comment_valid():
    raw_comment = {
        "id": "comm456",
        "parent_id": "t3_abc123",
        "link_id": "t3_abc123",
        "body": "Use recursive character chunking with overlap.",
        "author": "expert_user",
        "created_utc": 1700003600,
        "score": 42,
        "community": "rag",
    }
    normalized = normalize_comment(raw_comment)
    assert normalized is not None
    assert normalized["id"] == "comm456"
    assert normalized["parent_id"] == "t3_abc123"
    assert normalized["post_id"] == "abc123"
    assert normalized["title"] == ""
    assert normalized["body"] == "Use recursive character chunking with overlap."
    assert normalized["author"] == "expert_user"
    assert normalized["score"] == 42


def test_run_ingestion_synthetic(tmp_path: Path):
    posts_path = tmp_path / "test_posts.jsonl"
    comments_path = tmp_path / "test_comments.jsonl"
    output_parquet = tmp_path / "output.parquet"
    report_path = tmp_path / "report.json"

    # Write synthetic posts
    posts_data = [
        {"id": "p1", "title": "Post 1", "selftext": "Content 1", "author": "u1", "created_utc": 1700000000, "score": 5},
        {"id": "p2", "title": "Post 2", "selftext": "[deleted]", "author": "[deleted]", "created_utc": 1700001000, "score": 0},
    ]
    with posts_path.open("w", encoding="utf-8") as f:
        for p in posts_data:
            f.write(json.dumps(p) + "\n")
        f.write("{malformed json line\n")

    # Write synthetic comments
    comments_data = [
        {"id": "c1", "parent_id": "t3_p1", "link_id": "t3_p1", "body": "Great answer", "author": "u2", "created_utc": 1700002000, "score": 3},
    ]
    with comments_path.open("w", encoding="utf-8") as f:
        for c in comments_data:
            f.write(json.dumps(c) + "\n")

    report = run_ingestion(
        posts_path=posts_path,
        comments_path=comments_path,
        output_parquet=output_parquet,
        report_path=report_path,
        batch_size=10,
    )

    assert report["total_records"] == 3
    assert report["posts"] == 2
    assert report["comments"] == 1
    assert report["malformed"] == 1
    assert report["deleted"] == 1
    assert output_parquet.exists()
    assert report_path.exists()

    table = pq.read_table(output_parquet)
    assert table.num_rows == 3
    assert "source" in table.column_names
    assert "post_id" in table.column_names
    assert "community" in table.column_names
