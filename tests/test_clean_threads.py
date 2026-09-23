"""Unit tests for thread cleaning and conversation stitching pipeline."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.clean_threads import (
    build_thread_conversation,
    is_bot_author,
    is_valid_body,
    run_clean_threads,
)


def test_is_bot_author():
    assert is_bot_author("AutoModerator") is True
    assert is_bot_author("remindmebot") is True
    assert is_bot_author("rag_developer") is False
    assert is_bot_author("[deleted]") is False


def test_is_valid_body():
    assert is_valid_body("Use ColBERTv2.", min_length=15) is False  # 14 chars
    assert is_valid_body("Try BM25 hybrid search.", min_length=15) is True
    assert is_valid_body("thanks!", min_length=15) is False
    assert is_valid_body("[deleted]", min_length=15) is False
    assert is_valid_body("[removed]", min_length=15) is False
    assert is_valid_body("", min_length=15) is False


def test_build_thread_conversation():
    post = {
        "id": "p1",
        "author": "alice_op",
        "title": "Why is hybrid search better than dense retrieval alone?",
        "body": "I noticed vector search misses exact keyword IDs like SKU numbers.",
        "score": 10,
        "created_at": "2024-01-01T10:00:00+00:00",
    }
    comments = [
        {
            "id": "c1",
            "author": "bob_expert",
            "score": 25,
            "created_at": "2024-01-01T10:15:00+00:00",
            "body": "Because dense embeddings map semantic similarity, but fail on rare lexical matches.",
        },
        {
            "id": "c2",
            "author": "alice_op",
            "score": 5,
            "created_at": "2024-01-01T10:20:00+00:00",
            "body": "That makes total sense. What fusion method do you recommend?",
        },
        {
            "id": "c3",
            "author": "charlie_dev",
            "score": 18,
            "created_at": "2024-01-01T10:30:00+00:00",
            "body": "Use Reciprocal Rank Fusion (RRF) with constant k=60.",
        },
    ]

    messages, transcript, best_ans = build_thread_conversation(post, comments)

    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert "hybrid search" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert "dense embeddings" in messages[1]["content"]
    assert messages[2]["role"] == "user"  # OP reply recognized as user
    assert "fusion method" in messages[2]["content"]
    assert messages[3]["role"] == "assistant"
    assert "Reciprocal Rank Fusion" in messages[3]["content"]

    # Best answer should be c1 (score 25)
    assert best_ans["id"] == "c1"
    assert best_ans["score"] == 25
    assert best_ans["author"] == "bob_expert"

    # Transcript should contain title and author
    assert "# Why is hybrid search" in transcript
    assert "bob_expert" in transcript


def test_run_clean_threads_synthetic(tmp_path: Path):
    input_parquet = tmp_path / "ingested.parquet"
    output_parquet = tmp_path / "cleaned.parquet"
    report_path = tmp_path / "report.json"

    # Create dummy ingested records
    schema = pa.schema([
        ("source", pa.string()),
        ("community", pa.string()),
        ("id", pa.string()),
        ("parent_id", pa.string()),
        ("post_id", pa.string()),
        ("author", pa.string()),
        ("created_at", pa.string()),
        ("title", pa.string()),
        ("body", pa.string()),
        ("score", pa.int64()),
    ])

    data = [
        # Pre-2023 record (should be filtered)
        ("community_forum", "rag", "old1", "", "old1", "user1", "2015-01-01T00:00:00+00:00", "Old post", "Some old text here", 1),
        # Valid post
        ("community_forum", "rag", "p100", "", "p100", "user2", "2024-01-01T00:00:00+00:00", "How to evaluate RAG faithfulness?", "Need metric recommendations.", 10),
        # Valid comment for p100 from deleted user (should be preserved!)
        ("community_forum", "rag", "c100", "t3_p100", "p100", "[deleted]", "2024-01-01T01:00:00+00:00", "", "Use Ragas or TruLens faithfulness rubric.", 15),
        # Short comment (should be filtered)
        ("community_forum", "rag", "c101", "t3_p100", "p100", "user3", "2024-01-01T02:00:00+00:00", "", "lol thanks", 1),
        # Bot comment (should be filtered)
        ("community_forum", "rag", "c102", "t3_p100", "p100", "AutoModerator", "2024-01-01T00:01:00+00:00", "", "Be civil and follow the rules of the forum.", 1),
    ]

    batch_dict = {col: [row[i] for row in data] for i, col in enumerate(schema.names)}
    table = pa.Table.from_pydict(batch_dict, schema=schema)
    pq.write_table(table, str(input_parquet))

    report = run_clean_threads(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        report_path=report_path,
        min_date="2023-01-01",
        min_body_length=15,
    )

    assert report["total_cleaned_threads"] == 1
    assert report["filtered_pre_min_date"] == 1
    assert report["filtered_bots"] == 1
    assert report["filtered_invalid_body"] == 1  # "lol thanks" (< 15 chars)

    out_table = pq.read_table(output_parquet)
    assert out_table.num_rows == 1
    row = out_table.to_pylist()[0]
    assert row["thread_id"] == "p100"
    assert row["comment_count"] == 1
    assert row["best_answer_author"] == "[deleted]"
    assert "Ragas" in row["best_answer_body"]
