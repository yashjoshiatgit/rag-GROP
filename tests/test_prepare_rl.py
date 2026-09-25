"""Unit tests for RL / GRPO dataset preparation."""

from pathlib import Path
import polars as pl
import pytest

from scripts.prepare_rl_dataset import format_rl_prompt, prepare_grpo_split


def test_format_rl_prompt():
    messages = format_rl_prompt("How to optimize HNSW?", "I need faster retrieval latency on 1M vectors.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "expert RAG systems architect" in messages[0]["content"]
    assert "<think>" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "How to optimize HNSW?" in messages[1]["content"]


def test_prepare_grpo_split(tmp_path: Path):
    input_parquet = tmp_path / "test_train.parquet"
    output_parquet = tmp_path / "test_grpo.parquet"

    # Create dummy input data
    records = [
        {
            "thread_id": "t1",
            "title": "Optimizing chunk overlap",
            "post_body": "What is the best overlap ratio for technical documentation?",
            "best_answer_body": "Use 10-20% chunk overlap to maintain context across chunk boundaries.",
            "taxonomy_topic": "chunking_and_indexing",
        },
        {
            "thread_id": "t2",
            "title": "Short",
            "post_body": "Hi",  # Total length < 20, should be dropped
            "best_answer_body": "Hello",
            "taxonomy_topic": "tooling_and_frameworks",
        }
    ]
    pl.DataFrame(records).write_parquet(input_parquet)

    stats = prepare_grpo_split(input_parquet, output_parquet, min_body_chars=20)

    assert stats["total_prompts"] == 1
    assert output_parquet.exists()

    df = pl.read_parquet(output_parquet)
    assert len(df) == 1
    assert df["thread_id"][0] == "t1"
    assert df["taxonomy_topic"][0] == "chunking_and_indexing"
    assert "10-20% chunk overlap" in df["reference_answer"][0]
