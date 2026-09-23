"""Unit tests for dataset partitioning pipeline."""

from pathlib import Path
import polars as pl
import pytest

from scripts.partition_dataset import stratified_split, run_partition


def test_stratified_split():
    # Construct a synthetic dataset across 3 categories
    records = []
    categories = ["retrieval", "chunking_and_indexing", "evaluation_and_benchmarks"]
    for cat in categories:
        for i in range(100):
            records.append({
                "thread_id": f"{cat}_{i}",
                "taxonomy_topic": cat,
                "title": f"Question {i} about {cat}",
                "rag_confidence": 0.85,
            })

    df = pl.DataFrame(records)
    train_df, val_df, test_df = stratified_split(
        df,
        stratify_col="taxonomy_topic",
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
    )

    # Check total rows preserved exactly
    assert len(train_df) + len(val_df) + len(test_df) == len(df) == 300
    assert len(train_df) == 240
    assert len(val_df) == 30
    assert len(test_df) == 30

    # Verify per-category stratification
    for cat in categories:
        train_cat_count = len(train_df.filter(pl.col("taxonomy_topic") == cat))
        val_cat_count = len(val_df.filter(pl.col("taxonomy_topic") == cat))
        test_cat_count = len(test_df.filter(pl.col("taxonomy_topic") == cat))

        assert train_cat_count == 80
        assert val_cat_count == 10
        assert test_cat_count == 10

    # Ensure no data leakage (disjoint sets)
    train_ids = set(train_df["thread_id"].to_list())
    val_ids = set(val_df["thread_id"].to_list())
    test_ids = set(test_df["thread_id"].to_list())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_run_partition_end_to_end(tmp_path: Path):
    input_parquet = tmp_path / "rag_clean.parquet"
    output_dir = tmp_path / "processed"
    report_path = tmp_path / "outputs" / "report.json"

    # Create dummy parquet
    records = [
        {"thread_id": f"t_{i}", "taxonomy_topic": "retrieval", "text": "abc"}
        for i in range(50)
    ] + [
        {"thread_id": f"t_{i+50}", "taxonomy_topic": "chunking_and_indexing", "text": "def"}
        for i in range(50)
    ]
    df = pl.DataFrame(records)
    df.write_parquet(input_parquet)

    report = run_partition(
        input_parquet=input_parquet,
        output_dir=output_dir,
        report_path=report_path,
        seed=42,
    )

    assert report["total_records"] == 100
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert (output_dir / "test.parquet").exists()
    assert report_path.exists()
