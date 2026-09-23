"""Stratified Dataset Partitioning Pipeline for RAG-RL.

Partitions data/processed/rag_clean.parquet into:
- train.parquet (80%): Training data for SFT and RL policy learning
- val.parquet   (10%): Validation data for hyperparameter tuning & reward checkpointing
- test.parquet  (10%): Internal held-out test split for offline metrics

Uses stratified sampling across all 6 RAG taxonomy topics with a fixed random seed
for strict cross-machine reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.partition")


def stratified_split(
    df: pl.DataFrame,
    stratify_col: str = "taxonomy_topic",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Perform deterministic stratified split across topic categories."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    random.seed(seed)
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    unique_categories = df[stratify_col].unique().to_list()
    logger.info(f"Stratifying across {len(unique_categories)} categories: {unique_categories}")

    for cat in unique_categories:
        cat_df = df.filter(pl.col(stratify_col) == cat)
        cat_rows = cat_df.to_dicts()
        # Shuffle deterministically
        random.shuffle(cat_rows)

        n = len(cat_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remainder goes to test to ensure exact preservation of all rows
        cat_train = cat_rows[:n_train]
        cat_val = cat_rows[n_train:n_train + n_val]
        cat_test = cat_rows[n_train + n_val:]

        train_rows.extend(cat_train)
        val_rows.extend(cat_val)
        test_rows.extend(cat_test)

    # Final shuffle of each split
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    train_df = pl.DataFrame(train_rows, schema=df.schema)
    val_df = pl.DataFrame(val_rows, schema=df.schema)
    test_df = pl.DataFrame(test_rows, schema=df.schema)

    return train_df, val_df, test_df


def run_partition(
    input_parquet: Path,
    output_dir: Path,
    report_path: Path,
    stratify_col: str = "taxonomy_topic",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute stratified partitioning and export splits and audit report."""
    logger.info(f"Loading curated dataset from: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(input_parquet)
    total_rows = len(df)
    logger.info(f"Total curated rows to partition: {total_rows:,}")

    train_df, val_df, test_df = stratified_split(
        df=df,
        stratify_col=stratify_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.write_parquet(train_path, compression="zstd")
    val_df.write_parquet(val_path, compression="zstd")
    test_df.write_parquet(test_path, compression="zstd")

    def get_distribution(split_df: pl.DataFrame) -> Dict[str, int]:
        counts = split_df[stratify_col].value_counts().to_dicts()
        return {item[stratify_col]: item["count"] for item in counts}

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_records": total_rows,
        "splits": {
            "train": {
                "count": len(train_df),
                "pct": round(len(train_df) / total_rows * 100, 2),
                "file": str(train_path),
                "size_bytes": train_path.stat().st_size,
                "taxonomy_distribution": get_distribution(train_df),
            },
            "val": {
                "count": len(val_df),
                "pct": round(len(val_df) / total_rows * 100, 2),
                "file": str(val_path),
                "size_bytes": val_path.stat().st_size,
                "taxonomy_distribution": get_distribution(val_df),
            },
            "test": {
                "count": len(test_df),
                "pct": round(len(test_df) / total_rows * 100, 2),
                "file": str(test_path),
                "size_bytes": test_path.stat().st_size,
                "taxonomy_distribution": get_distribution(test_df),
            },
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Train split: {len(train_df):,} rows ({len(train_df)/total_rows*100:.1f}%) -> {train_path}")
    logger.info(f"Val split:   {len(val_df):,} rows ({len(val_df)/total_rows*100:.1f}%) -> {val_path}")
    logger.info(f"Test split:  {len(test_df):,} rows ({len(test_df)/total_rows*100:.1f}%) -> {test_path}")
    logger.info(f"Partition report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Stratified dataset partitioning for RAG-RL.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/processed/rag_clean.parquet"),
        help="Path to curated input Parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save train, val, and test splits",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/dataset_partition_report.json"),
        help="Path to save partition report JSON",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-col", type=str, default="taxonomy_topic")

    args = parser.parse_args()

    run_partition(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        report_path=args.report_path,
        stratify_col=args.stratify_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
