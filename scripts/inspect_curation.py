"""Inspect Laya Curation Results.

Prints a readable sample of curated threads and rejected threads along with
Laya's scores to verify semantic classification quality and accuracy.
"""

import sys
from pathlib import Path
import polars as pl

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def inspect_curation(
    curated_path: Path = Path("data/processed/rag_clean.parquet"),
    num_samples: int = 20,
):
    if not curated_path.exists():
        print(f"Error: {curated_path} does not exist.")
        return

    df = pl.read_parquet(curated_path)
    print("=" * 80)
    print(f"CURATED THREADS AUDIT (Total Passed: {len(df)})")
    print("=" * 80)

    sample_size = min(num_samples, len(df))
    for i in range(sample_size):
        row = df.row(i, named=True)
        title = row["title"]
        rag_prob = row["is_rag_related"]
        quality = row["technical_quality_score"]
        topic = row["taxonomy_topic"]
        conf = row["topic_confidence"]
        post_snip = row["post_body"][:140].replace("\n", " ")
        ans_snip = row["best_answer_body"][:140].replace("\n", " ")

        print(f"\n[{i+1}/{sample_size}] ID: {row['thread_id']}")
        print(f"  Title:     {title}")
        print(f"  RAG Prob:  {rag_prob:.3f} | Quality: {quality:.2f} | Topic: '{topic}' (conf: {conf:.2f})")
        print(f"  Post:      {post_snip}...")
        if ans_snip:
            print(f"  Best Ans:  {ans_snip}...")
        print("-" * 80)


if __name__ == "__main__":
    inspect_curation()
