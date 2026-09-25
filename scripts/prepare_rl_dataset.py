"""Prepare RL / GRPO Training Dataset for RAG-RL.

Transforms curated discussion threads (data/processed/train.parquet and val.parquet)
into standardized conversational prompt-completion datasets ready for TRL's GRPOTrainer.

Each record contains:
- prompt: Formatted messages list ([{"role": "system", ...}, {"role": "user", ...}])
- taxonomy_topic: RAG domain area (passed to reward_rag_concept_density)
- reference_answer: Community-verified solution (passed to reward_grounding_alignment)
- thread_id: Thread identifier for tracking
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.prepare_rl_dataset")

SYSTEM_PROMPT = (
    "You are an expert RAG systems architect and researcher. "
    "When presented with a technical challenge or question, first reason through the problem step-by-step "
    "inside <think>...</think> tags, considering architectural trade-offs, failure modes, and underlying mechanisms. "
    "Then, provide your definitive, grounded technical recommendation inside <answer>...</answer> tags."
)


def format_rl_prompt(title: str, post_body: str) -> List[Dict[str, str]]:
    """Format title and body into chat messages list."""
    clean_title = (title or "").strip()
    clean_body = (post_body or "").strip()

    if clean_body and clean_body != clean_title:
        user_content = f"Title: {clean_title}\n\nTechnical Question / Context:\n{clean_body}"
    else:
        user_content = f"Technical Question:\n{clean_title}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prepare_grpo_split(
    input_parquet: Path,
    output_parquet: Path,
    min_body_chars: int = 20,
) -> Dict[str, Any]:
    """Process a single parquet split into GRPOTrainer format."""
    logger.info(f"Loading input parquet: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    df = pl.read_parquet(input_parquet)
    logger.info(f"Read {len(df)} rows from {input_parquet}")

    processed_records: List[Dict[str, Any]] = []

    for row in df.iter_rows(named=True):
        title = row.get("title") or ""
        post_body = row.get("post_body") or ""
        best_answer = row.get("best_answer_body") or ""
        topic = row.get("taxonomy_topic") or "retrieval"
        thread_id = row.get("thread_id") or ""

        # Ensure minimal viable technical problem text
        full_text = f"{title} {post_body}".strip()
        if len(full_text) < min_body_chars:
            continue

        prompt_messages = format_rl_prompt(title, post_body)

        processed_records.append({
            "thread_id": thread_id,
            "prompt": prompt_messages,
            "taxonomy_topic": topic,
            "reference_answer": best_answer,
            "title": title,
        })

    out_df = pl.DataFrame(processed_records)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_parquet)

    logger.info(f"Saved {len(out_df)} GRPO training prompts -> {output_parquet}")

    topic_counts = dict(out_df["taxonomy_topic"].value_counts().iter_rows())

    return {
        "source_file": str(input_parquet),
        "output_file": str(output_parquet),
        "total_prompts": len(out_df),
        "topic_distribution": topic_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for GRPO training.")
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Path to curated train.parquet",
    )
    parser.add_argument(
        "--val-input",
        type=Path,
        default=Path("data/processed/val.parquet"),
        help="Path to curated val.parquet",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/processed/grpo_train_dataset.parquet"),
        help="Path to output grpo_train_dataset.parquet",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/processed/grpo_val_dataset.parquet"),
        help="Path to output grpo_val_dataset.parquet",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/grpo_dataset_report.json"),
        help="Path to output report JSON",
    )

    args = parser.parse_args()

    train_stats = prepare_grpo_split(args.train_input, args.train_output)
    val_stats = prepare_grpo_split(args.val_input, args.val_output)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train": train_stats,
        "val": val_stats,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"GRPO dataset preparation report saved -> {args.report_path}")


if __name__ == "__main__":
    main()
