"""Thread Stitching and Deterministic Cleaning Pipeline for RAG-RL.

Transforms normalized records from data/interim/ingested.parquet into cleaned,
multi-turn conversation threads and structured discussion transcripts.

Features:
- Filters records created before 2023 (focuses on modern RAG era).
- Preserves valuable comments from deleted accounts (author '[deleted]') if body is intact.
- Filters out empty/deleted bodies or text shorter than 15 characters (drops '+1', 'lol', etc.).
- Filters out negative-score spam/trolls (score < 0).
- Filters out automated bot accounts (e.g., AutoModerator).
- Reconstructs reply tree into multi-turn conversations (OP as 'user', community as 'assistant').
- Generates formatted full discussion transcripts for domain pre-training.
- Identifies the highest-upvoted response for direct SFT/GRPO pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.clean_threads")

KNOWN_BOTS = {
    "automoderator",
    "remindmebot",
    "savevideobot",
    "sneakpeekbot",
    "haikusbot",
    "markdownbot",
    "qualityvote",
}

CLEANED_THREAD_SCHEMA = pa.schema([
    ("thread_id", pa.string()),
    ("community", pa.string()),
    ("title", pa.string()),
    ("post_author", pa.string()),
    ("post_created_at", pa.string()),
    ("post_score", pa.int64()),
    ("post_body", pa.string()),
    ("comment_count", pa.int64()),
    ("best_answer_id", pa.string()),
    ("best_answer_author", pa.string()),
    ("best_answer_score", pa.int64()),
    ("best_answer_body", pa.string()),
    ("conversation_turns", pa.int64()),
    ("conversation_json", pa.string()),  # JSON string of [{"role": ..., "content": ...}]
    ("discussion_transcript", pa.string()),  # Markdown formatted transcript
    ("total_word_count", pa.int64()),
    ("total_char_count", pa.int64()),
])


def is_bot_author(author: str) -> bool:
    """Check if author matches known automated bot patterns."""
    if not author:
        return False
    lower = author.strip().lower()
    if lower in KNOWN_BOTS:
        return True
    if lower.endswith("bot") and len(lower) > 3 and "_" in lower:
        return True
    return False


def is_valid_body(body: str, min_length: int = 15) -> bool:
    """Check if body contains actual text and meets min length."""
    if not body:
        return False
    stripped = body.strip()
    if stripped.lower() in ("[deleted]", "[removed]", "none", "null"):
        return False
    if len(stripped) < min_length:
        return False
    return True


def build_thread_conversation(
    post: Dict[str, Any],
    comments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
    """Build multi-turn conversation and transcript from post and comments."""
    post_author = post["author"]
    post_title = post["title"]
    post_body = post["body"]

    # Initial user prompt
    if post_body and post_body != post_title:
        prompt_content = f"{post_title}\n\n{post_body}".strip()
    else:
        prompt_content = post_title.strip()

    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_content}]
    transcript_lines: List[str] = [
        f"# {post_title}",
        f"**Author:** {post_author} | **Score:** {post['score']} | **Date:** {post['created_at']}",
        "",
        post_body if post_body else "*(No body text)*",
        "",
        "---",
        "## Discussion & Solutions",
        "",
    ]

    # Sort comments chronologically
    sorted_comments = sorted(comments, key=lambda c: c.get("created_at") or "")

    # Track best answer by score
    best_answer = {
        "id": "",
        "author": "",
        "score": -999999,
        "body": "",
    }

    for c in sorted_comments:
        c_score = int(c.get("score") or 0)
        c_body = c.get("body", "").strip()
        c_author = c.get("author", "[deleted]")
        c_date = c.get("created_at", "")

        if c_score > best_answer["score"]:
            best_answer = {
                "id": c["id"],
                "author": c_author,
                "score": c_score,
                "body": c_body,
            }

        # Role assignment: OP replies are 'user', community members are 'assistant'
        role = "user" if c_author != "[deleted]" and c_author == post_author else "assistant"

        # Avoid consecutive duplicate roles if desired, or append
        if messages and messages[-1]["role"] == role:
            # Append as an additional paragraph or separate turn
            messages[-1]["content"] += f"\n\n{c_body}"
        else:
            messages.append({"role": role, "content": c_body})

        transcript_lines.append(
            f"**[{c_author}]** *(Score: {c_score}, {c_date})*:\n{c_body}\n"
        )

    if best_answer["score"] == -999999:
        best_answer = {"id": "", "author": "", "score": 0, "body": ""}

    transcript_text = "\n".join(transcript_lines)
    return messages, transcript_text, best_answer


def run_clean_threads(
    input_parquet: Path,
    output_parquet: Path,
    report_path: Path,
    min_date: str = "2023-01-01",
    min_body_length: int = 15,
) -> Dict[str, Any]:
    """Execute cleaning and thread reconstruction pipeline."""
    logger.info(f"Loading normalized data from: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(input_parquet)
    total_input = len(df)
    logger.info(f"Loaded {total_input:,} records.")

    # 1. Date filter (focus on modern RAG era)
    date_filtered_df = df.filter(pl.col("created_at") >= min_date)
    filtered_pre_date = total_input - len(date_filtered_df)
    logger.info(f"Filtered {filtered_pre_date:,} pre-{min_date} records. Remaining: {len(date_filtered_df):,}")

    # Convert to Python dicts for thread reconstruction
    records = date_filtered_df.to_dicts()

    posts_by_id: Dict[str, Dict[str, Any]] = {}
    comments_by_post: Dict[str, List[Dict[str, Any]]] = {}

    filtered_bots = 0
    filtered_negative_score = 0
    filtered_invalid_body = 0

    for r in records:
        author = r.get("author") or ""
        score = int(r.get("score") or 0)
        body = r.get("body") or ""
        title = r.get("title") or ""
        parent_id = r.get("parent_id") or ""
        post_id = r.get("post_id") or r.get("id") or ""

        # Check bot
        if is_bot_author(author):
            filtered_bots += 1
            continue

        # Check negative score
        if score < 0:
            filtered_negative_score += 1
            continue

        is_post = (parent_id == "")

        if is_post:
            # For post, title must be valid, or body must be valid
            has_valid_title = len(title.strip()) >= 5
            has_valid_body = is_valid_body(body, min_length=min_body_length)
            if not has_valid_title and not has_valid_body:
                filtered_invalid_body += 1
                continue
            posts_by_id[r["id"]] = r
        else:
            # For comment, body must be valid
            if not is_valid_body(body, min_length=min_body_length):
                filtered_invalid_body += 1
                continue
            comments_by_post.setdefault(post_id, []).append(r)

    logger.info(f"Valid posts identified: {len(posts_by_id):,}")
    logger.info(f"Valid comments identified: {sum(len(c) for c in comments_by_post.values()):,}")

    cleaned_threads: List[Dict[str, Any]] = []
    threads_with_comments = 0
    standalone_posts = 0

    for p_id, post in posts_by_id.items():
        comments = comments_by_post.get(p_id, [])
        if comments:
            threads_with_comments += 1
        else:
            standalone_posts += 1

        messages, transcript, best_ans = build_thread_conversation(post, comments)

        full_text = post["title"] + " " + post["body"] + " " + " ".join(c["body"] for c in comments)
        words = len(full_text.split())
        chars = len(full_text)

        cleaned_threads.append({
            "thread_id": p_id,
            "community": post.get("community", "rag"),
            "title": post.get("title", ""),
            "post_author": post.get("author", "[deleted]"),
            "post_created_at": post.get("created_at", ""),
            "post_score": int(post.get("score") or 0),
            "post_body": post.get("body", ""),
            "comment_count": len(comments),
            "best_answer_id": best_ans["id"],
            "best_answer_author": best_ans["author"],
            "best_answer_score": best_ans["score"],
            "best_answer_body": best_ans["body"],
            "conversation_turns": len(messages),
            "conversation_json": json.dumps(messages, ensure_ascii=False),
            "discussion_transcript": transcript,
            "total_word_count": words,
            "total_char_count": chars,
        })

    # Write cleaned Parquet table
    if cleaned_threads:
        batch_dict = {col: [row[col] for row in cleaned_threads] for col in CLEANED_THREAD_SCHEMA.names}
        table = pa.Table.from_pydict(batch_dict, schema=CLEANED_THREAD_SCHEMA)
        pq.write_table(table, str(output_parquet), compression="zstd")
        parquet_size = output_parquet.stat().st_size
    else:
        parquet_size = 0

    avg_comments = (
        sum(t["comment_count"] for t in cleaned_threads) / len(cleaned_threads)
        if cleaned_threads else 0.0
    )
    avg_turns = (
        sum(t["conversation_turns"] for t in cleaned_threads) / len(cleaned_threads)
        if cleaned_threads else 0.0
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_input_records": total_input,
        "filtered_pre_min_date": filtered_pre_date,
        "filtered_bots": filtered_bots,
        "filtered_negative_score": filtered_negative_score,
        "filtered_invalid_body": filtered_invalid_body,
        "total_cleaned_threads": len(cleaned_threads),
        "threads_with_comments": threads_with_comments,
        "standalone_posts": standalone_posts,
        "avg_comments_per_thread": round(avg_comments, 2),
        "avg_turns_per_conversation": round(avg_turns, 2),
        "output_parquet": str(output_parquet),
        "parquet_size_bytes": parquet_size,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Cleaned threads written to: {output_parquet}")
    logger.info(f"Cleaning report saved to: {report_path}")
    logger.info(
        f"Summary: Total Threads={len(cleaned_threads):,}, With Comments={threads_with_comments:,}, "
        f"Standalone={standalone_posts:,}, Avg Comments/Thread={avg_comments:.2f}, Avg Turns={avg_turns:.2f}"
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Clean and stitch normalized records into multi-turn threads.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/interim/ingested.parquet"),
        help="Path to normalized input Parquet file",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/interim/cleaned_threads.parquet"),
        help="Path to write output cleaned threads Parquet",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/data_cleaning_report.json"),
        help="Path to write cleaning report JSON",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default="2023-01-01",
        help="Earliest record timestamp to include",
    )
    parser.add_argument(
        "--min-body-length",
        type=int,
        default=15,
        help="Minimum character length for text bodies",
    )

    args = parser.parse_args()
    run_clean_threads(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        report_path=args.report_path,
        min_date=args.min_date,
        min_body_length=args.min_body_length,
    )


if __name__ == "__main__":
    main()
