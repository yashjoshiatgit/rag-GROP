"""Data Ingestion Pipeline for RAG-RL.

Normalizes raw community forum posts and comments into a uniform schema and stores the
result in Parquet format, accompanied by an ingestion health report.

Schema:
    source: str       (e.g., 'community_forum')
    community: str    (e.g., 'rag')
    id: str           (unique item id)
    parent_id: str    (parent id or empty/None for posts)
    post_id: str      (top-level post id)
    author: str       (author username or '[deleted]')
    created_at: str   (ISO 8601 UTC timestamp)
    title: str        (post title or empty for comments)
    body: str         (selftext or comment text)
    score: int        (upvotes minus downvotes)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.ingest")

TARGET_PYARROW_SCHEMA = pa.schema([
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


def _timestamp_to_iso(ts: Any) -> Optional[str]:
    """Convert Unix timestamp (seconds) or string to ISO 8601 string."""
    if ts is None:
        return None
    try:
        val = float(ts)
        # Reasonable bounds (year 2000 to 2100)
        if val < 946684800 or val > 4102444800:
            return None
        return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def normalize_post(raw: Dict[str, Any], default_source: str = "community_forum") -> Optional[Dict[str, Any]]:
    """Normalize a raw forum post dict into standard schema."""
    if "data" in raw and isinstance(raw["data"], dict):
        raw = raw["data"]

    post_id = str(raw.get("id") or "").strip()
    if not post_id:
        return None

    title = str(raw.get("title") or "").strip()
    body = str(raw.get("selftext") or raw.get("body") or "").strip()
    author = str(raw.get("author") or "[deleted]").strip()
    community = str(raw.get("subreddit") or raw.get("community") or "rag").strip().lower()

    created_utc = raw.get("created_utc") or raw.get("created")
    created_at = _timestamp_to_iso(created_utc)

    try:
        score = int(raw.get("score") or 0)
    except (ValueError, TypeError):
        score = 0

    return {
        "source": default_source,
        "community": community,
        "id": post_id,
        "parent_id": "",
        "post_id": post_id,
        "author": author,
        "created_at": created_at or "",
        "title": title,
        "body": body,
        "score": score,
    }


def normalize_comment(raw: Dict[str, Any], default_source: str = "community_forum") -> Optional[Dict[str, Any]]:
    """Normalize a raw forum comment dict into standard schema."""
    if "data" in raw and isinstance(raw["data"], dict):
        raw = raw["data"]

    comment_id = str(raw.get("id") or "").strip()
    if not comment_id:
        return None

    # Handle post_id resolution from link_id (e.g., 't3_14nzqq' -> '14nzqq')
    link_id = str(raw.get("link_id") or "").strip()
    parent_id = str(raw.get("parent_id") or "").strip()

    if link_id.startswith("t3_"):
        post_id = link_id[3:]
    elif parent_id.startswith("t3_"):
        post_id = parent_id[3:]
    else:
        post_id = link_id or parent_id

    body = str(raw.get("body") or "").strip()
    author = str(raw.get("author") or "[deleted]").strip()
    community = str(raw.get("subreddit") or raw.get("community") or "rag").strip().lower()

    created_utc = raw.get("created_utc") or raw.get("created")
    created_at = _timestamp_to_iso(created_utc)

    try:
        score = int(raw.get("score") or 0)
    except (ValueError, TypeError):
        score = 0

    return {
        "source": default_source,
        "community": community,
        "id": comment_id,
        "parent_id": parent_id,
        "post_id": post_id,
        "author": author,
        "created_at": created_at or "",
        "title": "",
        "body": body,
        "score": score,
    }


def stream_jsonl(
    file_path: Path, is_comment: bool, source_name: str = "community_forum"
) -> Generator[Tuple[Optional[Dict[str, Any]], bool], None, None]:
    """Stream and yield (normalized_dict, is_malformed) tuples from a JSONL file."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                if is_comment:
                    normalized = normalize_comment(record, default_source=source_name)
                else:
                    normalized = normalize_post(record, default_source=source_name)

                if normalized is None:
                    yield (None, True)
                else:
                    yield (normalized, False)
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield (None, True)


def run_ingestion(
    posts_path: Path,
    comments_path: Path,
    output_parquet: Path,
    report_path: Path,
    batch_size: int = 50000,
    source_name: str = "community_forum",
) -> Dict[str, Any]:
    """Execute the full ingestion pipeline and output Parquet and report."""
    logger.info("Starting ingestion pipeline...")
    logger.info(f"Posts source: {posts_path}")
    logger.info(f"Comments source: {comments_path}")
    logger.info(f"Output Parquet: {output_parquet}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    total_records = 0
    post_count = 0
    comment_count = 0
    deleted_count = 0
    empty_count = 0
    malformed_count = 0
    duplicate_count = 0
    min_date: Optional[str] = None
    max_date: Optional[str] = None

    current_batch: list[Dict[str, Any]] = []

    # Open Parquet writer
    writer: Optional[pq.ParquetWriter] = None

    def flush_batch():
        nonlocal writer, current_batch
        if not current_batch:
            return
        batch_dict = {col: [row[col] for row in current_batch] for col in TARGET_PYARROW_SCHEMA.names}
        table = pa.Table.from_pydict(batch_dict, schema=TARGET_PYARROW_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(str(output_parquet), TARGET_PYARROW_SCHEMA, compression="zstd")
        writer.write_table(table)
        current_batch = []

    # 1. Ingest Posts
    logger.info("Ingesting posts...")
    for norm, is_malformed in stream_jsonl(posts_path, is_comment=False, source_name=source_name):
        if is_malformed:
            malformed_count += 1
            continue

        item_id = norm["id"]
        if item_id in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(item_id)

        total_records += 1
        post_count += 1

        is_deleted = norm["author"] == "[deleted]" or norm["body"] in ("[deleted]", "[removed]")
        if is_deleted:
            deleted_count += 1

        if not norm["title"] and not norm["body"]:
            empty_count += 1

        dt = norm["created_at"]
        if dt:
            if min_date is None or dt < min_date:
                min_date = dt
            if max_date is None or dt > max_date:
                max_date = dt

        current_batch.append(norm)
        if len(current_batch) >= batch_size:
            flush_batch()

    logger.info(f"Posts ingested: {post_count}")

    # 2. Ingest Comments
    logger.info("Ingesting comments...")
    for norm, is_malformed in stream_jsonl(comments_path, is_comment=True, source_name=source_name):
        if is_malformed:
            malformed_count += 1
            continue

        item_id = norm["id"]
        if item_id in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(item_id)

        total_records += 1
        comment_count += 1

        is_deleted = norm["author"] == "[deleted]" or norm["body"] in ("[deleted]", "[removed]")
        if is_deleted:
            deleted_count += 1

        if not norm["body"]:
            empty_count += 1

        dt = norm["created_at"]
        if dt:
            if min_date is None or dt < min_date:
                min_date = dt
            if max_date is None or dt > max_date:
                max_date = dt

        current_batch.append(norm)
        if len(current_batch) >= batch_size:
            flush_batch()

    # Flush remaining records
    flush_batch()
    if writer:
        writer.close()

    logger.info(f"Comments ingested: {comment_count}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": total_records,
        "posts": post_count,
        "comments": comment_count,
        "deleted": deleted_count,
        "empty": empty_count,
        "malformed": malformed_count,
        "duplicate_ids": duplicate_count,
        "date_range": {
            "min": min_date,
            "max": max_date,
        },
        "output_parquet": str(output_parquet),
        "parquet_size_bytes": output_parquet.stat().st_size if output_parquet.exists() else 0,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Ingestion report saved to {report_path}")
    logger.info(f"Report Summary: Total={total_records:,} (Posts={post_count:,}, Comments={comment_count:,})")
    logger.info(f"Deleted={deleted_count:,}, Empty={empty_count:,}, Malformed={malformed_count:,}, Duplicates={duplicate_count:,}")
    logger.info(f"Date Range: {min_date} to {max_date}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Ingest community forum raw data into standardized Parquet.")
    parser.add_argument(
        "--posts-path",
        type=Path,
        default=Path("data/raw/posts.jsonl"),
        help="Path to raw posts JSONL file",
    )
    parser.add_argument(
        "--comments-path",
        type=Path,
        default=Path("data/raw/comments.jsonl"),
        help="Path to raw comments JSONL file",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/interim/ingested.parquet"),
        help="Path to write output Parquet file",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/data_ingestion_report.json"),
        help="Path to write ingestion report JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for writing to Parquet",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="community_forum",
        help="Source tag identifier for records",
    )

    args = parser.parse_args()

    # Fallback to export/ folder if main files not at root of data/raw/
    if not args.posts_path.exists() and Path("data/raw/export/posts.jsonl").exists():
        args.posts_path = Path("data/raw/export/posts.jsonl")
    if not args.comments_path.exists() and Path("data/raw/export/comments.jsonl").exists():
        args.comments_path = Path("data/raw/export/comments.jsonl")

    run_ingestion(
        posts_path=args.posts_path,
        comments_path=args.comments_path,
        output_parquet=args.output_parquet,
        report_path=args.report_path,
        batch_size=args.batch_size,
        source_name=args.source_name,
    )


if __name__ == "__main__":
    main()
