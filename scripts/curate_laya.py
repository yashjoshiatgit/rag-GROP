"""Semantic Curation and Taxonomy Classification Pipeline using Laya.

Evaluates cleaned conversation threads using Laya's non-autoregressive decision engine:
1. Pre-filtering: Fast deterministic rejection of community meta, promo, and bare URLs.
2. is_rag_related (noul: calibrated probability P(true))
3. technical_quality (score: 0=low/superficial, 1=medium/practical, 2=deep/architectural)
4. taxonomy_topic (choice: retrieval, chunking_and_indexing, generation_and_grounding,
                   evaluation_and_benchmarks, agentic_and_graph_rag, tooling_and_frameworks)

Outputs curated gold RAG dataset to data/processed/rag_clean.parquet.
"""

from __future__ import annotations

import os
# Windows workaround: disable symlinks in Hugging Face Hub cache to prevent WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import huggingface_hub.constants as hf_constants
    hf_constants.HF_HUB_DISABLE_SYMLINKS = True
    hf_constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
except ImportError:
    pass

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.curate_laya")

# Deterministic patterns for community announcements, server invites, and meta posts
META_PROMO_PATTERNS = [
    r"\bdiscord(?:\.gg|\.com|\s+server)?\b",
    r"\btelegram\b",
    r"\bwhatsapp\b",
    r"\bjoin\s+(?:our|the)\s+(?:discord|server|community|group)\b",
    r"\bsubreddit\s+(?:rules|update|moderator|feedback)\b",
    r"\bwelcome\s+to\s+(?:the\s+)?(?:sub|community|rag)\b",
    r"\bweekly\s+discussion\s+thread\b",
    r"\bmonthly\s+recap\b",
]

RAG_CURATION_QUESTIONS = {
    "is_rag_related": {
        "type": "noul",
        "instructions": (
            "Is this question or discussion specifically about Retrieval-Augmented Generation (RAG), "
            "vector databases, embeddings, chunking, semantic or hybrid search, context retrieval, "
            "reranking, or LLM grounding?"
        ),
    },
    "technical_quality": {
        "type": "score",
        "instructions": "Rate the technical depth, quality, and usefulness of this discussion.",
        "criteria": [
            "low: vague, beginner rant, promotional, superficial, or low technical detail",
            "medium: standard practical question, bug troubleshooting, or working setup advice",
            "high: deep technical architecture, benchmark, detailed diagnosis, or working production solution",
        ],
    },
    "taxonomy_topic": {
        "type": "choice",
        "instructions": "What is the primary technical domain of this discussion?",
        "criteria": {
            "retrieval": "vector search, hybrid search, BM25, reranking, cross-encoders, query expansion",
            "chunking_and_indexing": "chunking strategies, embeddings, metadata filtering, vector databases like Qdrant/Chroma/Milvus",
            "generation_and_grounding": "prompt engineering, context grounding, hallucination mitigation, citations",
            "evaluation_and_benchmarks": "faithfulness, answer relevance, recall, precision, Ragas, TruLens, benchmarks",
            "agentic_and_graph_rag": "agentic workflows, knowledge graphs, corrective RAG (CRAG), multi-hop reasoning",
            "tooling_and_frameworks": "LangChain, LlamaIndex, Ollama, frameworks, configuration, local deployment",
        },
    },
}

CURATED_SCHEMA = pa.schema([
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
    ("conversation_json", pa.string()),
    ("discussion_transcript", pa.string()),
    ("total_word_count", pa.int64()),
    ("total_char_count", pa.int64()),
    ("is_rag_related", pa.float64()),
    ("technical_quality_score", pa.float64()),
    ("taxonomy_topic", pa.string()),
    ("topic_confidence", pa.float64()),
    ("curation_passed", pa.bool_()),
])


def is_bare_url_or_promo(title: str, body: str) -> bool:
    """Pre-filter out Discord/community promo and bare URL link drops."""
    combined = f"{title} {body}".lower().strip()

    for pat in META_PROMO_PATTERNS:
        if re.search(pat, combined):
            return True

    # Bare link posts with minimal words
    stripped_body = body.strip()
    if stripped_body.startswith("http://") or stripped_body.startswith("https://"):
        words = stripped_body.split()
        if len(words) <= 10:
            return True

    if len(stripped_body) < 100 and ("http://" in stripped_body or "https://" in stripped_body):
        words = [w for w in stripped_body.split() if not w.startswith("http")]
        if len(words) < 6:
            return True

    return False


def format_state(row: Dict[str, Any], max_text_chars: int = 600) -> Dict[str, str]:
    """Format row into a compact state dictionary within token budget."""
    title = str(row.get("title") or "").strip()
    post_body = str(row.get("post_body") or "").strip()[:max_text_chars]
    best_answer = str(row.get("best_answer_body") or "").strip()[:max_text_chars]

    state = {"title": title}
    if post_body:
        state["post_body"] = post_body
    if best_answer:
        state["best_answer"] = best_answer
    return state


def run_curate_laya(
    input_parquet: Path,
    output_parquet: Path,
    report_path: Path,
    rejected_parquet: Optional[Path] = None,
    model_id: str = "convaiinnovations/laya",
    subfolder: Optional[str] = "typed-decisions",
    device: str = "cpu",
    limit: Optional[int] = None,
    rag_threshold: float = 0.60,
    min_quality_score: float = 0.60,
    num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Laya curation pipeline over cleaned threads."""
    import laya

    if num_threads:
        torch.set_num_threads(num_threads)
        logger.info(f"Set PyTorch CPU threads: {num_threads}")

    logger.info(f"Loading input threads from: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if rejected_parquet:
        rejected_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(input_parquet)
    if limit and limit > 0:
        logger.info(f"Applying limit: {limit} threads (out of {len(df):,})")
        df = df.head(limit)
    rows = df.to_dicts()
    total_threads = len(rows)
    logger.info(f"Threads to evaluate: {total_threads:,}")

    logger.info(f"Loading Laya agent (model={model_id}, subfolder={subfolder}, device={device})...")
    load_start = time.perf_counter()
    try:
        if subfolder:
            agent = laya.load(model_id, device=device, subfolder=subfolder)
        else:
            agent = laya.load(model_id, device=device)
    except Exception as e:
        logger.warning(f"Could not load subfolder {subfolder!r}: {e}. Falling back to root checkpoint.")
        agent = laya.load(model_id, device=device)

    logger.info(f"Laya agent loaded in {time.perf_counter() - load_start:.2f}s")

    curated_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    topic_counts: Dict[str, int] = {}
    latencies: List[float] = []
    pre_filtered_promo = 0

    logger.info("Starting inference loop...")
    progress_bar = tqdm(rows, desc="Curating with Laya", unit="thread")

    for row in progress_bar:
        title = row.get("title", "")
        post_body = row.get("post_body", "")

        # 1. Deterministic Pre-filter for promos/bare URLs
        if is_bare_url_or_promo(title, post_body):
            pre_filtered_promo += 1
            curated_entry = dict(row)
            curated_entry.update({
                "is_rag_related": 0.05,
                "technical_quality_score": 0.1,
                "taxonomy_topic": "meta_or_promo",
                "topic_confidence": 0.99,
                "curation_passed": False,
            })
            rejected_records.append(curated_entry)
            progress_bar.set_postfix({"pass": len(curated_records), "topic": "promo_skip", "ms": "0.0"})
            continue

        # 2. Format State and Run Laya
        state = format_state(row)
        t0 = time.perf_counter()
        res = agent.predict(state, RAG_CURATION_QUESTIONS)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

        answers = res.get("answers", {})

        # Decision Primitives
        rag_prob = float(answers.get("is_rag_related", {}).get("noul", 0.0))
        quality_score = float(answers.get("technical_quality", {}).get("score", 0.0))
        topic_ans = answers.get("taxonomy_topic", {})
        topic_choice = str(topic_ans.get("choice", "other"))
        topic_conf = float(topic_ans.get("confidence", 0.0))

        topic_counts[topic_choice] = topic_counts.get(topic_choice, 0) + 1

        is_rag = rag_prob >= rag_threshold
        is_quality = quality_score >= min_quality_score
        passed = bool(is_rag and is_quality)

        curated_entry = dict(row)
        curated_entry.update({
            "is_rag_related": round(rag_prob, 4),
            "technical_quality_score": round(quality_score, 4),
            "taxonomy_topic": topic_choice,
            "topic_confidence": round(topic_conf, 4),
            "curation_passed": passed,
        })

        if passed:
            curated_records.append(curated_entry)
        else:
            rejected_records.append(curated_entry)

        progress_bar.set_postfix({
            "pass": len(curated_records),
            "topic": topic_choice,
            "ms": f"{dt:.1f}",
        })

    # Save curated Parquet
    if curated_records:
        batch_dict = {col: [row[col] for row in curated_records] for col in CURATED_SCHEMA.names}
        table = pa.Table.from_pydict(batch_dict, schema=CURATED_SCHEMA)
        pq.write_table(table, str(output_parquet), compression="zstd")
        parquet_size = output_parquet.stat().st_size
    else:
        parquet_size = 0

    # Save rejected Parquet if requested
    if rejected_parquet and rejected_records:
        rej_dict = {col: [row[col] for row in rejected_records] for col in CURATED_SCHEMA.names}
        rej_table = pa.Table.from_pydict(rej_dict, schema=CURATED_SCHEMA)
        pq.write_table(rej_table, str(rejected_parquet), compression="zstd")

    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    pass_rate = (len(curated_records) / total_threads * 100) if total_threads > 0 else 0.0

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_evaluated": total_threads,
        "pre_filtered_promos": pre_filtered_promo,
        "curated_passed": len(curated_records),
        "curated_rejected": len(rejected_records),
        "pass_rate_pct": round(pass_rate, 2),
        "thresholds": {
            "rag_relevance_min_prob": rag_threshold,
            "technical_quality_min_score": min_quality_score,
        },
        "taxonomy_distribution": topic_counts,
        "inference_stats": {
            "device": device,
            "subfolder": subfolder,
            "avg_latency_ms": round(avg_ms, 2),
            "total_inference_sec": round(sum(latencies) / 1000.0, 2),
        },
        "output_parquet": str(output_parquet),
        "parquet_size_bytes": parquet_size,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Curated dataset written to: {output_parquet}")
    logger.info(f"Curation report saved to: {report_path}")
    logger.info(
        f"Summary: Evaluated={total_threads:,}, Pre-filtered Promo/Links={pre_filtered_promo:,}, "
        f"Passed={len(curated_records):,} ({pass_rate:.1f}%), Avg Latency={avg_ms:.1f} ms/thread"
    )
    logger.info(f"Taxonomy breakdown: {json.dumps(topic_counts)}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Curate conversation threads with Laya decision engine.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/interim/cleaned_threads.parquet"),
        help="Path to cleaned input threads Parquet",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/processed/rag_clean.parquet"),
        help="Path to write curated output Parquet",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/data_curation_report.json"),
        help="Path to write curation report JSON",
    )
    parser.add_argument(
        "--rejected-parquet",
        type=Path,
        default=Path("data/interim/laya_rejected.parquet"),
        help="Path to write rejected threads Parquet",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="convaiinnovations/laya",
        help="Hugging Face repo ID for Laya",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="typed-decisions",
        help="Subfolder for checkpoint (e.g., 'typed-decisions')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of threads to process (useful for pilot testing)",
    )
    parser.add_argument(
        "--rag-threshold",
        type=float,
        default=0.60,
        help="Minimum calibrated probability P(true) for is_rag_related",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.60,
        help="Minimum expected quality score (0.0 to 2.0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of PyTorch CPU threads",
    )

    args = parser.parse_args()

    run_curate_laya(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        report_path=args.report_path,
        rejected_parquet=args.rejected_parquet,
        model_id=args.model_id,
        subfolder=args.subfolder,
        device=args.device,
        limit=args.limit,
        rag_threshold=args.rag_threshold,
        min_quality_score=args.min_quality_score,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()
