"""Automated Candidate Baseline Evaluation Harness for RAG-RL.

Evaluates candidate language models against the held-out RAG evaluation benchmark
(`data/evaluation/rag_eval.jsonl`). Computes:
- Concept Recall & Coverage (% of essential RAG mechanisms identified)
- Reasoning Structure & Trace Analysis (e.g., <think> tags, step-by-step logic)
- Inference Throughput & Latency (tokens/sec, time-to-completion)
- Taxonomy Category Breakdown across all 6 core RAG areas

Supports:
- Qwen/Qwen2.5-1.5B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- meta-llama/Llama-3.2-1B-Instruct
- Local CPU / Apple MPS / CUDA GPU (Tesla T4, RTX 3050, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.eval_baseline")

SYSTEM_PROMPT = (
    "You are an expert RAG systems architect and researcher. "
    "Analyze the technical question carefully and provide a rigorous, grounded, and comprehensive explanation. "
    "Detail the underlying mechanisms, failure modes, trade-offs, and standard architectural solutions."
)


def normalize_term(term: str) -> str:
    """Normalize string for robust token/concept matching."""
    term = term.lower()
    # Remove punctuation except hyphens
    term = re.sub(r"[^\w\s-]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    return term


def extract_concept_variants(concept: str) -> List[str]:
    """Extract matchable variants from a concept definition.

    E.g.:
    - 'BM25 / sparse retrieval' -> ['bm25', 'sparse retrieval']
    - 'Reciprocal Rank Fusion (RRF)' -> ['reciprocal rank fusion', 'rrf']
    - 'bi-encoder' -> ['bi-encoder', 'bi encoder', 'biencoder']
    """
    variants: Set[str] = set()

    # Split on slash or ' or '
    parts = re.split(r"\s*/\s*|\s+or\s+", concept)
    for part in parts:
        cleaned = normalize_term(part)
        if cleaned:
            variants.add(cleaned)
        # Extract acronyms inside parentheses e.g. (RRF)
        parens = re.findall(r"\(([^)]+)\)", part)
        for p in parens:
            p_clean = normalize_term(p)
            if p_clean:
                variants.add(p_clean)
        # Also clean without parens
        without_parens = normalize_term(re.sub(r"\([^)]*\)", "", part))
        if without_parens:
            variants.add(without_parens)

    # For hyphenated terms, also add spaced version
    hyphen_variants = set()
    for v in variants:
        if "-" in v:
            hyphen_variants.add(v.replace("-", " "))
            hyphen_variants.add(v.replace("-", ""))
    variants.update(hyphen_variants)

    return [v for v in variants if len(v) >= 2]


def score_concept_coverage(
    generated_text: str,
    expected_concepts: List[str],
) -> Tuple[float, List[str], List[str]]:
    """Measure recall of expected technical concepts in generated text."""
    normalized_text = normalize_term(generated_text)
    # Also keep text with hyphens replaced by space for matching
    spaced_text = normalized_text.replace("-", " ")

    matched: List[str] = []
    missing: List[str] = []

    for concept in expected_concepts:
        variants = extract_concept_variants(concept)
        is_matched = False
        for v in variants:
            # Word boundary matching where possible, or substring if multi-word
            if " " in v or len(v) >= 4:
                if v in normalized_text or v in spaced_text:
                    is_matched = True
                    break
            else:
                pattern = rf"\b{re.escape(v)}\b"
                if re.search(pattern, normalized_text) or re.search(pattern, spaced_text):
                    is_matched = True
                    break

        if is_matched:
            matched.append(concept)
        else:
            missing.append(concept)

    recall = len(matched) / len(expected_concepts) if expected_concepts else 0.0
    return round(recall, 4), matched, missing


def parse_reasoning_trace(text: str) -> Tuple[Optional[str], str]:
    """Separate reasoning/thinking trace from final answer (for reasoning models)."""
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_match:
        thinking_trace = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thinking_trace, answer
    return None, text.strip()


def run_evaluation(
    model_id: str,
    benchmark_path: Path,
    output_path: Path,
    device: str = "auto",
    torch_dtype: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.95,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full baseline evaluation on benchmark suite."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading benchmark suite: {benchmark_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    questions: List[Dict[str, Any]] = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if sample_limit and sample_limit > 0:
        questions = questions[:sample_limit]
        logger.info(f"Limiting evaluation to first {sample_limit} questions.")

    logger.info(f"Loading model & tokenizer: {model_id}")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(torch_dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        resolved_device_map = "auto"
    else:
        resolved_device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        device_map=resolved_device_map,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)

    model.eval()
    logger.info(f"Model successfully loaded on device: {model.device}")

    eval_results: List[Dict[str, Any]] = []
    category_recalls: Dict[str, List[float]] = {}
    difficulty_recalls: Dict[str, List[float]] = {}
    total_tokens_generated = 0
    total_generation_seconds = 0.0

    start_eval_time = time.perf_counter()

    for idx, item in enumerate(questions, start=1):
        q_id = item["id"]
        category = item["category"]
        difficulty = item["difficulty"]
        question = item["question"]
        expected_concepts = item.get("expected_concepts", [])

        logger.info(f"[{idx}/{len(questions)}] Evaluating {q_id} ({category} | {difficulty})...")

        # Format conversation prompt using chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"System: {SYSTEM_PROMPT}\nUser: {question}\nAssistant:"

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_token_count = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            output_ids = model.generate(**inputs, **gen_kwargs)
        t1 = time.perf_counter()

        generation_time = t1 - t0
        generated_tokens = output_ids[0][prompt_token_count:]
        num_generated_tokens = len(generated_tokens)
        tokens_per_sec = num_generated_tokens / generation_time if generation_time > 0 else 0.0

        full_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        clean_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        thinking_trace, final_answer = parse_reasoning_trace(clean_output)
        recall, matched, missing = score_concept_coverage(clean_output, expected_concepts)

        total_tokens_generated += num_generated_tokens
        total_generation_seconds += generation_time

        category_recalls.setdefault(category, []).append(recall)
        difficulty_recalls.setdefault(difficulty, []).append(recall)

        eval_results.append({
            "id": q_id,
            "category": category,
            "difficulty": difficulty,
            "test_type": item.get("test_type", ""),
            "question": question,
            "expected_concepts": expected_concepts,
            "matched_concepts": matched,
            "missing_concepts": missing,
            "concept_recall": recall,
            "has_thinking_trace": thinking_trace is not None,
            "thinking_length_chars": len(thinking_trace) if thinking_trace else 0,
            "answer_length_chars": len(final_answer),
            "generated_tokens": num_generated_tokens,
            "latency_seconds": round(generation_time, 3),
            "tokens_per_second": round(tokens_per_sec, 2),
            "generated_answer": final_answer,
            "thinking_trace": thinking_trace,
        })

    total_eval_duration = time.perf_counter() - start_eval_time
    avg_recall = (
        sum(r["concept_recall"] for r in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )
    overall_tok_per_sec = (
        total_tokens_generated / total_generation_seconds
        if total_generation_seconds > 0
        else 0.0
    )

    category_summary = {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in category_recalls.items()
    }
    difficulty_summary = {
        diff: round(sum(scores) / len(scores), 4)
        for diff, scores in difficulty_recalls.items()
    }

    report = {
        "model_id": model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_questions_evaluated": len(eval_results),
        "overall_concept_recall": round(avg_recall, 4),
        "total_tokens_generated": total_tokens_generated,
        "total_generation_seconds": round(total_generation_seconds, 2),
        "overall_tokens_per_second": round(overall_tok_per_sec, 2),
        "total_eval_duration_seconds": round(total_eval_duration, 2),
        "category_summary": category_summary,
        "difficulty_summary": difficulty_summary,
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "torch_dtype": torch_dtype,
            "device": str(model.device),
        },
        "results": eval_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Evaluation Complete for: {model_id}")
    logger.info(f"Overall Concept Recall: {avg_recall * 100:.2f}%")
    logger.info(f"Average Throughput:    {overall_tok_per_sec:.2f} tok/s")
    logger.info(f"Category Breakdown:    {category_summary}")
    logger.info(f"Report written to:     {output_path}")
    logger.info("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on RAG benchmark.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID or local directory path",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("data/evaluation/rag_eval.jsonl"),
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to output evaluation report JSON (defaults to outputs/baseline_eval_{model}.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: 'auto', 'cuda', 'cpu', or 'mps'",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Precision: 'bfloat16', 'float16', or 'auto'",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum generation length per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Generation temperature",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N questions",
    )

    args = parser.parse_args()

    if args.output_path is None:
        clean_model_name = re.sub(r"[^\w-]", "_", args.model_id)
        args.output_path = Path(f"outputs/baseline_eval_{clean_model_name}.json")

    run_evaluation(
        model_id=args.model_id,
        benchmark_path=args.benchmark_path,
        output_path=args.output_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
