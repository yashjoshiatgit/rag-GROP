"""Unit tests for baseline evaluation harness."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.eval_baseline import (
    extract_concept_variants,
    normalize_term,
    parse_reasoning_trace,
    run_evaluation,
    score_concept_coverage,
)


def test_normalize_term():
    assert normalize_term("BM25 / Sparse Retrieval") == "bm25 sparse retrieval"
    assert normalize_term("Bi-Encoder") == "bi-encoder"
    assert normalize_term("Reciprocal Rank Fusion (RRF)!") == "reciprocal rank fusion rrf"


def test_extract_concept_variants():
    v1 = extract_concept_variants("BM25 / sparse retrieval")
    assert "bm25" in v1
    assert "sparse retrieval" in v1

    v2 = extract_concept_variants("Reciprocal Rank Fusion (RRF)")
    assert "reciprocal rank fusion" in v2
    assert "rrf" in v2

    v3 = extract_concept_variants("bi-encoder")
    assert "bi-encoder" in v3
    assert "bi encoder" in v3
    assert "biencoder" in v3


def test_score_concept_coverage():
    expected = [
        "dense embeddings",
        "BM25 / sparse retrieval",
        "Reciprocal Rank Fusion (RRF)",
        "cross-encoder",
    ]

    generated_text = (
        "In this scenario, dense embeddings fail to capture exact SKU tokens. "
        "We should employ BM25 lexical search and fuse the candidates using RRF. "
        "Afterwards, a cross encoder reranker can score the top 50 passages."
    )

    recall, matched, missing = score_concept_coverage(generated_text, expected)

    assert recall == 1.0
    assert len(matched) == 4
    assert len(missing) == 0

    # Partial match
    partial_text = "We should use dense embeddings and nothing else."
    recall_partial, matched_partial, missing_partial = score_concept_coverage(
        partial_text, expected
    )
    assert recall_partial == 0.25
    assert matched_partial == ["dense embeddings"]
    assert "cross-encoder" in missing_partial


def test_parse_reasoning_trace():
    # Case with <think> tag
    text_with_think = "<think>The user is asking about dense retrieval failure.</think>Use BM25 hybrid search."
    trace, answer = parse_reasoning_trace(text_with_think)
    assert trace == "The user is asking about dense retrieval failure."
    assert answer == "Use BM25 hybrid search."

    # Case without <think> tag
    normal_text = "Use BM25 hybrid search directly."
    trace_none, answer_same = parse_reasoning_trace(normal_text)
    assert trace_none is None
    assert answer_same == "Use BM25 hybrid search directly."


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_run_evaluation_mocked(mock_tok_cls, mock_model_cls, tmp_path: Path):
    benchmark_file = tmp_path / "test_benchmark.jsonl"
    output_file = tmp_path / "eval_out.json"

    item = {
        "id": "rag_test_001",
        "category": "retrieval",
        "difficulty": "medium",
        "test_type": "diagnostic",
        "question": "How to solve dense vector failure on SKU codes?",
        "expected_concepts": ["dense embeddings", "BM25", "hybrid search"],
    }
    benchmark_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    # Mock tokenizer
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<eos>"
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1
    mock_tok.apply_chat_template.return_value = "Prompt text"

    import torch
    dummy_input_ids = torch.tensor([[1, 2, 3]])
    mock_tok.return_value = {"input_ids": dummy_input_ids}
    mock_tok.decode.return_value = "Dense embeddings struggle with exact codes. Use BM25 and hybrid search."
    mock_tok_cls.return_value = mock_tok

    # Mock model
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.to.return_value = mock_model
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    mock_model_cls.return_value = mock_model

    report = run_evaluation(
        model_id="mock/test-model",
        benchmark_path=benchmark_file,
        output_path=output_file,
        device="cpu",
    )

    assert output_file.exists()
    assert report["total_questions_evaluated"] == 1
    assert report["overall_concept_recall"] == 1.0
    assert report["category_summary"]["retrieval"] == 1.0
    assert len(report["results"]) == 1
    assert report["results"][0]["matched_concepts"] == ["dense embeddings", "BM25", "hybrid search"]
