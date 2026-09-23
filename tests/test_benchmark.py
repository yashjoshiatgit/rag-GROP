"""Unit tests for RAG evaluation benchmark suite."""

import json
from pathlib import Path

from scripts.build_benchmark import BENCHMARK_SUITE, build_benchmark


def test_benchmark_suite_integrity():
    assert len(BENCHMARK_SUITE) >= 10, "Benchmark suite should contain at least 10 canonical questions"

    required_keys = {
        "id",
        "category",
        "difficulty",
        "test_type",
        "question",
        "expected_concepts",
        "reference_reasoning",
        "scoring_rubric",
    }
    required_rubric_keys = {"correctness", "grounding", "tradeoffs"}

    expected_categories = {
        "retrieval",
        "chunking_and_indexing",
        "generation_and_grounding",
        "evaluation_and_benchmarks",
        "agentic_and_graph_rag",
        "tooling_and_frameworks",
    }

    seen_ids = set()
    found_categories = set()

    for item in BENCHMARK_SUITE:
        # Check keys
        assert required_keys.issubset(item.keys()), f"Missing keys in {item.get('id')}"
        assert required_rubric_keys.issubset(item["scoring_rubric"].keys())

        # Uniqueness of ID
        assert item["id"] not in seen_ids, f"Duplicate ID: {item['id']}"
        seen_ids.add(item["id"])

        # Content non-empty
        assert len(item["question"].strip()) > 20
        assert len(item["expected_concepts"]) >= 3
        assert len(item["reference_reasoning"].strip()) > 50

        found_categories.add(item["category"])

    # Verify all 6 taxonomy categories are covered
    assert found_categories == expected_categories


def test_build_benchmark_file_output(tmp_path: Path):
    output_jsonl = tmp_path / "rag_eval.jsonl"
    metadata_json = tmp_path / "rag_eval_metadata.json"

    meta = build_benchmark(output_jsonl=output_jsonl, output_metadata=metadata_json)
    assert meta["total_questions"] == len(BENCHMARK_SUITE)
    assert metadata_json.exists()

    # Read jsonl
    lines = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == len(BENCHMARK_SUITE)
    assert lines[0]["id"] == BENCHMARK_SUITE[0]["id"]
