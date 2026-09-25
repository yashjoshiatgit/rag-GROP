"""Unit tests for GRPO multi-objective reward functions."""

import pytest
from src.rag_rl.rewards import (
    reward_reasoning_format,
    reward_rag_concept_density,
    reward_grounding_alignment,
    reward_anti_repetition,
)


def test_reward_reasoning_format():
    valid = (
        "<think>\n"
        "The user is experiencing low recall on dense vector retrieval because of exact SKU codes. "
        "Dense vectors map words into continuous space, failing on unique alphanumeric strings. "
        "We need hybrid search.\n"
        "</think>\n"
        "<answer>\n"
        "Use hybrid search combining BM25 lexical retrieval with dense vectors, fused via RRF.\n"
        "</answer>"
    )
    unclosed = "<think>We should use BM25 hybrid search because dense fails"
    missing = "Just use hybrid search with BM25."
    empty = ""

    scores = reward_reasoning_format([valid, unclosed, missing, empty])
    assert scores[0] == 1.0, "Valid thinking + answer should receive full +1.0"
    assert scores[1] == -0.5, "Unclosed tag should receive penalty"
    assert scores[2] == -1.0, "Missing tags should receive -1.0 penalty"
    assert scores[3] == -1.0, "Empty completion should receive -1.0 penalty"


def test_reward_rag_concept_density():
    retrieval_text = (
        "We should implement hybrid search combining BM25 and dense embedding bi-encoder. "
        "Then fuse using reciprocal rank fusion (RRF) and pass to a cross-encoder reranker."
    )
    fluff_text = "I think you should just try harder and check the documentation carefully."

    scores = reward_rag_concept_density(
        completions=[retrieval_text, fluff_text],
        taxonomy_topic=["retrieval", "retrieval"],
        max_reward=2.0,
    )

    assert scores[0] >= 1.2, f"Expected high concept score for rich RAG text, got {scores[0]}"
    assert scores[1] == 0.0, f"Expected 0.0 for text with zero RAG concepts, got {scores[1]}"
    assert scores[0] <= 2.0, "Should cap at max_reward"


def test_reward_grounding_alignment():
    reference = "Use BM25 lexical search with dense embeddings fused via RRF or cross-encoder."
    aligned = "I recommend using BM25 lexical search combined with dense embeddings via RRF."
    irrelevant = "The weather is very sunny today in California."

    scores = reward_grounding_alignment(
        completions=[aligned, irrelevant],
        reference_answer=[reference, reference],
    )

    assert scores[0] > scores[1], "Aligned completion should score higher than irrelevant completion"
    assert scores[0] >= 0.4
    assert scores[1] <= 0.1


def test_reward_anti_repetition():
    normal_text = (
        "To optimize your vector database index, configure HNSW parameters: "
        "M=16, efConstruction=200, and efSearch=64. This balances latency and recall."
    )
    looping_text = "use hybrid search and " * 30
    empty_text = "ok"

    scores = reward_anti_repetition([normal_text, looping_text, empty_text])

    assert scores[0] == 0.0, "Normal text should have zero penalty"
    assert scores[1] < 0.0, "Repetitive looping text should receive negative penalty"
    assert scores[2] == -1.5, "Too short/empty text should receive -1.5 penalty"


def test_reward_functions_with_list_of_dicts():
    # TRL GRPOTrainer can pass completions as list of message dicts:
    # [[{"role": "assistant", "content": "..."}]]
    chat_completion = [
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    "We need to solve dense embedding failure on exact SKU numbers using hybrid BM25 search.\n"
                    "</think>\n"
                    "<answer>\n"
                    "Use BM25 lexical search with Reciprocal Rank Fusion.\n"
                    "</answer>"
                ),
            }
        ]
    ]

    format_scores = reward_reasoning_format(chat_completion)
    assert format_scores[0] == 1.0

    concept_scores = reward_rag_concept_density(
        chat_completion,
        taxonomy_topic=["retrieval"],
    )
    assert concept_scores[0] >= 0.8

    repetition_scores = reward_anti_repetition(chat_completion)
    assert repetition_scores[0] == 0.0
