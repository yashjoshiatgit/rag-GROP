"""Multi-Objective Reward Functions for RAG-RL GRPO Policy Optimization.

Defines mathematically stable, modular reward functions compatible with TRL GRPOTrainer:
1. reward_reasoning_format: Enforces strict <think>...</think> and <answer>...</answer> structure.
2. reward_rag_concept_density: Scores presence and density of valid domain-specific RAG mechanisms.
3. reward_grounding_alignment: Rewards semantic/lexical overlap with verified reference solutions.
4. reward_anti_repetition: Penalizes degenerate n-gram looping, excessive length, or emptiness.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# TAXONOMY CONCEPT VOCABULARIES FOR REWARD SCORING
# =============================================================================

RAG_CONCEPT_TAXONOMY: Dict[str, List[str]] = {
    "retrieval": [
        "dense embedding",
        "bi-encoder",
        "cross-encoder",
        "reranker",
        "bm25",
        "sparse retrieval",
        "splade",
        "hybrid search",
        "reciprocal rank fusion",
        "rrf",
        "hnsw",
        "approximate nearest neighbor",
        "ann",
        "cosine similarity",
        "dot product",
        "colbert",
        "late interaction",
        "hyde",
        "query expansion",
        "query rewrite",
        "two-stage retrieval",
    ],
    "chunking_and_indexing": [
        "chunk size",
        "chunk overlap",
        "sliding window",
        "recursive character splitter",
        "semantic chunking",
        "parent document retriever",
        "sentence window retrieval",
        "hierarchical indexing",
        "metadata filtering",
        "vector database",
        "payload filtering",
        "qdrant",
        "milvus",
        "chroma",
        "pinecone",
        "pgvector",
        "index construction",
    ],
    "generation_and_grounding": [
        "in-context learning",
        "context stuffing",
        "lost in the middle",
        "faithfulness",
        "hallucination mitigation",
        "citation grounding",
        "system prompt",
        "context window",
        "groundedness",
        "temperature",
        "chain-of-thought",
        "cot",
        "attribution",
    ],
    "evaluation_and_benchmarks": [
        "ragas",
        "truelens",
        "deepeval",
        "context precision",
        "context recall",
        "faithfulness",
        "answer relevance",
        "llm-as-a-judge",
        "golden dataset",
        "ground truth",
        "mrr",
        "mean reciprocal rank",
        "ndcg",
        "hit rate",
    ],
    "agentic_and_graph_rag": [
        "agentic rag",
        "graph rag",
        "knowledge graph",
        "neo4j",
        "cypher",
        "entity extraction",
        "triplets",
        "adaptive routing",
        "corrective rag",
        "crag",
        "self-rag",
        "tool calling",
        "multi-hop reasoning",
        "sub-query decomposition",
    ],
    "tooling_and_frameworks": [
        "langchain",
        "llamaindex",
        "haystack",
        "dspy",
        "ollama",
        "vllm",
        "tensorrt-llm",
        "text-generation-inference",
        "fastapi",
        "rag pipeline",
        "docker",
        "production deployment",
    ],
}

# Flat set of all known RAG terms for general scoring
ALL_RAG_CONCEPTS: Set[str] = {
    term
    for sublist in RAG_CONCEPT_TAXONOMY.values()
    for term in sublist
}


def extract_completion_text(completion: Any) -> str:
    """Extract string content from completion whether it is str, dict, or list of message dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def normalize_text(text: Any) -> str:
    """Normalize text for consistent concept matching."""
    text_str = extract_completion_text(text)
    text_str = text_str.lower()
    text_str = re.sub(r"[^\w\s-]", " ", text_str)
    return re.sub(r"\s+", " ", text_str).strip()


# =============================================================================
# REWARD FUNCTION 1: REASONING FORMAT REWARD
# =============================================================================

def reward_reasoning_format(
    completions: List[str],
    min_think_chars: int = 50,
    min_answer_chars: int = 30,
    **kwargs: Any,
) -> List[float]:
    """Reward completion format adherence to <think>...</think> and answer sections.

    Scoring logic:
    +1.0: Contains complete <think>...</think> tags with substantive reasoning AND an answer.
    +0.5: Contains reasoning tags, but answer section is very brief.
    -0.5: Contains partial/unclosed tags (e.g. <think> without </think>).
    -1.0: Completely missing thinking tags.
    """
    rewards: List[float] = []

    for text in completions:
        text = extract_completion_text(text)

        # Check for open and closing think tags
        has_open = "<think>" in text
        has_close = "</think>" in text

        if has_open and has_close:
            match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
            if match:
                think_content = match.group(1).strip()
                answer_content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

                if len(think_content) >= min_think_chars and len(answer_content) >= min_answer_chars:
                    # Explicit answer tags bonus (+1.0), or standard (+0.8)
                    has_ans_tags = "<answer>" in text and "</answer>" in text
                    rewards.append(1.0 if has_ans_tags else 0.8)
                elif len(think_content) >= min_think_chars:
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        elif has_open or has_close:
            # Unclosed tag (cut off or malformed)
            rewards.append(-0.5)
        else:
            # No thinking tags at all
            rewards.append(-1.0)

    return rewards


# =============================================================================
# REWARD FUNCTION 2: RAG CONCEPT DENSITY REWARD
# =============================================================================

def reward_rag_concept_density(
    completions: List[Any],
    taxonomy_topic: Optional[List[str]] = None,
    max_reward: float = 2.0,
    **kwargs: Any,
) -> List[float]:
    """Score density and presence of domain-specific RAG architectural concepts.

    Awards points for:
    - Domain-specific terms from the target taxonomy category (0.4 points per matched concept)
    - General RAG concepts (0.2 points per matched concept)
    - Chain-of-Thought planning bonus for concepts brainstormed in <think> (0.15 points per concept)
    Capped at max_reward (default +2.0).
    """
    rewards: List[float] = []

    for idx, text in enumerate(completions):
        text = extract_completion_text(text)
        norm_text = normalize_text(text)
        spaced_text = norm_text.replace("-", " ")

        score = 0.0
        matched_terms: Set[str] = set()

        # Category-specific concepts get higher reward weight
        target_category = taxonomy_topic[idx] if taxonomy_topic and idx < len(taxonomy_topic) else None
        category_vocab = RAG_CONCEPT_TAXONOMY.get(target_category or "", [])

        # Check for concepts planned inside <think> scratchpad
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if think_match:
            think_norm = normalize_text(think_match.group(1))
            think_spaced = think_norm.replace("-", " ")
            for term in (category_vocab if target_category else ALL_RAG_CONCEPTS):
                term_clean = normalize_text(term)
                if term_clean in think_norm or term_clean in think_spaced:
                    score += 0.15

        for term in category_vocab:
            term_clean = normalize_text(term)
            if term_clean in norm_text or term_clean in spaced_text:
                if term_clean not in matched_terms:
                    matched_terms.add(term_clean)
                    score += 0.4

        # General RAG concepts
        for term in ALL_RAG_CONCEPTS:
            term_clean = normalize_text(term)
            if term_clean in norm_text or term_clean in spaced_text:
                if term_clean not in matched_terms:
                    matched_terms.add(term_clean)
                    score += 0.2

        rewards.append(min(round(score, 3), max_reward))

    return rewards


# =============================================================================
# REWARD FUNCTION 3: SOLUTION GROUNDING & ALIGNMENT
# =============================================================================

def reward_grounding_alignment(
    completions: List[Any],
    reference_answer: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[float]:
    """Reward semantic and lexical alignment with community-verified reference solution.

    Measures token Jaccard similarity and key technical term recall against the gold answer.
    Score ranges from 0.0 to +1.0.
    """
    rewards: List[float] = []

    for idx, text in enumerate(completions):
        text = extract_completion_text(text)
        if not reference_answer or idx >= len(reference_answer) or not reference_answer[idx]:
            rewards.append(0.5)  # Neutral score if reference is absent
            continue

        ref_text = extract_completion_text(reference_answer[idx])
        norm_comp = set(normalize_text(text).split())
        norm_ref = set(normalize_text(ref_text).split())

        # Filter out common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "in", "on", "for", "with",
            "at", "by", "from", "of", "and", "or", "that", "this", "it", "you", "i", "can",
            "should", "would", "be", "do", "have", "as", "if", "so", "what", "how", "why"
        }
        content_comp = {w for w in norm_comp if w not in stop_words and len(w) > 2}
        content_ref = {w for w in norm_ref if w not in stop_words and len(w) > 2}

        if not content_ref:
            rewards.append(0.5)
            continue

        intersection = content_comp.intersection(content_ref)
        # Jaccard overlap on content words
        jaccard = len(intersection) / len(content_comp.union(content_ref)) if content_comp else 0.0

        # Reference keyword recall
        recall = len(intersection) / len(content_ref)

        composite_score = 0.5 * jaccard + 0.5 * min(recall * 1.5, 1.0)
        rewards.append(round(min(composite_score, 1.0), 3))

    return rewards


# =============================================================================
# REWARD FUNCTION 4: ANTI-REPETITION & LENGTH PENALTY
# =============================================================================

def reward_anti_repetition(
    completions: List[Any],
    max_char_len: int = 3500,
    **kwargs: Any,
) -> List[float]:
    """Penalize empty responses, degenerative n-gram looping, or excessive length.

    Returns:
    0.0: Healthy response without degenerative loops.
    -0.5 to -1.5: Penalties for repetitive token loops, runaway rambling, or empty text.
    """
    rewards: List[float] = []

    for text in completions:
        text = extract_completion_text(text)
        penalty = 0.0

        # Empty or near-empty penalty
        if len(text.strip()) < 40:
            rewards.append(-1.5)
            continue

        # Excessive length penalty (runaway generation)
        if len(text) > max_char_len:
            penalty -= min(1.0, (len(text) - max_char_len) / 1000.0)

        # Repetitive 4-gram loop detection
        words = text.lower().split()
        if len(words) >= 16:
            four_grams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
            unique_ratio = len(set(four_grams)) / len(four_grams)
            if unique_ratio < 0.65:
                # Significant degenerative looping
                penalty -= (0.65 - unique_ratio) * 2.0

        # Hallucinated URL penalty (-0.4)
        if re.search(r"https?://|www\.", text, flags=re.IGNORECASE):
            penalty -= 0.4

        # Forum chatter / casual sign-off penalty (-0.3)
        if re.search(r"\b(hope (this|it) helps|good luck|cheers|let me know if you have questions)\b", text, flags=re.IGNORECASE):
            penalty -= 0.3

        rewards.append(round(penalty, 3))

    return rewards
