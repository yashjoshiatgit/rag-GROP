"""Constructs the Held-Out RAG Evaluation Benchmark.

Builds a rigorous, multi-domain evaluation benchmark across all 6 RAG taxonomy areas:
- retrieval
- chunking_and_indexing
- generation_and_grounding
- evaluation_and_benchmarks
- agentic_and_graph_rag
- tooling_and_frameworks

Each test case contains:
- id: unique benchmark identifier
- category: taxonomy area
- difficulty: easy / medium / hard
- test_type: conceptual / diagnostic / architectural
- question: the evaluation prompt given to candidate models
- expected_concepts: list of essential technical terms and mechanisms
- reference_reasoning: the gold-standard step-by-step reasoning
- scoring_rubric: explicit criteria for correctness, grounding, and trade-offs

Outputs:
- data/evaluation/rag_eval.jsonl
- data/evaluation/rag_eval_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.benchmark")

BENCHMARK_SUITE: List[Dict[str, Any]] = [
    # =========================================================================
    # 1. RETRIEVAL
    # =========================================================================
    {
        "id": "rag_ret_001",
        "category": "retrieval",
        "difficulty": "medium",
        "test_type": "diagnostic",
        "question": (
            "A production RAG system retrieves documents using dense cosine similarity with OpenAI text-embedding-3-small. "
            "Users report that queries containing specific part numbers (e.g., 'SKU-8921-X') consistently return irrelevant "
            "general catalog pages instead of the exact product specifications. Why does this happen, and what is the standard architectural fix?"
        ),
        "expected_concepts": [
            "dense embeddings",
            "out-of-vocabulary tokens",
            "lexical search",
            "BM25 / sparse retrieval",
            "hybrid search",
            "Reciprocal Rank Fusion (RRF)",
        ],
        "reference_reasoning": (
            "Dense bi-encoders map text into a continuous semantic vector space. Exact technical identifiers like SKU numbers, "
            "serial codes, or hashes often get sub-tokenized into generic byte-pair fragments that lack strong semantic clusters, "
            "causing dense similarity to favor broader generic text. The standard architectural fix is Hybrid Search: combining "
            "sparse lexical retrieval (BM25 or SPLADE) with dense vector retrieval, fused via Reciprocal Rank Fusion (RRF) or convex score combination, "
            "optionally followed by a cross-encoder reranker."
        ),
        "scoring_rubric": {
            "correctness": "Accurately diagnoses the tokenization/semantic limitation of dense bi-encoders on exact identifiers.",
            "grounding": "Prescribes hybrid search (BM25/sparse + dense) and a fusion method like RRF.",
            "tradeoffs": "Mentions trade-offs in index maintenance or latency from combining two retrieval paths.",
        },
    },
    {
        "id": "rag_ret_002",
        "category": "retrieval",
        "difficulty": "hard",
        "test_type": "architectural",
        "question": (
            "Compare bi-encoder dense retrieval (e.g., bge-large) against cross-encoder reranking (e.g., bge-reranker-large). "
            "Why can't we simply use a cross-encoder across the entire document collection of 10 million passages?"
        ),
        "expected_concepts": [
            "bi-encoder",
            "cross-encoder",
            "computational complexity",
            "cross-attention",
            "offline indexing",
            "two-stage retrieval",
        ],
        "reference_reasoning": (
            "Bi-encoders encode queries and documents independently into fixed-dimensional vectors, allowing all 10M document "
            "embeddings to be precomputed offline and queried in sub-50ms using approximate nearest neighbor (ANN) search. In contrast, "
            "cross-encoders process the query and document concatenated together in a single transformer pass with full cross-attention "
            "between all query and document tokens (O(L^2) attention complexity). Running a cross-encoder over 10 million pairs for every "
            "live search query would require 10 million transformer forward passes per request, which is computationally intractable at run-time. "
            "Therefore, industry standard architecture uses a two-stage pipeline: fast bi-encoder retrieval for the top-100 candidates, "
            "followed by cross-encoder reranking on only those top-100."
        ),
        "scoring_rubric": {
            "correctness": "Distinguishes independent vector representation vs full cross-attention mechanism.",
            "grounding": "Explains why full-corpus cross-encoding is computationally prohibitive at 10M scale.",
            "tradeoffs": "Outlines the standard two-stage retrieval + rerank architecture.",
        },
    },
    {
        "id": "rag_ret_003",
        "category": "retrieval",
        "difficulty": "medium",
        "test_type": "diagnostic",
        "question": (
            "What is Hypothetical Document Embeddings (HyDE), and in what specific scenario does HyDE degrade retrieval performance instead of improving it?"
        ),
        "expected_concepts": [
            "HyDE",
            "zero-shot generation",
            "hallucination propagation",
            "domain-specific terminology",
            "semantic drift",
        ],
        "reference_reasoning": (
            "HyDE uses an LLM to generate a hypothetical answer to a user query, then embeds that synthetic document to retrieve real documents "
            "via vector similarity. This helps bridge the semantic gap between short questions and long answer passages. However, HyDE degrades "
            "performance when: 1) The query involves niche, proprietary, or rapidly changing facts where the generator hallucinates plausible "
            "but factually wrong concepts, steering vector search toward completely incorrect documents; 2) The query is already specific and lexical, "
            "where HyDE introduces semantic drift."
        ),
        "scoring_rubric": {
            "correctness": "Accurately defines HyDE mechanism and embedding strategy.",
            "grounding": "Identifies hallucinated premise leading to semantic drift as the failure mode.",
            "tradeoffs": "Notes the latency cost of an extra LLM generation step before retrieval.",
        },
    },

    # =========================================================================
    # 2. CHUNKING & INDEXING
    # =========================================================================
    {
        "id": "rag_idx_001",
        "category": "chunking_and_indexing",
        "difficulty": "medium",
        "test_type": "architectural",
        "question": (
            "A developer splits documents into 2048-token chunks to ensure the LLM has maximum context. "
            "However, retrieval accuracy plummets compared to using 256-token chunks. Explain the technical phenomenon "
            "behind this degradation in embedding search."
        ),
        "expected_concepts": [
            "embedding dilution",
            "semantic pooling",
            "vector representational capacity",
            "granularity mismatch",
            "chunk size trade-off",
        ],
        "reference_reasoning": (
            "Standard embedding models compress an entire passage into a single fixed-size vector (e.g., 768 or 1536 dimensions) "
            "using mean pooling. When a chunk spans 2048 tokens across multiple disparate sub-topics, the single pooled vector becomes "
            "an average of many different semantic concepts ('vector dilution'). Specific facts get washed out by surrounding text. "
            "Smaller chunks (256–512 tokens) maintain sharp, localized semantic focus that matches narrow user queries closely. "
            "The standard solution to balance retrieval precision and generation context is hierarchical chunking (small-to-big) or "
            "parent-document retrieval, where search hits small chunks but feeds the parent section to the LLM."
        ),
        "scoring_rubric": {
            "correctness": "Explains embedding dilution caused by mean pooling over heterogeneous subtopics.",
            "grounding": "Identifies the trade-off between localized search specificity and contextual completeness.",
            "tradeoffs": "Recommends small-to-big / parent-document retrieval as the architectural mitigation.",
        },
    },
    {
        "id": "rag_idx_002",
        "category": "chunking_and_indexing",
        "difficulty": "hard",
        "test_type": "architectural",
        "question": (
            "In an HNSW (Hierarchical Navigable Small World) vector index, what do the hyperparameters 'M', 'efConstruction', "
            "and 'efSearch' control, and what is the trade-off of increasing efSearch at query time?"
        ),
        "expected_concepts": [
            "HNSW graph",
            "M (bidirectional links)",
            "efConstruction",
            "efSearch",
            "recall vs latency",
            "priority queue size",
        ],
        "reference_reasoning": (
            "In HNSW: 'M' is the maximum number of bidirectional connections per node in each graph layer (higher M improves recall on high-dimensional data at the cost of larger index RAM). "
            "'efConstruction' is the size of the dynamic candidate list evaluated during index construction (higher values build a higher-quality graph but increase build time). "
            "'efSearch' is the size of the dynamic priority queue maintained during runtime search. "
            "Trade-off: Increasing efSearch at query time explores deeper into the graph, monotonically increasing Recall@K toward exact kNN, "
            "but linearly increases query latency (p99 latency) and CPU utilization as more distance computations must be performed."
        ),
        "scoring_rubric": {
            "correctness": "Correctly defines all three parameters in the HNSW graph topology.",
            "grounding": "Identifies that efSearch controls runtime priority queue depth.",
            "tradeoffs": "Clearly states the Recall@K vs query latency / CPU consumption trade-off.",
        },
    },
    {
        "id": "rag_idx_003",
        "category": "chunking_and_indexing",
        "difficulty": "medium",
        "test_type": "diagnostic",
        "question": (
            "Explain the difference between pre-filtering and post-filtering when applying metadata filters in a vector database "
            "(e.g., filtering by tenant_id = 'org_42'). Why can naive post-filtering cause a query to return zero results even when matching documents exist?"
        ),
        "expected_concepts": [
            "pre-filtering",
            "post-filtering",
            "top-k cutoff",
            "tenant isolation",
            "vector payload filtering",
            "HNSW graph traversal",
        ],
        "reference_reasoning": (
            "Post-filtering executes approximate nearest neighbor search across the entire vector space first, retrieves the top-K nearest neighbors, "
            "and then discards any candidates that do not match the metadata filter (tenant_id). If all top-K globally nearest vectors happen to belong "
            "to other tenants, post-filtering discards all K candidates, returning 0 results even if tenant 'org_42' has hundreds of relevant documents in the index. "
            "Pre-filtering restricts the search to vectors meeting the metadata criteria before or during graph traversal (e.g. filtered HNSW traversal or inverted index intersection), "
            "guaranteeing that all K returned results satisfy the metadata constraint."
        ),
        "scoring_rubric": {
            "correctness": "Explains top-K vector search before filter vs filtered graph traversal.",
            "grounding": "Explains zero-result failure mode when global top-K is dominated by other metadata groups.",
            "tradeoffs": "Advocates for single-stage filtered graph traversal or pre-filtering in multi-tenant systems.",
        },
    },

    # =========================================================================
    # 3. GENERATION & GROUNDING
    # =========================================================================
    {
        "id": "rag_gen_001",
        "category": "generation_and_grounding",
        "difficulty": "medium",
        "test_type": "diagnostic",
        "question": (
            "What is the 'Lost in the Middle' phenomenon in transformer context windows, and how should retrieved RAG chunks "
            "be positioned in the prompt to maximize generation accuracy?"
        ),
        "expected_concepts": [
            "Lost in the Middle",
            "primacy bias",
            "recency bias",
            "chunk ordering",
            "attention degradation",
        ],
        "reference_reasoning": (
            "Research (Liu et al.) demonstrated that decoder-only language models show U-shaped performance curves across long contexts: "
            "information at the very beginning (primacy bias) and at the very end (recency bias) is attended to with high fidelity, while "
            "information positioned in the middle of long context windows suffers significant degradation in retrieval adherence. "
            "To maximize generation accuracy, the highest-relevance retrieved chunks should be placed at the very top (immediately following instructions) "
            "and at the very bottom (immediately preceding the final user question), rather than burying the most relevant evidence in the center."
        ),
        "scoring_rubric": {
            "correctness": "Accurately identifies the U-shaped attention curve and primacy/recency bias.",
            "grounding": "Explains why middle chunks suffer attention decay.",
            "tradeoffs": "Recommends positioning top-ranked chunks at the extremities of the context.",
        },
    },
    {
        "id": "rag_gen_002",
        "category": "generation_and_grounding",
        "difficulty": "hard",
        "test_type": "architectural",
        "question": (
            "How can a system design engineer enforce strict citation attribution and prevent hallucinated references in a RAG response? "
            "Describe at least two independent validation layers beyond standard system prompting."
        ),
        "expected_concepts": [
            "citation attribution",
            "verbatim quote extraction",
            "string matching / fuzzy verification",
            "nli / entailment model",
            "hallucination mitigation",
        ],
        "reference_reasoning": (
            "System prompting alone ('cite sources') is non-deterministic and frequently generates phantom citations. "
            "Two robust architectural validation layers are: "
            "1) Verbatim Quote-Forcing with String Verification: Constrain the generator to output exact supporting quote spans "
            "enclosed in specific tags (e.g. <quote>...</quote>). A deterministic post-processor verifies that the quoted string exists "
            "character-for-character (or via fuzzy Levenshtein distance) in the source chunk. Quotes failing verification are rejected as NOT_FOUND. "
            "2) Natural Language Inference (NLI) Entailment Gate: Run a compact cross-encoder NLI model (e.g. DeBERTa-v3-large-MNLI) taking "
            "the source passage as premise and the generated claim as hypothesis. If P(entailment) falls below a strict threshold (e.g. 0.85), "
            "the claim is flagged as ungrounded and suppressed."
        ),
        "scoring_rubric": {
            "correctness": "Goes beyond prompting to describe deterministic and model-based verification.",
            "grounding": "Explains verbatim substring verification and NLI premise-hypothesis verification.",
            "tradeoffs": "Notes latency overhead of post-generation verification passes.",
        },
    },

    # =========================================================================
    # 4. EVALUATION & BENCHMARKS
    # =========================================================================
    {
        "id": "rag_eval_001",
        "category": "evaluation_and_benchmarks",
        "difficulty": "medium",
        "test_type": "conceptual",
        "question": (
            "In RAG evaluation frameworks (such as Ragas or TruLens), distinguish between 'Faithfulness' (or Groundedness) and 'Answer Relevance'. "
            "Can a response have 100% Faithfulness but 0% Answer Relevance? Give a concrete example."
        ),
        "expected_concepts": [
            "faithfulness / groundedness",
            "answer relevance",
            "retrieved context adherence",
            "query addressing",
            "independent metrics",
        ],
        "reference_reasoning": (
            "Faithfulness measures whether every claim made in the generated answer is strictly entailed by and grounded in the retrieved context, "
            "penalizing hallucinations. Answer Relevance measures whether the generated response directly and completely answers the user's specific query, "
            "penalizing evasive, tangential, or off-topic responses. "
            "Yes, a response can have 100% Faithfulness and 0% Answer Relevance: "
            "User Query: 'How do I resolve CUDA Out of Memory during LoRA fine-tuning?' "
            "Retrieved Context: 'PyTorch was initially released in 2016 by Meta AI.' "
            "Generated Answer: 'PyTorch was released in 2016 by Meta AI.' "
            "The answer is 100% faithful to the context (zero hallucination), but 0% relevant to the user's technical question."
        ),
        "scoring_rubric": {
            "correctness": "Accurately defines faithfulness vs answer relevance.",
            "grounding": "Provides a valid example where an answer is 100% faithful to context but fails relevance completely.",
            "tradeoffs": "Explains why measuring retrieval and generation separately is mandatory.",
        },
    },
    {
        "id": "rag_eval_002",
        "category": "evaluation_and_benchmarks",
        "difficulty": "hard",
        "test_type": "diagnostic",
        "question": (
            "When using an LLM-as-a-judge to evaluate RAG answer quality, what are the three most prevalent systematic biases, "
            "and what calibration techniques neutralize them?"
        ),
        "expected_concepts": [
            "LLM-as-a-judge",
            "verbosity bias",
            "position bias",
            "self-enhancement bias",
            "swap evaluation",
            "reference-guided rubrics",
        ],
        "reference_reasoning": (
            "The three major systematic biases in LLM-as-a-judge are: "
            "1) Verbosity Bias: Judges favor longer, wordier answers even when they contain padding, repetition, or lower information density. "
            "2) Position Bias: In pairwise comparisons, judges systematically favor whichever candidate is presented first (or second, depending on model). "
            "3) Self-Enhancement / Egocentric Bias: Models systematically rate generations from their own model family higher than competitors. "
            "Calibration techniques: "
            "- Position swapping: Run evaluation twice with candidate orders inverted [A, B] and [B, A], only counting wins when consistent. "
            "- De-biasing length: Normalize scores by length or constrain candidates to identical token ceilings. "
            "- Few-shot reference anchors: Provide explicit scoring rubrics with anchor examples for each ordinal grade (1 to 5) rather than unconstrained judging."
        ),
        "scoring_rubric": {
            "correctness": "Identifies verbosity, position, and self-enhancement biases.",
            "grounding": "Explains specific mechanisms (length correlation, ordering artifacts).",
            "tradeoffs": "Provides actionable calibration strategies (position swap, anchor rubrics).",
        },
    },

    # =========================================================================
    # 5. ADVANCED RAG
    # =========================================================================
    {
        "id": "rag_adv_001",
        "category": "agentic_and_graph_rag",
        "difficulty": "medium",
        "test_type": "architectural",
        "question": (
            "What is Corrective RAG (CRAG), and how does its retrieval confidence evaluator dynamically route between "
            "document refinement and external search?"
        ),
        "expected_concepts": [
            "Corrective RAG (CRAG)",
            "retrieval evaluator",
            "confidence threshold",
            "correct / incorrect / ambiguous",
            "external web search fallback",
            "strip irrelevant sentences",
        ],
        "reference_reasoning": (
            "Corrective RAG (CRAG) introduces a dedicated lightweight retrieval evaluator to assess the quality of retrieved documents "
            "before generation, outputting a confidence score partitioned into three trigger states: "
            "1) Correct (High Confidence): Retrieved passages are confirmed relevant; an internal strip-and-refine algorithm filters out "
            "irrelevant sentences within the chunks, preserving high information density. "
            "2) Incorrect (Low Confidence): Retrieved passages are deemed irrelevant; CRAG discards the internal context entirely and triggers "
            "an external fallback search (e.g. web search API) to retrieve fresh grounding data. "
            "3) Ambiguous (Moderate Confidence): Combines both refined internal chunks and external search results to ensure robust coverage."
        ),
        "scoring_rubric": {
            "correctness": "Defines the 3-state routing mechanism of CRAG.",
            "grounding": "Explains the retrieval evaluation module and external fallback path.",
            "tradeoffs": "Notes the latency cost of evaluating retrieval before proceeding to generation.",
        },
    },
    {
        "id": "rag_adv_002",
        "category": "agentic_and_graph_rag",
        "difficulty": "hard",
        "test_type": "architectural",
        "question": (
            "In what specific problem domain does GraphRAG (knowledge graph-based RAG) fundamentally outperform chunk-based vector search, "
            "and what are the primary engineering bottlenecks in building a GraphRAG index?"
        ),
        "expected_concepts": [
            "GraphRAG",
            "multi-hop reasoning",
            "relational connectivity",
            "global dataset summarization",
            "entity extraction bottleneck",
            "graph construction cost",
        ],
        "reference_reasoning": (
            "GraphRAG fundamentally outperforms vector search in two domains: "
            "1) Multi-hop Relational Reasoning: Answering queries where facts are separated across multiple distant documents connected by "
            "relationships (e.g. 'How did Person A's subsidiary acquire Company B's patent through Entity C?'). Vector search retrieves isolated chunks, "
            "missing the relational chain. "
            "2) Global Sensemaking / Dataset Summarization: Answering holistic queries like 'What are the main themes across all 50,000 incident reports?' "
            "Vector search only retrieves top-K localized chunks, whereas GraphRAG clusters entities into communities and creates hierarchical community summaries. "
            "Engineering bottlenecks: "
            "- Massive LLM extraction cost: Extracting entities, relationships, and claims across entire corpora requires thousands of LLM calls. "
            "- Entity deduplication & resolution: Merging entity aliases (e.g. 'Apple Inc.', 'Apple', 'AAPL') at scale. "
            "- Graph traversal latency vs ANN vector speed."
        ),
        "scoring_rubric": {
            "correctness": "Accurately contrasts localized vector top-K vs multi-hop relational traversal and global community summaries.",
            "grounding": "Identifies LLM entity extraction cost and entity resolution as primary bottlenecks.",
            "tradeoffs": "Discusses high index construction overhead vs query-time benefits.",
        },
    },

    # =========================================================================
    # 6. TOOLING & FRAMEWORKS
    # =========================================================================
    {
        "id": "rag_tool_001",
        "category": "tooling_and_frameworks",
        "difficulty": "medium",
        "test_type": "architectural",
        "question": (
            "In a production RAG application serving 50,000 active users, the vector database index needs to be updated hourly as new documents arrive. "
            "How do you implement hot-swapping of vector indices without taking retrieval offline or experiencing latency spikes?"
        ),
        "expected_concepts": [
            "hot-swapping",
            "blue-green indexing",
            "memory-mapped files",
            "shadow index / read replica",
            "atomic pointer swap",
            "zero downtime",
        ],
        "reference_reasoning": (
            "To achieve zero-downtime hourly index updates without latency spikes: "
            "1) Blue-Green Indexing (Shadow Indexing): Build the new index (e.g. updated FAISS or Qdrant collection) offline on a separate background worker "
            "or shadow collection while the active index continues serving live read traffic. "
            "2) Memory-Mapped Files (mmap): Use memory-mapped vector stores so loading the new index into RAM is instantaneous (lazy page cache loading) "
            "rather than doing slow deserialization. "
            "3) Atomic Pointer Swap: Once the shadow index reaches readiness, atomically swap an in-memory pointer or load-balancer route to redirect incoming "
            "queries to the new collection, followed by a graceful teardown of the old index after in-flight queries drain."
        ),
        "scoring_rubric": {
            "correctness": "Explains background shadow building followed by atomic routing/pointer swap.",
            "grounding": "Highlights the use of memory-mapping to avoid memory allocation latency spikes.",
            "tradeoffs": "Accounts for the temporary 2x disk/memory overhead during the shadow rebuild.",
        },
    },
]


def build_benchmark(
    output_jsonl: Path = Path("data/evaluation/rag_eval.jsonl"),
    output_metadata: Path = Path("data/evaluation/rag_eval_metadata.json"),
) -> Dict[str, Any]:
    """Export benchmark suite and metadata."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in BENCHMARK_SUITE:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    category_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    test_type_counts: Dict[str, int] = {}

    for item in BENCHMARK_SUITE:
        cat = item["category"]
        diff = item["difficulty"]
        tt = item["test_type"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        test_type_counts[tt] = test_type_counts.get(tt, 0) + 1

    metadata = {
        "benchmark_name": "RAGForge-Eval-Core",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(BENCHMARK_SUITE),
        "categories": category_counts,
        "difficulties": difficulty_counts,
        "test_types": test_type_counts,
        "isolation_rule": "MUST NOT BE INCLUDED IN ANY TRAINING SPLIT (ZERO DATA CONTAMINATION).",
        "output_file": str(output_jsonl),
    }

    with output_metadata.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Built benchmark with {len(BENCHMARK_SUITE)} questions -> {output_jsonl}")
    logger.info(f"Categories: {category_counts}")
    logger.info(f"Metadata saved -> {output_metadata}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build RAG evaluation benchmark.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/evaluation/rag_eval.jsonl"),
        help="Path to output benchmark JSONL",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("data/evaluation/rag_eval_metadata.json"),
        help="Path to output benchmark metadata JSON",
    )
    args = parser.parse_args()

    build_benchmark(output_jsonl=args.output_jsonl, output_metadata=args.output_metadata)


if __name__ == "__main__":
    main()
