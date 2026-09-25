"""TRL GRPO Policy Optimization Training Pipeline for RAG-RL.

Trains compact open-weights models (e.g., Qwen/Qwen2.5-1.5B-Instruct or
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using Group Relative Policy Optimization (GRPO)
grounded on community RAG engineering discourse.

Key Features:
- Low-VRAM design: LoRA (PEFT r=16), gradient checkpointing, bfloat16/float16.
- Multi-component rule-based reward suite:
  1. reward_reasoning_format (enforces <think>...</think> and <answer>...</answer>)
  2. reward_rag_concept_density (awards domain-specific RAG architectural concepts)
  3. reward_grounding_alignment (alignment with verified community solutions)
  4. reward_anti_repetition (anti-looping and length penalty)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.rag_rl.rewards import (
    reward_anti_repetition,
    reward_grounding_alignment,
    reward_rag_concept_density,
    reward_reasoning_format,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_rl.train_grpo")


def build_grpo_reward_funcs():
    """Return the ordered composite reward functions for GRPOTrainer."""
    return [
        reward_reasoning_format,
        reward_rag_concept_density,
        reward_grounding_alignment,
        reward_anti_repetition,
    ]


def run_grpo_training(
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = 250,
    learning_rate: float = 1.5e-5,
    num_generations: int = 4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_prompt_length: int = 512,
    max_completion_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    sample_limit: Optional[int] = None,
    save_steps: int = 50,
    logging_steps: int = 10,
) -> None:
    """Execute GRPO training run."""
    logger.info("=" * 70)
    logger.info("Initializing RAGForge GRPO Reinforcement Learning Training")
    logger.info(f"Model ID:              {model_id}")
    logger.info(f"Dataset Path:          {dataset_path}")
    logger.info(f"Output Directory:      {output_dir}")
    logger.info(f"Group Size (G):        {num_generations}")
    logger.info(f"Learning Rate:         {learning_rate}")
    logger.info(f"Max Steps:             {max_steps}")
    logger.info(f"LoRA Rank (r) / Alpha: {lora_r} / {lora_alpha}")
    logger.info("=" * 70)

    if not dataset_path.exists():
        raise FileNotFoundError(f"GRPO training dataset not found at: {dataset_path}")

    # Load dataset
    logger.info("Loading prompt dataset from parquet...")
    if str(dataset_path).endswith(".parquet"):
        import polars as pl
        df = pl.read_parquet(dataset_path)
        if sample_limit and sample_limit > 0:
            df = df.head(sample_limit)
        dataset = Dataset.from_pandas(df.to_pandas())
    else:
        dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
        if sample_limit and sample_limit > 0:
            dataset = dataset.select(range(min(sample_limit, len(dataset))))

    logger.info(f"Loaded {len(dataset)} training prompts.")

    # Precision setup
    has_cuda = torch.cuda.is_available()
    has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not has_bf16

    logger.info(f"Compute Backend: CUDA={has_cuda}, bf16={has_bf16}, fp16={use_fp16}")

    # LoRA PEFT Configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # GRPO Training Configuration
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=has_bf16,
        use_cpu=not has_cuda,
        gradient_checkpointing=True,
        report_to="none",  # Can be switched to 'wandb'
    )

    reward_funcs = build_grpo_reward_funcs()
    logger.info(f"Registered {len(reward_funcs)} reward functions: {[fn.__name__ for fn in reward_funcs]}")

    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    logger.info("Starting GRPO optimization loop...")
    trainer.train()

    logger.info(f"Training completed. Saving final adapter weights to: {output_dir}")
    trainer.save_model(str(output_dir))
    logger.info("GRPO policy training pipeline successfully finished!")


def main():
    parser = argparse.ArgumentParser(description="Run GRPO Reinforcement Learning for RAG.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model ID from HuggingFace",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/grpo_train_dataset.parquet"),
        help="Path to grpo_train_dataset.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_qwen_1.5b_adapter"),
        help="Directory to save fine-tuned LoRA adapter",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Total training optimization steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-5,
        help="Peak learning rate for LoRA adapter",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Group size G for GRPO relative advantage computation",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of training samples for quick testing",
    )

    args = parser.parse_args()

    run_grpo_training(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
