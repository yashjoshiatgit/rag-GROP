"""Unit tests for GRPO training configuration and initialization."""

from pathlib import Path
from trl import GRPOConfig
from scripts.train_grpo import build_grpo_reward_funcs


def test_build_grpo_reward_funcs():
    funcs = build_grpo_reward_funcs()
    assert len(funcs) == 4
    names = [fn.__name__ for fn in funcs]
    assert "reward_reasoning_format" in names
    assert "reward_rag_concept_density" in names
    assert "reward_grounding_alignment" in names
    assert "reward_anti_repetition" in names


def test_grpo_config_initialization(tmp_path: Path):
    import torch
    has_cuda = torch.cuda.is_available()
    cfg = GRPOConfig(
        output_dir=str(tmp_path / "out"),
        learning_rate=1e-5,
        num_generations=4,
        max_completion_length=256,
        bf16=False,
        fp16=False,
        use_cpu=not has_cuda,
        gradient_checkpointing=True,
    )
    assert cfg.num_generations == 4
    assert cfg.learning_rate == 1e-5
    assert cfg.gradient_checkpointing is True
