# RAGForge (RAG-RL)

> **Training compact, high-reasoning open-weights language models for RAG via Reinforcement Learning (TRL + GRPOTrainer) grounded on real-world engineering discourse.**

---

## 🧭 Overview

RAGForge explores post-training reinforcement learning strategies (GRPO, SFT, direct RL) to enhance reasoning, citation accuracy, and hallucination diagnosis in compact models.

```
                    RAG-RL
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
   DATASET         BASELINE       TRAINING
       │              │              │
       ↓              ↓              ↓
   RAG Corpus     Model Eval     SFT/Domain
       │              │              ↓
       ↓              │             GRPO
   Benchmark ─────────┴──────────────┤
                                     ↓
                               Final Evaluation
                                     ↓
                               Hugging Face
```

---

## 🛠️ Stack & Reproducibility

- **Environment & Package Manager:** `uv` with committed `uv.lock` for cross-platform parity (Windows dev workstation <-> WSL2/Ubuntu GPU rig)
- **Language:** Python 3.11 / 3.12
- **ML Framework:** PyTorch, Hugging Face `transformers`, `datasets`
- **Post-Training / RL:** Hugging Face `trl` (`GRPOTrainer`), `peft` (LoRA/QLoRA)
- **Decision Engine / Curation:** `laya` (non-autoregressive typed decisions)
- **Data Engineering:** `polars`, `pyarrow` (Parquet + JSONL)
- **Quality & Testing:** `ruff`, `pytest`, `pydantic`

---

## 📁 Repository Structure

```
rag-rl/
│
├── data/
│   ├── raw/
│   ├── interim/       # Columnar normalized & cleaned candidates (gitignored)
│   ├── processed/     # Laya-curated & taxonomy-classified RAG corpus (gitignored)
│   └── evaluation/    # Held-out RAG evaluation benchmark
│
├── src/
│   └── rag_rl/        # Core library: reward functions, trainers, dataset builders
│
├── scripts/           # Ingestion, cleaning, evaluation, and training entry points
├── configs/           # Pydantic/YAML experiment configurations
├── tests/             # Unit and integration test suite
├── notebooks/         # Exploratory analysis & evaluation notebooks
├── experiments/       # Experiment run artifacts, metrics, and logs
├── outputs/           # Ingestion reports, statistics, and evaluation figures
│
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Exact reproducible dependency lockfile
├── README.md          # Project documentation
├── .gitignore         # Strict exclusion for raw data and model weights
└── LICENSE            # Apache 2.0 License
```

---

## 🚀 Quickstart

### 1. Install `uv`
If you haven't installed `uv`:
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync Environment
Install the exact locked environment:
```bash
uv sync --all-extras
```

### 3. Run Quality Checks
```bash
uv run ruff check .
uv run pytest
```

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
