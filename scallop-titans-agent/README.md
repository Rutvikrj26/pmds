# Scallop-Titans Agent

A neuro-symbolic reasoning agent combining Google's Titans adaptive memory architecture with the Scallop differentiable logic programming framework.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SCALLOP-TITANS AGENT                           │
├─────────────────────────────────────────────────────────────────────┤
│  LLM (Qwen3-32B)  →  Titans Memory  →  Scallop Engine  →  Response  │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

- **LLM**: Qwen3-32B with special tokens `<|call_scallop|>`, `<|scallop_result|>`
- **Titans Memory**: MLP with fast weights and surprise-based updates
- **Scallop Engine**: Differentiable Datalog reasoning

## Installation

```bash
poetry install
```

## Usage

```bash
# Run demo
poetry run demo

# Train SFT
poetry run train-sft

# Train GRPO
poetry run train-grpo
```

## Development

```bash
# Run tests
poetry run pytest

# Lint
poetry run ruff check .
```

## License

MIT
