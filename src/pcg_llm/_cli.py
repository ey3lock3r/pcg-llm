"""CLI implementation for PCG-LLM training and evaluation.

Subcommands:
    train          Start or resume training.
    evaluate       Run benchmark evaluation on a checkpoint.
    export-config  Serialize the resolved TrainingConfig to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-count parser  ("10B", "100B", "1T", plain ints)
# ---------------------------------------------------------------------------

def _parse_tokens(value: str) -> int:
    """Parse a token-count string like '10B' or '100B' into an integer."""
    value = value.strip()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    upper = value.upper()
    for suffix, mult in multipliers.items():
        if upper.endswith(suffix):
            try:
                return int(float(upper[:-1]) * mult)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Cannot parse token count '{value}' (expected e.g. '10B', '100M')"
                )
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Cannot parse token count '{value}'"
        )


# ---------------------------------------------------------------------------
# Shared argument registration
# ---------------------------------------------------------------------------

def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add all TrainingConfig-equivalent flags to *parser*."""
    # Config file / preset
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config file (all other flags override it)")
    parser.add_argument("--preset", type=str, choices=["tiny", "3b"], default=None,
                        help="Named preset: 'tiny' or '3b'")

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Node state dimension d (default: 512)")
    parser.add_argument("--seq-len", type=int, default=None, dest="max_seq_len",
                        help="Max training context length N (default: 2048)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="RigL block side length (default: 32)")

    # Sparsity / RigL
    parser.add_argument("--initial-sparsity", type=float, default=None,
                        help="Adjacency initialization sparsity (default: 0.70)")

    # DEQ solver
    parser.add_argument("--max-solver-iters", type=int, default=None,
                        help="DEQ solver iteration cap (default: 12)")
    parser.add_argument("--solver-tol", type=float, default=None, dest="solver_tolerance",
                        help="DEQ convergence threshold (default: 1e-2)")

    # Optimizer / precision
    parser.add_argument("--optimizer", type=str, choices=["adamw", "muon_adamw"], default=None,
                        help="Optimizer type (default: 'muon_adamw')")
    parser.add_argument("--optimizer-bits", type=int, choices=[32, 8], default=None,
                        help="Optimizer state bits: 32 or 8 (default: 32)")
    parser.add_argument("--normalize", type=str, choices=["standard", "ngpt"], default=None,
                        help="Normalization: 'standard' or 'ngpt' (default: 'ngpt')")
    parser.add_argument("--projection", type=str, choices=["dense", "monarch"], default=None,
                        help="Projection: 'dense' or 'monarch' (default: 'monarch')")

    # Training hyperparameters
    parser.add_argument("--base-lr", type=float, default=None,
                        help="Base learning rate (default: 1e-3)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="LR warmup steps (default: 500)")
    parser.add_argument("--total-tokens", type=_parse_tokens, default=None,
                        help="Total training tokens, e.g. '10B' or '100B' (default: '10B')")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU micro-batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=None, dest="grad_accum_steps",
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--grad-checkpoint", action="store_true", default=None,
                        help="Enable gradient checkpointing (default: True)")
    parser.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint",
                        help="Disable gradient checkpointing")
    parser.add_argument("--cpu-offload-mask", action="store_true", default=None,
                        help="Offload RigL mask to CPU (default: False)")
    parser.add_argument("--no-cpu-offload-mask", action="store_false", dest="cpu_offload_mask",
                        help="Do not offload RigL mask to CPU")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Steps between checkpoint saves (default: 500)")
    parser.add_argument("--checkpoint-backend", type=str, choices=["local", "gcs"], default=None,
                        help="Checkpoint backend: 'local' or 'gcs' (default: 'local')")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint storage path (default: '/kaggle/working/checkpoints')")

    # Monitoring
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (None = disable W&B)")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42)")


# ---------------------------------------------------------------------------
# Config resolution: file → preset → CLI overrides → defaults
# ---------------------------------------------------------------------------

def _resolve_config(args: argparse.Namespace) -> "TrainingConfig":  # type: ignore[name-defined]  # noqa: F821
    """Build a TrainingConfig from the parsed args.

    Resolution order (later wins):
    1. TrainingConfig defaults
    2. --config JSON file
    3. --preset named preset
    4. Explicit CLI flags
    """
    from pcg_llm.config import TrainingConfig

    base: dict[str, Any] = {}

    # Step 1: load JSON config file
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        try:
            with open(config_path, encoding="utf-8") as fh:
                file_cfg = json.load(fh)
            base.update(file_cfg)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read config file '{args.config}': {exc}")
            sys.exit(1)

    # Step 2: apply preset (overrides file; later CLI flags override preset)
    preset_config: TrainingConfig | None = None
    if args.preset:
        try:
            preset_config = TrainingConfig.from_preset(args.preset)
            base.update(preset_config.to_dict())
        except ValueError as exc:
            logger.error(f"Invalid preset: {exc}")
            sys.exit(1)

    # Step 3: apply explicit CLI flags (only those the user actually set)
    _CLI_TO_CONFIG: dict[str, str] = {
        "hidden_dim": "hidden_dim",
        "max_seq_len": "max_seq_len",
        "block_size": "block_size",
        "initial_sparsity": "initial_sparsity",
        "max_solver_iters": "max_solver_iters",
        "solver_tolerance": "solver_tolerance",
        "optimizer": "optimizer",
        "optimizer_bits": "optimizer_bits",
        "normalize": "normalize",
        "projection": "projection",
        "base_lr": "base_lr",
        "warmup_steps": "warmup_steps",
        "total_tokens": "total_tokens",
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "grad_checkpoint": "grad_checkpoint",
        "cpu_offload_mask": "cpu_offload_mask",
        "checkpoint_interval": "checkpoint_interval",
        "checkpoint_backend": "checkpoint_backend",
        "checkpoint_dir": "checkpoint_dir",
        "wandb_project": "wandb_project",
    }

    for attr, cfg_key in _CLI_TO_CONFIG.items():
        val = getattr(args, attr, None)
        if val is not None:
            base[cfg_key] = val

    # Build the config; convert list fields to tuples where needed
    list_to_tuple = {"dataset_fineweb_frac", "dataset_stack_frac"}
    kwargs: dict[str, Any] = {}
    for k, v in base.items():
        if k in list_to_tuple and isinstance(v, list):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v

    try:
        if base:
            return TrainingConfig(**kwargs)
        # No overrides at all — return the default config
        return TrainingConfig()
    except (ValueError, TypeError) as exc:
        logger.error(f"Configuration error: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> int:
    """Execute the train subcommand."""
    import random

    import torch

    from pcg_llm.training.trainer import PCGTrainer

    # Seed
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    config = _resolve_config(args)

    # Validate optimizer_bits=8 dependency
    if config.optimizer_bits == 8:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            logger.error(
                "--optimizer-bits 8 requires bitsandbytes to be installed. "
                "Install it with: pip install bitsandbytes"
            )
            return 1

    resume = getattr(args, "resume", True)

    trainer = PCGTrainer(config=config)

    # Build a synthetic dataloader for standalone runs (real data comes from DataCurriculum)
    def _synthetic_loader():
        """Infinite synthetic token stream for testing/smoke runs."""
        while True:
            yield torch.randint(
                0, config.vocab_size, (config.batch_size, config.max_seq_len)
            )

    dataloader = _synthetic_loader()

    try:
        trainer.train(dataloader=dataloader, resume=resume)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    except KeyboardInterrupt:
        return 2
    except Exception as exc:
        logger.exception(f"Fatal training error: {exc}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate subcommand."""
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    tasks = getattr(args, "tasks", "arc_challenge,gsm8k,humaneval,hellaswag,mmlu")
    quantize = getattr(args, "quantize", "4bit")
    output_path = getattr(args, "output", None)

    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Quantization: {quantize}")

    try:
        import torch
        ckpt = torch.load(checkpoint_path, weights_only=False)
        config_dict = ckpt.get("config", {})

        from pcg_llm.config import TrainingConfig
        config = TrainingConfig.from_dict(config_dict) if config_dict else TrainingConfig()
    except Exception as exc:
        logger.error(f"Failed to load checkpoint: {exc}")
        return 1

    # Attempt to use BenchmarkHarness if available; otherwise emit a stub result
    results: dict[str, Any] = {}
    try:
        from pcg_llm.evaluation.harness import BenchmarkHarness
        harness = BenchmarkHarness(config=config, quantize=quantize)
        results = harness.run(
            checkpoint_path=str(checkpoint_path),
            tasks=tasks.split(","),
        )
    except ImportError:
        logger.warning("BenchmarkHarness not available; producing stub evaluation output")
        for task in tasks.split(","):
            results[task.strip()] = {"score": None, "note": "harness unavailable"}
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        return 1

    output_json = json.dumps(results, indent=2)
    if output_path:
        try:
            Path(output_path).write_text(output_json, encoding="utf-8")
            logger.info(f"Results written to {output_path}")
        except OSError as exc:
            logger.error(f"Could not write results to '{output_path}': {exc}")
            return 1
    else:
        print(output_json)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: export-config
# ---------------------------------------------------------------------------

def _cmd_export_config(args: argparse.Namespace) -> int:
    """Execute the export-config subcommand."""
    config = _resolve_config(args)
    output_path = getattr(args, "output", "config.json")

    config_dict = config.to_dict()
    output_json = json.dumps(config_dict, indent=2)

    try:
        Path(output_path).write_text(output_json, encoding="utf-8")
        logger.info(f"Config exported to {output_path}")
        print(f"Config exported to {output_path}")
    except OSError as exc:
        logger.error(f"Could not write config to '{output_path}': {exc}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcg_llm",
        description="PCG-LLM: Probabilistic Circuit Graph Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Start or resume PCG-LLM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(train_parser)
    train_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Auto-detect and resume from latest checkpoint (default: True)",
    )
    train_parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start training from scratch, ignoring any existing checkpoints",
    )
    train_parser.set_defaults(func=_cmd_train)

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run benchmark evaluation on a saved checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file",
    )
    eval_parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,gsm8k,humaneval,hellaswag,mmlu",
        help="Comma-separated benchmark tasks",
    )
    eval_parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help="Quantization mode for inference",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON (default: stdout)",
    )
    eval_parser.set_defaults(func=_cmd_evaluate)

    # ------------------------------------------------------------------
    # export-config
    # ------------------------------------------------------------------
    export_parser = subparsers.add_parser(
        "export-config",
        help="Export the resolved TrainingConfig to a JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_args(export_parser)
    export_parser.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="Output JSON path (default: config.json)",
    )
    export_parser.set_defaults(func=_cmd_export_config)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand handler.

    Returns an exit code: 0 (success), 1 (error), 2 (SIGTERM / interrupt).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = getattr(args, "log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Dispatch
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    return func(args)
