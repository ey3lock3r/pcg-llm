"""TrainingConfig: central configuration dataclass for PCG-LLM.

All training parameters, optimization flags, and platform settings.
Frozen dataclass with __post_init__ validation per contracts/config.md.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Single source of truth for all PCG-LLM training parameters.

    Fields match data-model.md entity 1. Validation rules from contracts/config.md.
    """

    # Model architecture
    hidden_dim: int = 512
    max_seq_len: int = 2048
    vocab_size: int = 128256
    block_size: int = 32

    # Sparsity / RigL
    initial_sparsity: float = 0.85  # Must satisfy SC-004 hard lower bound of 80%
    rigl_interval: int = 100
    rigl_drop_fraction: float = 0.10
    rigl_reroute_start: float = 0.30
    rigl_freeze_step_frac: float = 0.80

    # DEQ solver
    max_solver_iters: int = 12
    solver_tolerance: float = 1e-2
    anderson_window: int = 3
    anderson_beta: float = 1e-4
    lipschitz_target: float = 1.0

    # Loss coefficients
    lambda_sparse: float = 0.01
    gamma_variance: float = 0.01
    variance_floor: float = 0.1

    # EAGLE head
    eagle_k: int = 8
    eagle_draft_len: int = 4
    eagle_accept_threshold: float = 0.65

    # Training hyperparameters
    base_lr: float = 1e-3
    warmup_steps: int = 500
    total_tokens: int = 10_000_000_000
    batch_size: int = 8
    grad_accum_steps: int = 8

    # Optimizer / precision flags
    optimizer: str = "muon_adamw"
    optimizer_bits: int = 32
    normalize: str = "ngpt"
    projection: str = "monarch"
    grad_checkpoint: bool = True
    cpu_offload_mask: bool = False

    # Checkpointing
    checkpoint_interval: int = 500
    checkpoint_backend: str = "local"
    checkpoint_dir: str = "/kaggle/working/checkpoints"

    # Dataset mixing (per-epoch lists, epochs 0..2)
    dataset_fineweb_frac: tuple[float, ...] = (0.92, 0.85, 0.70)
    dataset_stack_frac: tuple[float, ...] = (0.08, 0.15, 0.30)

    # Monitoring
    wandb_project: str | None = "pcg-llm-tiny"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate all fields per contracts/config.md. Raises ValueError on violation."""
        # Mixing fractions must sum to 1.0 per epoch
        if len(self.dataset_fineweb_frac) != len(self.dataset_stack_frac):
            raise ValueError(
                "dataset_fineweb_frac and dataset_stack_frac must have the same length"
            )
        for i, (fw, sv) in enumerate(
            zip(self.dataset_fineweb_frac, self.dataset_stack_frac, strict=False)
        ):
            if abs(fw + sv - 1.0) > 1e-9:
                raise ValueError(
                    f"Mixing fractions must sum to 1.0 at epoch {i}, "
                    f"got {fw} + {sv} = {fw + sv}"
                )

        # Block size must divide max_seq_len
        if self.max_seq_len % self.block_size != 0:
            raise ValueError(
                f"block_size={self.block_size} must divide max_seq_len={self.max_seq_len}"
            )

        # Optimizer bits
        if self.optimizer_bits not in {32, 8}:
            raise ValueError(f"optimizer_bits must be 32 or 8, got {self.optimizer_bits}")

        # Normalize value
        if self.normalize not in {"standard", "ngpt"}:
            raise ValueError(f"normalize must be 'standard' or 'ngpt', got '{self.normalize}'")

        # Projection value
        if self.projection not in {"dense", "monarch"}:
            raise ValueError(f"projection must be 'dense' or 'monarch', got '{self.projection}'")

        # Monarch min dim guard
        if self.projection == "monarch" and self.hidden_dim < 64:
            raise ValueError(
                f"Monarch projection requires hidden_dim >= 64 (64x64 butterfly blocks), "
                f"got hidden_dim={self.hidden_dim}"
            )

        # GCS backend requires gs:// prefix
        if self.checkpoint_backend == "gcs" and not self.checkpoint_dir.startswith("gs://"):
            raise ValueError(
                f"checkpoint_backend='gcs' requires checkpoint_dir to start with 'gs://', "
                f"got '{self.checkpoint_dir}'"
            )

        # Sparsity in (0, 1)
        if not (0.0 < self.initial_sparsity < 1.0):
            raise ValueError(f"initial_sparsity must be in (0.0, 1.0), got {self.initial_sparsity}")

        # Anderson window bounds [1, 10]
        if not (1 <= self.anderson_window <= 10):
            raise ValueError(f"anderson_window must be in [1, 10], got {self.anderson_window}")

        # Eagle draft len [1, 16]
        if not (1 <= self.eagle_draft_len <= 16):
            raise ValueError(f"eagle_draft_len must be in [1, 16], got {self.eagle_draft_len}")

        # Eagle K [1, 32]
        if not (1 <= self.eagle_k <= 32):
            raise ValueError(f"eagle_k must be in [1, 32], got {self.eagle_k}")

        # Solver tolerance must be positive
        if self.solver_tolerance <= 0:
            raise ValueError(f"solver_tolerance must be > 0, got {self.solver_tolerance}")

        # LR must be positive
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {self.base_lr}")

        # Grad accum must be >= 1
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict (JSON-compatible)."""
        d = dataclasses.asdict(self)
        # Convert tuple fields to lists for JSON serialization
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(value)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Reconstruct a TrainingConfig from a plain dict."""
        # Convert list fields back to tuples for frozen dataclass
        list_fields = {"dataset_fineweb_frac", "dataset_stack_frac"}
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            if key in list_fields and isinstance(value, list):
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_preset(cls, preset: str) -> TrainingConfig:
        """Load a named preset configuration.

        Available presets: 'tiny', '3b'.
        """
        if preset == "tiny":
            return cls(
                hidden_dim=512,
                max_seq_len=2048,
                vocab_size=128256,
                block_size=32,
                initial_sparsity=0.85,  # SC-004: hard lower bound is 80%; start above it
                max_solver_iters=12,
                solver_tolerance=1e-2,
                base_lr=1e-3,
                warmup_steps=500,
                total_tokens=10_000_000_000,
                batch_size=8,
                grad_accum_steps=8,
                checkpoint_interval=500,
                checkpoint_backend="local",
                checkpoint_dir="/kaggle/working/checkpoints",
                optimizer="muon_adamw",
                optimizer_bits=32,
                normalize="ngpt",
                projection="monarch",
                grad_checkpoint=True,
                cpu_offload_mask=False,
                wandb_project="pcg-llm-tiny",
            )
        if preset == "3b":
            return cls(
                hidden_dim=3072,
                max_seq_len=4096,
                vocab_size=128256,
                block_size=64,
                initial_sparsity=0.90,
                max_solver_iters=25,
                solver_tolerance=1e-5,
                base_lr=3e-4,
                warmup_steps=2000,
                total_tokens=100_000_000_000,
                batch_size=4,
                grad_accum_steps=8,
                checkpoint_interval=250,
                checkpoint_backend="gcs",
                checkpoint_dir="gs://pcg-llm-checkpoints/3b",
                optimizer="muon_adamw",
                optimizer_bits=8,
                normalize="ngpt",
                projection="monarch",
                grad_checkpoint=True,
                cpu_offload_mask=True,
                wandb_project="pcg-llm-3b",
            )
        raise ValueError(f"Unknown preset '{preset}'. Available presets: 'tiny', '3b'")
