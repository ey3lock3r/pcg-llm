"""Unit tests for src/pcg_llm/_cli.py (0% coverage module)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _parse_tokens
# ---------------------------------------------------------------------------


class TestParseTokens:
    """Tests for the _parse_tokens helper."""

    def _fn(self):
        from pcg_llm._cli import _parse_tokens
        return _parse_tokens

    def test_10b(self):
        assert self._fn()("10B") == 10_000_000_000

    def test_100m(self):
        assert self._fn()("100M") == 100_000_000

    def test_1k(self):
        assert self._fn()("1K") == 1_000

    def test_plain_int(self):
        assert self._fn()("500") == 500

    def test_1t(self):
        assert self._fn()("1T") == 1_000_000_000_000

    def test_case_insensitive_lowercase_b(self):
        assert self._fn()("5b") == 5_000_000_000

    def test_invalid_string_raises(self):
        from pcg_llm._cli import _parse_tokens
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_tokens("notanumber")

    def test_invalid_suffix_with_text_raises(self):
        from pcg_llm._cli import _parse_tokens
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_tokens("abcB")


# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for the _build_parser factory."""

    def test_returns_argument_parser(self):
        from pcg_llm._cli import _build_parser
        parser = _build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_export_config_subcommand_exists(self):
        from pcg_llm._cli import _build_parser
        parser = _build_parser()
        # Should not raise — export-config is a valid subcommand
        args = parser.parse_args(["export-config", "--output", "dummy.json"])
        assert args.command == "export-config"

    def test_train_subcommand_exists(self):
        from pcg_llm._cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"

    def test_evaluate_subcommand_exists(self):
        from pcg_llm._cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["evaluate", "--checkpoint", "ckpt.pt"])
        assert args.command == "evaluate"


# ---------------------------------------------------------------------------
# main — export-config integration
# ---------------------------------------------------------------------------


class TestMainExportConfig:
    """Tests for 'main(["export-config", ...])' writing a JSON file."""

    def test_creates_valid_json_file(self, tmp_path: Path):
        from pcg_llm._cli import main
        output = tmp_path / "out.json"
        rc = main(["export-config", "--output", str(output)])
        assert rc == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, dict)
        assert "hidden_dim" in data

    def test_tiny_preset_output(self, tmp_path: Path):
        from pcg_llm._cli import main
        output = tmp_path / "tiny.json"
        rc = main(["export-config", "--preset", "tiny", "--output", str(output)])
        assert rc == 0
        data = json.loads(output.read_text())
        # tiny preset uses default architecture values
        assert data["hidden_dim"] == 512
        assert data["max_seq_len"] == 2048

    def test_returns_1_on_oserror(self, tmp_path: Path):
        from pcg_llm._cli import main
        # Point to a directory that doesn't exist and can't be created (invalid path)
        bad_output = str(tmp_path / "no_such_dir" / "subdir" / "out.json")
        # Patch Path.write_text to raise OSError
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            rc = main(["export-config", "--output", bad_output])
        assert rc == 1


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    """Tests for _resolve_config with various args namespaces."""

    def _make_empty_args(self, **overrides) -> argparse.Namespace:
        """Build a minimal Namespace mimicking the export-config subcommand defaults."""
        defaults = {
            "config": None,
            "preset": None,
            "hidden_dim": None,
            "max_seq_len": None,
            "block_size": None,
            "initial_sparsity": None,
            "max_solver_iters": None,
            "solver_tolerance": None,
            "optimizer": None,
            "optimizer_bits": None,
            "normalize": None,
            "projection": None,
            "base_lr": None,
            "warmup_steps": None,
            "total_tokens": None,
            "batch_size": None,
            "grad_accum_steps": None,
            "grad_checkpoint": None,
            "cpu_offload_mask": None,
            "checkpoint_interval": None,
            "checkpoint_backend": None,
            "checkpoint_dir": None,
            "wandb_project": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_overrides_returns_default_config(self):
        from pcg_llm._cli import _resolve_config
        from pcg_llm.config import TrainingConfig
        args = self._make_empty_args()
        cfg = _resolve_config(args)
        assert isinstance(cfg, TrainingConfig)
        # Verify it matches the dataclass defaults
        default = TrainingConfig()
        assert cfg.hidden_dim == default.hidden_dim
        assert cfg.max_seq_len == default.max_seq_len

    def test_preset_tiny_sets_values(self):
        from pcg_llm._cli import _resolve_config
        args = self._make_empty_args(preset="tiny")
        cfg = _resolve_config(args)
        assert cfg.hidden_dim == 512
        assert cfg.max_seq_len == 2048
        assert cfg.checkpoint_backend == "local"

    def test_json_config_file(self, tmp_path: Path):
        from pcg_llm._cli import _resolve_config
        # Write a partial config JSON
        cfg_file = tmp_path / "custom.json"
        cfg_file.write_text(
            json.dumps({"hidden_dim": 512, "max_seq_len": 2048, "block_size": 32}),
            encoding="utf-8",
        )
        args = self._make_empty_args(config=str(cfg_file))
        cfg = _resolve_config(args)
        assert cfg.hidden_dim == 512
        assert cfg.block_size == 32

    def test_cli_override_wins_over_preset(self):
        from pcg_llm._cli import _resolve_config
        # preset=tiny sets warmup_steps=500; override to 10
        args = self._make_empty_args(preset="tiny", warmup_steps=10)
        cfg = _resolve_config(args)
        assert cfg.warmup_steps == 10


# ---------------------------------------------------------------------------
# _cmd_evaluate
# ---------------------------------------------------------------------------


class TestCmdEvaluate:
    """Tests for the _cmd_evaluate subcommand handler."""

    def test_returns_1_when_checkpoint_missing(self, tmp_path: Path):
        from pcg_llm._cli import _cmd_evaluate
        args = argparse.Namespace(
            checkpoint=str(tmp_path / "nonexistent.pt"),
            tasks="arc_challenge,mmlu",
            quantize="4bit",
            output=None,
        )
        rc = _cmd_evaluate(args)
        assert rc == 1

    def test_stub_results_when_harness_unavailable(self, tmp_path: Path, capsys):
        """With a real .pt file and no lm-eval installed, produce stub JSON output."""
        import torch

        ckpt_path = tmp_path / "test.pt"
        torch.save({"config": {}, "step": 0}, str(ckpt_path))

        from pcg_llm._cli import _cmd_evaluate

        args = argparse.Namespace(
            checkpoint=str(ckpt_path),
            tasks="arc_challenge,mmlu",
            quantize="4bit",
            output=None,
        )

        # Patch BenchmarkHarness import to simulate ImportError (lm-eval not installed)
        with patch.dict("sys.modules", {"pcg_llm.evaluation.harness": None}):
            rc = _cmd_evaluate(args)

        # Should succeed (return 0) with stub output
        assert rc == 0
        captured = capsys.readouterr()
        result_data = json.loads(captured.out)
        assert "arc_challenge" in result_data
        assert result_data["arc_challenge"]["score"] is None

    def test_writes_results_to_output_file(self, tmp_path: Path):
        """Results should be written to file when --output is set."""
        import torch

        ckpt_path = tmp_path / "test.pt"
        torch.save({"config": {}, "step": 0}, str(ckpt_path))

        output_path = tmp_path / "results.json"

        from pcg_llm._cli import _cmd_evaluate

        args = argparse.Namespace(
            checkpoint=str(ckpt_path),
            tasks="arc_challenge",
            quantize="none",
            output=str(output_path),
        )

        with patch.dict("sys.modules", {"pcg_llm.evaluation.harness": None}):
            rc = _cmd_evaluate(args)

        assert rc == 0
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "arc_challenge" in data
