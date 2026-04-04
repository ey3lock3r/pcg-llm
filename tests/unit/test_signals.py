"""Tests for SignalHandler — covering SIGTERM handling, shutdown_requested property,
and check_and_save logic including the slow-save warning path."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch


class TestSignalHandlerInit:
    """Tests for construction and _register."""

    def test_import(self) -> None:
        from pcg_llm.checkpointing.signals import SignalHandler  # noqa: F401

    def test_shutdown_requested_is_false_initially(self) -> None:
        """A freshly constructed SignalHandler should report shutdown_requested=False."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        assert handler.shutdown_requested is False

    def test_sigterm_received_at_is_none_initially(self) -> None:
        """_sigterm_received_at should be None before any signal is received."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        assert handler._sigterm_received_at is None

    def test_register_silently_skips_on_os_error(self) -> None:
        """_register must not raise when signal.signal raises OSError."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        with patch("signal.signal", side_effect=OSError("not supported")):
            # Should not raise
            handler = SignalHandler(checkpoint_manager=manager)
        assert handler.shutdown_requested is False

    def test_register_silently_skips_on_value_error(self) -> None:
        """_register must not raise when signal.signal raises ValueError."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        with patch("signal.signal", side_effect=ValueError("main thread only")):
            handler = SignalHandler(checkpoint_manager=manager)
        assert handler.shutdown_requested is False


class TestSignalHandlerSigtermHandling:
    """Tests for _handle_sigterm (lines 46-48)."""

    def test_handle_sigterm_sets_shutdown_requested_true(self) -> None:
        """Calling _handle_sigterm directly must set _shutdown_requested=True."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        handler._handle_sigterm(15, None)
        assert handler._shutdown_requested is True

    def test_handle_sigterm_sets_sigterm_received_at(self) -> None:
        """_handle_sigterm must record a timestamp in _sigterm_received_at."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)

        fake_time = 123.456
        with patch("time.monotonic", return_value=fake_time):
            handler._handle_sigterm(15, None)

        assert handler._sigterm_received_at == fake_time

    def test_shutdown_requested_property_reflects_internal_flag(self) -> None:
        """shutdown_requested property must mirror _shutdown_requested attribute."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        assert handler.shutdown_requested is False
        handler._handle_sigterm(15, None)
        assert handler.shutdown_requested is True

    def test_handle_sigterm_logs_warning(self, caplog) -> None:
        """_handle_sigterm should log a warning message."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)

        with caplog.at_level(logging.WARNING, logger="pcg_llm.checkpointing.signals"):
            handler._handle_sigterm(15, None)

        assert any("SIGTERM" in record.message for record in caplog.records)


class TestSignalHandlerCheckAndSave:
    """Tests for check_and_save (lines 64-87)."""

    def test_check_and_save_returns_false_when_not_shutdown(self) -> None:
        """check_and_save must return False when shutdown has not been requested."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        result = handler.check_and_save({"model": "state"}, step=10)
        assert result is False

    def test_check_and_save_does_not_call_save_when_not_shutdown(self) -> None:
        """manager.save should not be called when shutdown is not requested."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        handler.check_and_save({"model": "state"}, step=10)
        manager.save.assert_not_called()

    def test_check_and_save_returns_true_when_shutdown_requested(self) -> None:
        """check_and_save must return True after SIGTERM has been received."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        handler._handle_sigterm(15, None)

        with patch("time.monotonic", return_value=0.0):
            result = handler.check_and_save({"model": "state"}, step=5)

        assert result is True

    def test_check_and_save_calls_manager_save_when_shutdown_requested(self) -> None:
        """manager.save must be called with the checkpoint dict and step."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        handler._handle_sigterm(15, None)

        checkpoint = {"weights": [1, 2, 3]}
        with patch("time.monotonic", return_value=0.0):
            handler.check_and_save(checkpoint, step=42)

        manager.save.assert_called_once_with(checkpoint, 42)

    def test_check_and_save_logs_warning_when_save_exceeds_threshold(self, caplog) -> None:
        """A warning must be logged when the save duration exceeds GCP_CHECKPOINT_WARN_SECONDS (20s)."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        handler._handle_sigterm(15, None)

        # Simulate t_start=0.0, then elapsed check returns 21.0 (> 20s threshold).
        # time.monotonic is called three times inside check_and_save:
        #   1. t_start = time.monotonic()        → 0.0
        #   2. elapsed = time.monotonic() - 0.0  → 21.0  (21.0 - 0.0 = 21.0)
        #   3. total_since_sigterm check          → needs _sigterm_received_at set
        # We only need to control the first two calls for the elapsed-time warning.
        monotonic_values = iter([0.0, 21.0, 21.0])

        with caplog.at_level(logging.WARNING, logger="pcg_llm.checkpointing.signals"):
            with patch("time.monotonic", side_effect=monotonic_values):
                handler.check_and_save({"model": "state"}, step=1)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "21" in msg or "exceeded" in msg.lower() or "soft deadline" in msg.lower()
            for msg in warning_messages
        ), f"Expected a slow-save warning in: {warning_messages}"

    def test_check_and_save_no_warning_for_fast_save(self, caplog) -> None:
        """No warning should be logged when the save completes quickly (< 20s)."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)
        handler._handle_sigterm(15, None)

        # Simulate a 1-second save — well under the 20s threshold
        monotonic_values = iter([0.0, 1.0, 1.0])

        with caplog.at_level(logging.WARNING, logger="pcg_llm.checkpointing.signals"):
            with patch("time.monotonic", side_effect=monotonic_values):
                handler.check_and_save({"model": "state"}, step=1)

        # No "exceeded" warning should appear (deadline-related messages only)
        deadline_warnings = [
            r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING and "soft deadline" in r.message.lower()
        ]
        assert (
            len(deadline_warnings) == 0
        ), f"Unexpected slow-save warning for a fast (1s) checkpoint: {deadline_warnings}"

    def test_check_and_save_logs_error_when_total_time_exceeds_deadline(self, caplog) -> None:
        """An error must be logged when total time since SIGTERM exceeds 25s."""
        from pcg_llm.checkpointing.signals import SignalHandler

        manager = MagicMock()
        handler = SignalHandler(checkpoint_manager=manager)

        # Set _sigterm_received_at to simulate signal received 30s ago
        handler._shutdown_requested = True
        handler._sigterm_received_at = 0.0

        # time.monotonic calls: t_start, elapsed check, total_since_sigterm check
        # t_start=100.0, after save=101.0, total_since_sigterm = 101.0 - 0.0 = 101.0 > 25s
        monotonic_values = iter([100.0, 101.0, 101.0])

        with caplog.at_level(logging.ERROR, logger="pcg_llm.checkpointing.signals"):
            with patch("time.monotonic", side_effect=monotonic_values):
                handler.check_and_save({"model": "state"}, step=1)

        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            "SIGTERM" in msg or "reclaim" in msg.lower() for msg in error_messages
        ), f"Expected a GCP-deadline error in: {error_messages}"
