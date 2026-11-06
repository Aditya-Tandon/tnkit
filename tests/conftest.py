"""Session-level preflight: fail ONCE with a diagnosis instead of thirty cryptic errors.

The torch<->numpy bridge breaks whole-environment, not per-test: a torch wheel built
against the numpy 1.x ABI (every torch < 2.3) imported next to numpy >= 2 raises
"RuntimeError: Numpy is not available" from EVERY tensor.numpy() call. Seen live on the
CI 3.9 leg (torch 2.0.* + numpy 2.0.2): 18 failures + 15 errors, none of which named
the actual problem. This hook turns that class of environment skew into one line.
"""
import pytest


def pytest_configure(config):
    try:
        import torch
    except ImportError:
        return  # torch-free environment is supported; torch tests importorskip
    try:
        import numpy as np
        torch.zeros(2).numpy()
    except RuntimeError as e:
        raise pytest.UsageError(
            "torch<->numpy bridge is broken in this environment: "
            f"torch {torch.__version__} next to numpy {np.__version__} "
            f"({e}). torch < 2.3 is built against the numpy 1.x ABI -- "
            "install 'numpy<2' for this torch, or torch >= 2.3 for this numpy. "
            "Every torch-backend test would fail with the raw error above; "
            "failing the session once here instead."
        )
