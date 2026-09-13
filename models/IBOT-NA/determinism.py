from __future__ import annotations

import os
import random


DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_process_environment() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    os.environ.setdefault("PYTHONHASHSEED", "0")


def deterministic_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", DEFAULT_CUBLAS_WORKSPACE_CONFIG)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def configure_reproducibility(torch_module, seed: int | None = None, strict: bool = True) -> None:
    configure_process_environment()
    if seed is not None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)

    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    if hasattr(torch_module.backends, "cuda") and hasattr(torch_module.backends.cuda, "matmul"):
        torch_module.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch_module.backends.cudnn, "allow_tf32"):
        torch_module.backends.cudnn.allow_tf32 = False
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("highest")
    if strict and hasattr(torch_module, "use_deterministic_algorithms"):
        torch_module.use_deterministic_algorithms(True)
