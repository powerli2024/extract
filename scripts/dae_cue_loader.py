"""Resolve the Chinese DAE-TSE cue helper used by the challengecup chain.

The upstream ``DAE-TSE`` clone ships the English LibriMix cue converter, while
the Chinese checkpoint used by the challengecup lives with an external helper
(``tools/build_zh_text_cues.py``).  Keep that helper outside the upstream clone
and load it by an explicit path or a small, deterministic candidate list.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable


HELPER_RELATIVE = Path("tools") / "build_zh_text_cues.py"
DEFAULT_EXTERNAL_ROOTS = (
    Path("/root/autodl-fs/midea-dae/code/DAE-TSE"),
    Path("/root/autodl-tmp/projects/DAE-TSE"),
)


def cue_helper_candidates(dae_repo: Path, requested: Path | None = None) -> list[Path]:
    values: list[Path] = []
    if requested is not None:
        values.append(requested)
    env_value = os.environ.get("DAE_TSE_CUE_HELPER", "").strip()
    if env_value:
        values.append(Path(env_value))
    values.append(dae_repo / HELPER_RELATIVE)
    values.extend(root / HELPER_RELATIVE for root in DEFAULT_EXTERNAL_ROOTS)
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = value.expanduser().resolve()
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_cue_helper(dae_repo: Path, requested: Path | None = None) -> Path:
    candidates = cue_helper_candidates(dae_repo, requested)
    for path in candidates:
        if path.is_file():
            return path
    inspected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Chinese DAE cue helper not found. The official DAE-TSE clone does "
        "not ship build_zh_text_cues.py; provide --dae-cue-helper or set "
        f"DAE_TSE_CUE_HELPER. Inspected: {inspected}"
    )


def load_text_to_phone_labels(helper_path: Path) -> Callable[[str], tuple[list[list[int]], list[str]]]:
    """Load ``text_to_phone_labels`` from an external helper file."""

    path = helper_path.resolve()
    spec = importlib.util.spec_from_file_location("extract_sep_dae_zh_cues", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import DAE cue helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapper = getattr(module, "text_to_phone_labels", None)
    if not callable(mapper):
        raise AttributeError(
            f"DAE cue helper {path} must define text_to_phone_labels(text)"
        )
    return mapper
