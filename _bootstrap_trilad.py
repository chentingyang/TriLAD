"""Bootstrap TriLAD / hints / utils modules when .py sources were removed but .pyc remain."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_BOOTSTRAPPED = False

# Before utils.data_pipeline; vision_encoder/detector load after utils.
_TRILAD_CORE_ORDER = (
    "feature_selection",
    "schemas",
    "parsing",
    "llm_api",
    "few_shot",
    "few_shot_pool_sources",
    "prompting",
)

_TRILAD_LATE_ORDER = (
    "vision_encoder",
    "detector",
)

# data_pipeline needs TriLAD.feature_selection/schemas; prompt_intervals needs few_shot.
_UTILS_ORDER = (
    "utils",
    "utils.data_pipeline",
    "utils.metrics",
    "utils.aggregation",
    "utils.range_postprocess",
    "utils.precise_hit_postprocess",
    "utils.prompt_intervals",
)

_HINTS_ORDER = (
    "hints",
    "hints.swat_tag_kb",
    "hints.swat_window_hint",
    "hints.msl_window_hint",
    "hints.smd_window_hint",
    "hints.psm_window_hint",
)


def _ensure_pkg_init(pkg_dir: Path, docstring: str) -> None:
    init_py = pkg_dir / "__init__.py"
    if not init_py.exists():
        pkg_dir.mkdir(parents=True, exist_ok=True)
        init_py.write_text(docstring, encoding="utf-8")


def _load_module(modname: str, py: Path, pyc: Path) -> None:
    if modname in sys.modules:
        return
    if py.exists():
        importlib.import_module(modname)
        return
    if not pyc.exists():
        return
    spec = importlib.util.spec_from_file_location(modname, pyc)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)


def bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    _ensure_pkg_init(ROOT / "TriLAD", "")
    _ensure_pkg_init(
        ROOT / "hints",
        '"""Dataset window hints and SWaT tag knowledge base."""\n',
    )
    _ensure_pkg_init(
        ROOT / "utils",
        '"""Data I/O, aggregation, metrics, and offline post helpers."""\n',
    )

    def load_trilad(stem: str) -> None:
        modname = f"TriLAD.{stem}"
        py = ROOT / "TriLAD" / f"{stem}.py"
        pyc = ROOT / "TriLAD" / "__pycache__" / f"{stem}.cpython-313.pyc"
        _load_module(modname, py, pyc)

    def load_pkg(stem: str, pkg: str) -> None:
        # stem is "pkg" or "pkg.xxx"
        parts = stem.split(".")
        if stem == pkg:
            py = ROOT / pkg / "__init__.py"
            pyc = ROOT / pkg / "__pycache__" / "__init__.cpython-313.pyc"
        else:
            name = parts[-1]
            py = ROOT / pkg / f"{name}.py"
            pyc = ROOT / pkg / "__pycache__" / f"{name}.cpython-313.pyc"
        _load_module(stem, py, pyc)

    for _ in range(4):
        before = len(sys.modules)
        for stem in _HINTS_ORDER:
            load_pkg(stem, "hints")
        for stem in _TRILAD_CORE_ORDER:
            load_trilad(stem)
        for stem in _UTILS_ORDER:
            load_pkg(stem, "utils")
        for stem in _TRILAD_LATE_ORDER:
            load_trilad(stem)
        if len(sys.modules) == before:
            break

    _BOOTSTRAPPED = True
