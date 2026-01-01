from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib


@dataclass
class BundleMeta:
    project: str = "news_topics_classification"
    dataset: str = "Unspecified"
    text_columns: Tuple[str, ...] = ("headline", "short_description")
    label_column: str = "category"
    trained_at_utc: str = ""
    python: str = ""
    platform: str = ""
    versions: Dict[str, str] | None = None
    metrics: Dict[str, Any] | None = None
    params: Dict[str, Any] | None = None
    notes: str = ""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _collect_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for mod, name in [
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("optuna", "optuna"),
        ("joblib", "joblib"),
    ]:
        try:
            m = __import__(mod)
            versions[name] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    return versions


def save_bundle(
    out_dir: str | Path,
    *,
    preprocessor: Any,
    clf: Any,
    le: Any,
    le_aggregate: Any,
    aggregate_map: Dict[str, str],
    dataset: str = "Unspecified",
    text_columns: Tuple[str, ...] = ("headline", "short_description"),
    label_column: str = "category",
    best_params: Optional[Dict[str, Any]] = None,
    val_macro_f1: Optional[float] = None,
    test_macro_f1: Optional[float] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    compression: int = 3,
) -> tuple[Path, Path]:
    """
    Saves:
      - bundle.joblib : all objects required for inference
      - meta.json     : human-readable metadata

    Returns (bundle_path, meta_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "bundle.joblib"
    meta_path = out_dir / "meta.json"

    payload = {
        "preprocessor": preprocessor,
        "clf": clf,
        "le": le,
        "le_aggregate": le_aggregate,
        "aggregate_map": aggregate_map,
        "best_params": best_params,
        "val_macro_f1": val_macro_f1,
        "test_macro_f1": test_macro_f1,
    }
    joblib.dump(payload, bundle_path, compress=compression)

    meta = BundleMeta(
        dataset=dataset,
        text_columns=text_columns,
        label_column=label_column,
        trained_at_utc=_now_utc_iso(),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        versions=_collect_versions(),
        metrics={
            "val_macro_f1": val_macro_f1,
            "test_macro_f1": test_macro_f1,
        },
        params=best_params or {},
    )

    meta_dict = asdict(meta)
    if extra_meta:
        meta_dict.update(extra_meta)

    meta_path.write_text(
        json.dumps(meta_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return bundle_path, meta_path


def load_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    """
    Loads bundle.joblib and returns the payload dict.
    """
    bundle_dir = Path(bundle_dir)
    bundle_path = bundle_dir / "bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing {bundle_path}")

    payload = joblib.load(bundle_path)
    required = ["preprocessor", "clf", "le", "le_aggregate", "aggregate_map"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Corrupt bundle file; missing keys: {missing}")

    return payload