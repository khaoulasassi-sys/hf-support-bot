"""Chargement de la configuration (fichiers YAML dans configs/)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(profile: str = "hf", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Charge `configs/<profile>.yaml` (ou un chemin) puis applique :
    1. les seuils calibrés `results/<profile>/calibration.json` s'ils existent ;
    2. les `overrides` explicites (dict imbriqué).
    """
    path = Path(profile)
    if not path.suffix:
        path = ROOT / "configs" / f"{profile}.yaml"
    cfg = yaml.safe_load(path.read_text())
    name = cfg.get("profile", path.stem)
    cfg["profile"] = name
    calib = ROOT / "results" / name / "calibration.json"
    if calib.exists():
        cfg["confidence"].update(json.loads(calib.read_text())["confidence"])
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Chemin absolu d'une entrée de cfg['paths'] ; le dossier d'index dépend du profil."""
    p = Path(cfg["paths"][key])
    p = p if p.is_absolute() else ROOT / p
    if key == "index_dir":
        p = p / cfg["profile"]
    return p
