from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


IMG_PATTERN = re.compile(r"^(?P<prefix>.+?)_(?P<pid>\d+)_0000\.nii\.gz$")


def detect_existing_dir(path: str) -> str:
    """Resolve and validate an existing directory path."""
    if not path or not str(path).strip():
        raise ValueError("Directory path is empty.")

    resolved = Path(os.path.expandvars(os.path.expanduser(str(path).strip()))).resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"Directory not found: {resolved}")
    return str(resolved)


def parse_patient_id_from_filename(filename: str) -> str:
    """Extract patient id from image filename like topcow_ct_001_0000.nii.gz."""
    name = Path(filename).name
    match = IMG_PATTERN.match(name)
    if not match:
        raise ValueError(f"Invalid image filename format: {name}")
    return match.group("pid")


def _infer_label_dir(image_dir: str) -> str:
    p = Path(image_dir)
    name = p.name
    if name.startswith("imagesTr"):
        candidate = p.with_name(name.replace("imagesTr", "labelsTr", 1))
        if candidate.exists() and candidate.is_dir():
            return str(candidate.resolve())

    fallback = p.parent / "labelsTr_topbrain_ct"
    if fallback.exists() and fallback.is_dir():
        return str(fallback.resolve())

    raise FileNotFoundError(
        "Label directory could not be inferred from image directory. "
        "Provide --label-dir or TOPBRAIN_LABEL_DIR explicitly."
    )


def list_patient_files(image_dir: str, label_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """Return valid image/label file pairs for TopBrain dataset.

    Output format:
        [{"patient_id": "001", "img_path": "...", "lbl_path": "..."}, ...]
    """
    image_dir_abs = detect_existing_dir(image_dir)
    label_dir_abs = detect_existing_dir(label_dir) if label_dir else _infer_label_dir(image_dir_abs)

    img_root = Path(image_dir_abs)
    lbl_root = Path(label_dir_abs)

    pairs: List[Dict[str, str]] = []

    for img_path in sorted(img_root.glob("*.nii.gz")):
        match = IMG_PATTERN.match(img_path.name)
        if not match:
            continue

        pid = match.group("pid")
        prefix = match.group("prefix")
        lbl_name = f"{prefix}_{pid}.nii.gz"
        lbl_path = lbl_root / lbl_name

        if not lbl_path.exists():
            continue

        pairs.append(
            {
                "patient_id": pid,
                "img_path": str(img_path.resolve()),
                "lbl_path": str(lbl_path.resolve()),
            }
        )

    pairs.sort(key=lambda x: int(re.findall(r"\d+", x["patient_id"])[-1]))
    return pairs
