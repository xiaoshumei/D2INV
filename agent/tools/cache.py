"""
Pipeline-results cache helper for D2INV Agent tools.

Generated artifacts are persisted to ./results/<dataset>/ with stable,
fixed filenames (data_story.json, infographic_template.html, inv.html,
evaluate.html). These helpers let each generation tool read existing on-disk
results so a re-run of the same dataset can serve the cached output instead of
calling the LLM again. Regeneration is achieved by deleting the dataset's
results folder first (see ClearResultsTool) and then re-running the tools.
"""

from __future__ import annotations

import json
import os
import shutil

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)


def dataset_dir(dataset_name: str) -> str:
    """Return the results directory for a dataset (extension stripped)."""
    stem = os.path.splitext(os.path.basename(dataset_name))[0]
    return os.path.join(RESULTS_DIR, stem)


def read_text(dataset_name: str, filename: str):
    """Return file contents as str, or None if the file does not exist."""
    path = os.path.join(dataset_dir(dataset_name), filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def read_json(dataset_name: str, filename: str):
    """Return file contents parsed as JSON, or None if the file does not exist."""
    text = read_text(dataset_name, filename)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception as e:
        print(f"[cache] failed to parse {filename}: {e}")
        return None


def clear_dataset(dataset_name: str) -> bool:
    """Delete a dataset's results directory. Returns True if anything was removed."""
    d = dataset_dir(dataset_name)
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False