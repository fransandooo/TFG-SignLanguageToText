#!/usr/bin/env python3
"""
Patch 22a-22g LSTM notebooks to use RunPod filesystem paths.
Run this script on RunPod after uploading the notebooks:
    python patch_notebooks_for_runpod.py

RunPod paths assumed:
    /workspace/dataset/wlasl2000/       <- landmarks
    /workspace/dataset/subsets/         <- subset JSONs
    /workspace/experiments/             <- training outputs
"""

import json
import os
import re

NOTEBOOKS = [
    "22a_train_wlasl100_lstm.ipynb",
    "22b_train_wlasl300_lstm.ipynb",
    "22c_train_wlasl500_lstm.ipynb",
    "22d_train_wlasl700_lstm.ipynb",
    "22e_train_wlasl900_lstm.ipynb",
    "22f_train_wlasl1200_lstm.ipynb",
    "22g_train_wlasl1500_lstm.ipynb",
]

# Replacements: (pattern, replacement)
PATH_REPLACEMENTS = [
    # LANDMARKS_ROOT — any local path ending in wlasl2000 or wlasl/landmarks/wlasl2000
    (
        r'Path\("[^"]*wlasl2000"\)',
        'Path("/workspace/dataset/wlasl2000")',
    ),
    # SUBSET_PATH — any local path ending in subsets/wlasl_*_subset.json
    (
        r'"[^"]*subsets/(wlasl_\w+_subset\.json)"',
        r'"/workspace/dataset/subsets/\1"',
    ),
    # EXPERIMENTS_BASE — any local path ending in 05_experiments
    (
        r'Path\("[^"]*05_experiments"\)',
        'Path("/workspace/experiments")',
    ),
    # num_workers in DataLoader calls: 0 → 8 (parallel data loading)
    (
        r'num_workers=0',
        'num_workers=8',
    ),
    # num_workers=4 → 8 if already patched once
    (
        r'num_workers=4',
        'num_workers=8',
    ),
]

PIP_INSTALL_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install missing libraries on RunPod\n",
        "import subprocess\n",
        "subprocess.run(['pip', 'install', '-q', 'seaborn', 'scikit-learn'], check=True)\n",
        "print('Dependencies ready.')",
    ],
}


def patch_source(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        for pattern, replacement in PATH_REPLACEMENTS:
            line = re.sub(pattern, replacement, line)
        result.append(line)
    return result


def already_has_pip_cell(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "seaborn" in src and "scikit-learn" in src:
                return True
    return False


def patch_notebook(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Patch paths in all code cells
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["source"] = patch_source(cell["source"])

    # Prepend pip install cell if not already present
    if not already_has_pip_cell(nb):
        nb["cells"].insert(0, PIP_INSTALL_CELL)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  Patched: {path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Support running from repo root or from the notebooks dir
    candidates = [
        os.path.join(script_dir, "01_notebooks", "word_recognition"),
        script_dir,
    ]

    nb_dir = None
    for c in candidates:
        if os.path.exists(os.path.join(c, NOTEBOOKS[0])):
            nb_dir = c
            break

    if nb_dir is None:
        print(
            "ERROR: Could not find notebooks. "
            "Run this script from the repo root or the notebooks directory."
        )
        return

    print(f"Notebook directory: {nb_dir}")
    for nb_name in NOTEBOOKS:
        nb_path = os.path.join(nb_dir, nb_name)
        if not os.path.exists(nb_path):
            print(f"  SKIP (not found): {nb_name}")
            continue
        patch_notebook(nb_path)

    print("\nDone. All notebooks patched for RunPod.")
    print("Paths now use:")
    print("  LANDMARKS_ROOT  -> /workspace/dataset/wlasl2000")
    print("  SUBSET_PATH     -> /workspace/dataset/subsets/<name>.json")
    print("  EXPERIMENTS_BASE -> /workspace/experiments")


if __name__ == "__main__":
    main()
