"""
Convert all .py scripts with # %% markers to .ipynb notebooks.
Run from the project root:  python convert_to_notebooks.py
"""

import re
import json
import pathlib

# --- Modules whose .py files we want to convert ---
MODULES = [
    "module_01_fundamentals",
    "module_02_python_ml",
    "module_02_python_ml/mini_project_eda",
    "module_03_deep_learning",
    "module_04_computer_vision",
    "module_04_computer_vision/project_classifier/data",
    "module_04_computer_vision/project_classifier/models",
    "module_04_computer_vision/project_classifier/training",
    "module_04_computer_vision/project_classifier/evaluation",
    "module_05_generative_models",
    "module_06_conditional_generation",
    "module_06_conditional_generation/training",
    "module_06_conditional_generation/evaluation",
]

ROOT = pathlib.Path(__file__).parent


def parse_py_to_cells(source: str) -> list[dict]:
    """
    Split a .py script on # %% markers into notebook cells.

    Marker patterns:
      # %%              → code cell
      # %% [markdown]   → markdown cell (content is a comment block)
    """
    # Split using the # %% marker; keep the marker text as the first element
    parts = re.split(r"^(# %%.*?)$", source, flags=re.MULTILINE)

    cells: list[dict] = []

    # parts[0] is everything before the first marker (shebang / module docstring)
    header = parts[0].strip()
    if header:
        # Put the header as the first code cell
        cells.append(make_cell("code", header))

    # Iterate in pairs: (marker_line, cell_body)
    for i in range(1, len(parts) - 1, 2):
        marker: str = parts[i]
        body: str = parts[i + 1] if i + 1 < len(parts) else ""

        is_markdown = "[markdown]" in marker

        # Strip the leading newline that the split leaves behind
        body = body.lstrip("\n")

        if is_markdown:
            # Remove leading "# " from each comment line
            md_lines = []
            for line in body.splitlines():
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.startswith("#"):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            cell_source = "\n".join(md_lines).strip()
            cell_type = "markdown"
        else:
            # Include any inline marker title as a comment
            title = marker.replace("# %%", "").strip()
            cell_source = (f"# {title}\n" if title else "") + body.rstrip()
            cell_type = "code"

        if cell_source:
            cells.append(make_cell(cell_type, cell_source))

    return cells


def make_cell(cell_type: str, source: str) -> dict:
    """Return a minimal nbformat v4 cell dict."""
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def build_notebook(cells: list[dict]) -> dict:
    """Wrap cells in a minimal nbformat v4 notebook structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0",
            },
        },
        "cells": cells,
    }


def convert_file(py_path: pathlib.Path) -> None:
    source = py_path.read_text(encoding="utf-8")
    cells = parse_py_to_cells(source)

    if not cells:
        print(f"  --  No cells found - skipping {py_path.name}")
        return

    nb = build_notebook(cells)
    nb_path = py_path.with_suffix(".ipynb")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  OK  {py_path.relative_to(ROOT)}  ->  {nb_path.name}")


def main():
    print("=" * 60)
    print("  Converting .py scripts -> .ipynb notebooks")
    print("=" * 60)

    converted = 0
    for module_rel in MODULES:
        module_path = ROOT / module_rel
        py_files = sorted(module_path.glob("*.py"))

        if not py_files:
            continue

        print(f"\n[{module_rel}]")
        for py_file in py_files:
            # Skip utility-only files that have no %% markers
            source = py_file.read_text(encoding="utf-8")
            if "# %%" not in source:
                print(f"  --  {py_file.name}  (no cell markers - kept as .py only)")
                continue
            convert_file(py_file)
            converted += 1

    print(f"\n{'='*60}")
    print(f"  Done! {converted} notebooks created.")
    print("="*60)


if __name__ == "__main__":
    main()
