"""Convert the percent-format sources in src/ into .ipynb notebooks for Kaggle and Colab.

    python training/notebooks/build_notebooks.py

Cells are separated by "# %%" (code) and "# %% [markdown]" (markdown, each line starting "# ").
Edit the .py sources, not the generated notebooks.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cells(source: str) -> list[dict]:
    out = []
    for block in source.split("# %%")[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines()]
            out.append({"cell_type": "markdown", "metadata": {}, "source": _lines(lines)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": _lines(body.splitlines())})
    return out


def _lines(lines: list[str]) -> list[str]:
    return [line + "\n" for line in lines[:-1]] + lines[-1:]


def main() -> None:
    for src in sorted((HERE / "src").glob("*.py")):
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "cells": cells(src.read_text(encoding="utf-8")),
        }
        target = HERE / f"{src.stem}.ipynb"
        target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {target.name} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
