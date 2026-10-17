"""The notebook is generated code too: check every cell parses, and actually run the
analysis cells against the synthetic fixture data."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "01_explore_graphs.ipynb"
nbformat = pytest.importorskip("nbformat")


def _code_cells():
    return [c for c in nbformat.read(NB, as_version=4).cells if c.cell_type == "code"]


def _strip_magics(src: str) -> str:
    # IPython "!cmd" / "%magic" lines are not Python; replace them with `pass`, keeping indent.
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(line[: len(line) - len(stripped)] + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def test_notebook_is_up_to_date(tmp_path):
    """The committed notebook must equal what the build script produces."""
    before = NB.read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebook.py")], check=True)
    assert NB.read_text(encoding="utf-8") == before, "run `make notebook` and commit"


def test_all_code_cells_parse():
    cells = _code_cells()
    assert cells
    for c in cells:
        ast.parse(_strip_magics(c.source))
        assert c.metadata.get("tags") in (["setup"], ["analysis"])


def test_no_python_import_of_kaggle():
    # `import kaggle` deletes KAGGLE_API_TOKEN from the environment (kaggle-cli issue #882).
    for c in _code_cells():
        tree = ast.parse(_strip_magics(c.source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "kaggle" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "kaggle"


def test_analysis_cells_run_on_synthetic_data(data_dir, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setenv("PGQA_DATA_DIR", str(data_dir))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    ns: dict = {"__name__": "__notebook__"}
    for c in _code_cells():
        if c.metadata["tags"] == ["analysis"]:
            exec(compile(c.source, "<notebook-cell>", "exec"), ns)
    plt.close("all")
    assert ns["DATA_ROOT"] == data_dir
    assert len(ns["rows"]) == 1 and ns["rows"][0]["file_coverage"] == 1.0
    assert ns["items"] and ns["items"][0].template == "callers"
    assert any("Embedding model" in item for item, _ in ns["checklist"])
