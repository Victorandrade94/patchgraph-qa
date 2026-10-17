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


def _install_helpers(tmp_path: Path) -> dict:
    """Exec only the imports and function defs of the install cell (no Drive, no pip)."""
    cell = next(c for c in _code_cells() if "def find_zip" in c.source)
    tree = ast.parse(_strip_magics(cell.source))
    keep = [
        n
        for n in tree.body  # keep source order: defaults like ZIP_NAME precede the defs
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (isinstance(n, ast.Assign) and ast.unparse(n).startswith("ZIP_NAME"))
    ]
    ns: dict = {"Path": Path, "WORK": tmp_path / "work", "IN_KAGGLE": False}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<install-cell>", "exec"), ns)
    return ns


def _make_zip(path: Path) -> None:
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("patchgraph-qa/pyproject.toml", "[project]\nname='x'\n")
        zf.writestr("patchgraph-qa/src/patchgraph_qa/__init__.py", "__version__ = 'zip'\n")


def test_install_cell_prefers_drive_and_activates_src(tmp_path, monkeypatch):
    ns = _install_helpers(tmp_path)
    drive = tmp_path / "MyDrive"
    _make_zip(drive / "a" / "b" / ns["ZIP_NAME"])
    _make_zip(drive / "x" / "y" / "z" / "w" / ns["ZIP_NAME"])  # too deep: ignored
    (drive / ".hidden").mkdir()

    zp = ns["find_zip"](drive)
    assert zp == drive / "a" / "b" / ns["ZIP_NAME"]
    assert ns["find_zip"](drive, max_depth=1) is None
    assert ns["find_zip"](tmp_path / "missing") is None

    work = tmp_path / "work"
    work.mkdir()
    (work / "patchgraph-qa").mkdir()
    (work / "patchgraph-qa" / "stale.txt").write_text("old")
    repo = ns["extract_repo"](zp, work)
    assert repo == work / "patchgraph-qa" and not (repo / "stale.txt").exists()

    # The cell must put src/ on sys.path *after* pip, so the import works without a restart.
    cell = next(c for c in _code_cells() if "def find_zip" in c.source).source
    assert cell.index("!pip install -q -e") < cell.index("activate_src(repo)\n    import")
    assert cell.index("drive.mount") < cell.index("files.upload()")

    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m.startswith("patchgraph_qa")}
    try:
        ns["activate_src"](repo)
        assert sys.path[0] == str(repo / "src")
        import patchgraph_qa

        assert patchgraph_qa.__version__ == "zip"
    finally:
        for m in [m for m in sys.modules if m.startswith("patchgraph_qa")]:
            del sys.modules[m]
        sys.modules.update(saved)
