"""The notebook is generated code too: check every cell parses, and actually run the
analysis cells against the synthetic fixture data."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures_data import COMMIT

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
    assert ns["anchor_source"] == "patch"
    assert ns["oracle_scores"]["exact"] == 1.0
    assert any("Embedding model" in item for item, _ in ns["checklist"])


def _install_helpers(tmp_path: Path, marker: str = "def find_zip") -> dict:
    """Exec only the imports and function defs of one setup cell (no Drive, no pip, no CLI)."""
    cell = next(c for c in _code_cells() if marker in c.source)
    tree = ast.parse(_strip_magics(cell.source))
    keep = [
        n
        for n in tree.body  # keep source order: defaults like ZIP_NAME precede the defs
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        or (  # UPPER_CASE constants such as ZIP_NAME, BACKOFF_S, MOUNT_HELP
            isinstance(n, ast.Assign)
            and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)
            and "IN_" not in ast.unparse(n.value)  # skip values that depend on the runtime
        )
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
    _make_zip(drive / ns["ZIP_NAME"])  # older copy at top level, name sorts first
    _make_zip(drive / "x" / "y" / "z" / "w" / ns["ZIP_NAME"])  # too deep: ignored
    (drive / ".hidden").mkdir()
    os.utime(drive / ns["ZIP_NAME"], (1_000, 1_000))
    os.utime(drive / "a" / "b" / ns["ZIP_NAME"], (2_000, 2_000))

    zp = ns["find_zip"](drive)
    assert zp == drive / "a" / "b" / ns["ZIP_NAME"]  # newest wins, not first/sorted name
    # Drive's duplicate naming also matches, and a newer duplicate wins.
    _make_zip(drive / "patchgraph-qa-day1 (1).zip")
    os.utime(drive / "patchgraph-qa-day1 (1).zip", (3_000, 3_000))
    _make_zip(drive / "other.zip")
    os.utime(drive / "other.zip", (9_000, 9_000))
    assert ns["find_zip"](drive) == drive / "patchgraph-qa-day1 (1).zip"
    (drive / "patchgraph-qa-day1 (1).zip").unlink()
    (drive / ns["ZIP_NAME"]).unlink()
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


def test_download_cell_builds_commit_names_and_unzips(data_dir, tmp_path):
    from patchgraph_qa import io

    ns = _install_helpers(tmp_path, marker="def names_from_tasks")
    tasks_file = data_dir / "published" / "tasks.jsonl"
    names = ns["names_from_tasks"](tasks_file)
    # Same names as the package helper, and commit-named (not task-id-named).
    assert names == io.download_names(io.load_tasks(tasks_file))
    assert names == [f"graphs/pkg_{COMMIT}.json", f"embeddings/pkg_{COMMIT}.npz"]

    # Simulate the CLI serving a file wrapped in a zip (with a folder inside the archive).
    import zipfile

    def fake_cli(*args):
        name, dest = args[args.index("-f") + 1], Path(args[args.index("-p") + 1])
        with zipfile.ZipFile(dest / (name.split("/")[-1] + ".zip"), "w") as zf:
            zf.writestr(name, "{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    ns.update(kaggle_cli=fake_cli, SLUG="slug", SLEEP=lambda s: None)
    out = ns["download_one"](names[0], tmp_path / "dl" / "graphs")
    assert out == tmp_path / "dl" / "graphs" / f"pkg_{COMMIT}.json" and out.read_text() == "{}"
    assert not list((tmp_path / "dl" / "graphs").glob("*.zip"))
    # Already on disk: skipped without calling the CLI.
    ns["kaggle_cli"] = lambda *a: pytest.fail("should not download again")
    assert ns["download_one"](names[0], tmp_path / "dl" / "graphs") == out


def test_download_flattens_nested_cli_layout(tmp_path):
    """New kaggle CLI saves -f graphs/X.json -p data/graphs into data/graphs/graphs/X.json."""
    ns = _install_helpers(tmp_path, marker="def names_from_tasks")
    dest = tmp_path / "dl" / "graphs"
    dest.mkdir(parents=True)
    # Pre-existing nest from a previous partial run.
    nested = dest / "graphs"
    nested.mkdir()
    (nested / "pre.json").write_text('{"pre":1}')
    ns["flatten_download_dir"](dest)
    assert (dest / "pre.json").read_text() == '{"pre":1}'
    assert not nested.exists()

    def fake_cli(*args):
        name, out = args[args.index("-f") + 1], Path(args[args.index("-p") + 1])
        base = name.split("/")[-1]
        # Mimic new CLI: create out/<first-segment>/<base>
        nest = out / name.split("/")[0]
        nest.mkdir(parents=True, exist_ok=True)
        (nest / base).write_text("{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    ns.update(kaggle_cli=fake_cli, SLUG="slug", SLEEP=lambda s: None)
    out = ns["download_one"]("graphs/pkg_abc.json", dest)
    assert out == dest / "pkg_abc.json" and out.read_text() == "{}"
    assert not (dest / "graphs").exists()
    # Resume: download_all flattens before checking.
    (dest / "embeddings").mkdir()  # unrelated
    nest2 = dest / "graphs"
    nest2.mkdir()
    (nest2 / "other.json").write_text("x")
    ns["flatten_download_dir"](dest)
    assert (dest / "other.json").read_text() == "x"


def test_notebook_does_not_rely_on_listing_flags():
    src = "\n".join(c.source for c in _code_cells())
    assert "--page-size" not in src and "--csv" not in src
    assert "kaggle>=2.2" in src


def test_kaggle_input_dir_alone_does_not_mean_kaggle(tmp_path, monkeypatch):
    """Regression (Oct 17 2026): Colab has a /kaggle/input directory too."""
    ns = _install_helpers(tmp_path, marker="def detect_env")
    detect = ns["detect_env"]
    writable = str(tmp_path)  # stands in for a writable /kaggle/working

    # Pretend every path exists, including /kaggle/input: must not matter.
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    colab_env = {"COLAB_RELEASE_TAG": "release-colab_20261001"}
    assert detect(colab_env, {}, writable) == (True, False)
    assert detect({}, {"google.colab": object()}, writable) == (True, False)
    assert detect({}, {}, writable) == (False, False)  # plain local Jupyter

    # Real Kaggle kernel: env var set and /kaggle/working writable. Never both flags at once.
    kaggle_env = {"KAGGLE_KERNEL_RUN_TYPE": "Interactive", **colab_env}
    assert detect(kaggle_env, {"google.colab": object()}, writable) == (False, True)
    # Env var without a writable working dir is not trusted.
    only_var = {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}
    assert detect(only_var, {}, str(tmp_path / "missing")) == (False, False)

    cell = next(c for c in _code_cells() if "def detect_env" in c.source).source
    assert 'exists("/kaggle/input")' not in cell and "IN_COLAB, IN_KAGGLE = detect_env()" in cell


def test_download_retries_429_with_backoff_then_reports_missing(tmp_path, capsys):
    ns = _install_helpers(tmp_path, marker="def names_from_tasks")
    waits: list[float] = []
    calls: dict[str, int] = {}

    def fake_cli(*args):
        name, dest = args[args.index("-f") + 1], Path(args[args.index("-p") + 1])
        calls[name] = calls.get(name, 0) + 1
        if name.endswith("ok.npz") and calls[name] >= 3:  # succeeds on the 3rd try
            (dest / "ok.npz").write_text("x")
            return subprocess.CompletedProcess(args, 0, "", "")
        if name.endswith("gone.npz"):  # not a rate limit: no retries
            return subprocess.CompletedProcess(args, 1, "", "404 Client Error: Not Found")
        return subprocess.CompletedProcess(args, 1, "", "429 Client Error: Too Many Requests")

    ns.update(kaggle_cli=fake_cli, SLUG="slug", SLEEP=waits.append)
    names = ["embeddings/ok.npz", "embeddings/limited.npz", "embeddings/gone.npz"]
    missing = ns["download_all"](names, tmp_path / "d")

    assert missing == ["embeddings/limited.npz", "embeddings/gone.npz"]
    assert calls == {"embeddings/ok.npz": 3, "embeddings/limited.npz": 6, "embeddings/gone.npz": 1}
    # ok: 5, 10, then the 1 s pause; limited: the full ladder; gone: nothing.
    assert waits == [5, 10, ns["PAUSE_S"], 5, 10, 20, 40, 60]
    assert "429 rate limit" in capsys.readouterr().out

    # Re-run: files on disk are skipped without any request or pause.
    waits.clear()
    ns["kaggle_cli"] = lambda *a: pytest.fail("should not request an existing file")
    assert ns["download_one"]("embeddings/ok.npz", tmp_path / "d" / "embeddings") is not None
    assert waits == []

    cell = next(c for c in _code_cells() if "def download_all" in c.source).source
    assert "Re-run THIS cell" in cell


def test_drive_mount_skip_failure_and_no_auto_upload(tmp_path, monkeypatch, capsys):
    ns = _install_helpers(tmp_path)
    mounted = tmp_path / "drive"
    (mounted / "MyDrive").mkdir(parents=True)
    assert ns["mount_drive"](mounted) is True  # already mounted: google.colab never touched
    assert "already mounted" in capsys.readouterr().out

    # Mount raising (like Colab's "ValueError: mount failed") prints help and returns False.
    import types

    calls = []

    def bad_mount(path, force_remount=False):
        calls.append((path, force_remount))
        raise ValueError("mount failed")

    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=bad_mount)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", sys.modules.get("google", google))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert ns["mount_drive"](tmp_path / "nodrive") is False
    assert calls == [(str(tmp_path / "nodrive"), True)]
    out = capsys.readouterr().out
    assert "mount failed" in out and "Select all" in out and "USE_UPLOAD = True" in out

    cell = next(c for c in _code_cells() if "def find_zip" in c.source).source
    assert "USE_UPLOAD = False" in cell
    assert "if repo is None and USE_UPLOAD:" in cell
    assert cell.index("if repo is None and USE_UPLOAD:") < cell.index("files.upload()")
