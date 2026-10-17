from __future__ import annotations

import json

import pytest

from fixtures_data import COMMIT
from patchgraph_qa import io


def test_find_data_root_nested_tasks(data_dir, monkeypatch):
    monkeypatch.delenv("PGQA_DATA_DIR", raising=False)
    assert io.find_data_root([data_dir]) == data_dir
    assert io.find_tasks_file(data_dir) == data_dir / "published" / "tasks.jsonl"


def test_find_data_root_env_and_missing(tmp_path, data_dir, monkeypatch):
    monkeypatch.setenv("PGQA_DATA_DIR", str(data_dir))
    assert io.find_data_root([tmp_path / "nope"]) == data_dir
    monkeypatch.delenv("PGQA_DATA_DIR")
    assert io.find_data_root([tmp_path / "nope"]) is None


def test_load_tasks_flags_missing_fields(data_dir):
    tasks = io.load_tasks(io.find_tasks_file(data_dir))
    assert [t.instance_id for t in tasks] == ["pkg_1", "pkg_2"]
    assert tasks[0].missing == []
    assert "base_commit" in tasks[1].missing and tasks[1].base_commit == ""


def test_iter_jsonl_reports_bad_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        list(io.iter_jsonl(p))


def test_load_graph_accepts_edges_or_links(data_dir, graph_json, tmp_path):
    g = io.load_graph(data_dir / "graphs" / f"pkg_{COMMIT}.json")
    assert g.is_directed() and g.is_multigraph()
    assert g.number_of_nodes() == 8 and g.number_of_edges() == 6
    assert g.nodes["pkg.core.helper"]["text"].startswith("def helper")
    legacy = dict(graph_json)
    legacy["links"] = legacy.pop("edges")
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    assert io.load_graph(p).number_of_edges() == 6


def test_load_embeddings(data_dir):
    emb = io.load_embeddings(data_dir / "embeddings" / f"pkg_{COMMIT}.npz")
    assert "pkg.core.helper" in emb
    assert emb["pkg.core.helper"].shape == (256,)
    assert str(emb["pkg.core.helper"].dtype) == "float32"


def _task(**kw) -> io.Task:
    base = {"instance_id": "fastapi_15661", "repo": "FastAPI/FastAPI", "base_commit": "c0ffee"}
    base.update(kw)
    return io.Task(
        instance_id=base["instance_id"],
        repo=base["repo"],
        base_commit=base["base_commit"],
        problem_statement="",
        patch="",
        test_patch="",
        raw=base,
    )


def test_commit_key_and_stems():
    t = _task()
    assert io.repo_short("fastapi/fastapi/") == "fastapi"
    assert io.commit_key(t) == "fastapi_c0ffee"
    assert io.artifact_stems(t) == ["fastapi_15661", "fastapi_c0ffee"]
    assert io.commit_key(_task(base_commit="")) is None
    assert io.artifact_stems("only_this") == ["only_this"]


def test_paths_resolve_commit_named_files(data_dir):
    tasks = io.load_tasks(io.find_tasks_file(data_dir))
    gp = io.graph_path(data_dir / "graphs", tasks[0])
    ep = io.embedding_path(data_dir / "embeddings", tasks[0])
    assert gp.name == f"pkg_{COMMIT}.json" and gp.exists()  # repo "Owner/PKG" -> "pkg"
    assert ep.name == f"pkg_{COMMIT}.npz" and ep.exists()
    assert io.load_graph(gp).number_of_nodes() == 8
    # Task without a base_commit: falls back to the id name, which does not exist.
    assert not io.graph_path(data_dir / "graphs", tasks[1]).exists()


def test_paths_prefer_task_id_file_when_present(tmp_path):
    t = _task()
    (tmp_path / "fastapi_15661.json").write_text("{}")
    (tmp_path / "fastapi_c0ffee.json").write_text("{}")
    assert io.graph_path(tmp_path, t).name == "fastapi_15661.json"
    (tmp_path / "fastapi_15661.json").unlink()
    assert io.graph_path(tmp_path, t).name == "fastapi_c0ffee.json"
    (tmp_path / "fastapi_c0ffee.json").unlink()
    assert io.graph_path(tmp_path, t).name == "fastapi_c0ffee.json"  # best guess, not on disk


def test_download_names_dedupe_shared_commits():
    a, b = _task(instance_id="fastapi_1"), _task(instance_id="fastapi_2")  # same commit
    c = _task(instance_id="x_3", repo="o/X", base_commit="beef")
    d = _task(instance_id="bad", base_commit="")
    assert io.download_names([a, b, c, d]) == [
        "graphs/fastapi_c0ffee.json",
        "graphs/x_beef.json",
        "embeddings/fastapi_c0ffee.npz",
        "embeddings/x_beef.npz",
    ]
