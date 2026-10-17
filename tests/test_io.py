from __future__ import annotations

import json

import pytest

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
    g = io.load_graph(data_dir / "graphs" / "pkg_1.json")
    assert g.is_directed() and g.is_multigraph()
    assert g.number_of_nodes() == 8 and g.number_of_edges() == 6
    assert g.nodes["pkg.core.helper"]["text"].startswith("def helper")
    legacy = dict(graph_json)
    legacy["links"] = legacy.pop("edges")
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    assert io.load_graph(p).number_of_edges() == 6


def test_load_embeddings(data_dir):
    emb = io.load_embeddings(data_dir / "embeddings" / "pkg_1.npz")
    assert "pkg.core.helper" in emb
    assert emb["pkg.core.helper"].shape == (256,)
    assert str(emb["pkg.core.helper"].dtype) == "float32"
