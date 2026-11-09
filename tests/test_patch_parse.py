from __future__ import annotations

from fixtures_data import CORE_SRC, GOLD_PATCH
from patchgraph_qa import patch_parse as pp
from patchgraph_qa.io import graph_from_node_link


def test_parse_files_and_line_numbers():
    files = pp.parse_unified_diff(GOLD_PATCH)
    assert [f.path for f in files] == ["pkg/core.py", "tests/test_core.py", "pkg/new_module.py"]
    core = files[0]
    assert core.is_python and not core.is_test
    h1, h2 = core.hunks
    assert h1.removed_old_lines == [5] and h1.added_new_lines == [5]
    assert h1.header == "import os"
    assert h2.removed_old_lines == [] and h2.added_new_lines == [10]
    assert h2.insertion_old_lines == [9]  # inserted after old line 9 (def run)
    assert core.hunks[0].touched_old_lines() == [5]
    assert files[1].is_test
    assert files[2].old_path is None and files[2].new_path == "pkg/new_module.py"


def test_no_newline_marker_and_default_lengths():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -3 +3 @@\n-a = 1\n+a = 2\n\\ No newline at end of file\n"
    (fc,) = pp.parse_unified_diff(diff)
    (h,) = fc.hunks
    assert (h.old_start, h.old_len, h.new_start, h.new_len) == (3, 1, 3, 1)
    assert h.removed_old_lines == [3]


def test_module_from_path():
    assert pp.module_from_path("src/fastapi/routing.py") == "fastapi.routing"
    assert pp.module_from_path("pkg/__init__.py") == "pkg"
    assert pp.is_test_path("tests/test_core.py") and pp.is_test_path("pkg/conftest.py")
    assert not pp.is_test_path("pkg/core.py")


def test_enclosing_symbols_from_source():
    assert pp.enclosing_symbols_from_source(CORE_SRC, "pkg.core", [5]) == ["pkg.core.helper"]
    assert pp.enclosing_symbols_from_source(CORE_SRC, "pkg.core", [9, 10]) == [
        "pkg.core.Engine.run"
    ]
    assert pp.enclosing_symbols_from_source(CORE_SRC, "pkg.core", [1]) == ["pkg.core"]


def test_map_with_source_lookup_is_exact(graph_json):
    g = graph_from_node_link(graph_json)
    changes = pp.parse_unified_diff(GOLD_PATCH)
    src = {"pkg/core.py": CORE_SRC}
    matches = pp.map_changes_to_symbols(g, changes, source_lookup=src.get)
    by_id = {m.node_id: m.method for m in matches}
    assert by_id["pkg.core.helper"] == "source"
    assert by_id["pkg.core.Engine.run"] == "source"
    # tests and brand-new files are excluded by default
    assert not any(m.file.startswith("tests/") for m in matches)
    assert not any(m.file == "pkg/new_module.py" for m in matches)


def test_map_without_source_uses_weaker_methods(graph_json):
    g = graph_from_node_link(graph_json)
    matches = pp.map_changes_to_symbols(g, pp.parse_unified_diff(GOLD_PATCH))
    by_id = {m.node_id: m.method for m in matches}
    # "def helper" / "def run" appear in context lines -> name heuristic
    assert by_id.get("pkg.core.helper") == "name"
    assert by_id.get("pkg.core.Engine.run") == "name"
    assert {m.method for m in matches} <= {"name", "text"}


def test_span_method_used_only_when_attributes_exist(graph_json):
    for n in graph_json["nodes"]:
        if n["id"] == "pkg.core.helper":
            n.update({"file": "pkg/core.py", "start_line": 4, "end_line": 5})
    g = graph_from_node_link(graph_json)
    matches = pp.map_changes_to_symbols(g, pp.parse_unified_diff(GOLD_PATCH))
    assert {m.node_id: m.method for m in matches}["pkg.core.helper"] == "span"


def test_coverage_report(graph_json):
    g = graph_from_node_link(graph_json)
    rep = pp.coverage_for_patch(g, GOLD_PATCH)
    assert rep.files_considered == 1 and rep.files_matched == 1
    assert rep.file_coverage == 1.0 and rep.unmatched_files == []
    empty = pp.coverage_for_patch(g, "")
    assert empty.file_coverage is None  # undefined, not forced to 0 or 1


def test_module_matches_node_allows_omitted_leading_packages():
    assert pp.module_matches_node("docs_src.pkg.routing", "pkg.routing.APIRouter")
    assert pp.module_matches_node("pkg.routing", "pkg.routing.APIRouter")
    assert pp.module_matches_node("fastapi.applications", "fastapi.applications.FastAPI")
    assert not pp.module_matches_node("pkg.routing", "other.mod.Foo")


def test_real_shaped_graph_matches_docs_src_patch():
    """Regression: FastAPI-like ids omit docs_src; only 'calls' edges; many test nodes."""
    nodes = [
        {
            "id": "pkg.routing.APIRouter",
            "name": "pkg.routing.APIRouter",
            "text": "class APIRouter:\n    def include_router(self):\n        pass\n",
        },
        {
            "id": "pkg.routing.APIRouter.include_router",
            "name": "pkg.routing.APIRouter.include_router",
            "text": "def include_router(self):\n    return None\n",
        },
        {
            "id": "tests.test_routing.test_include",
            "name": "tests.test_routing.test_include",
            "text": "def test_include():\n    assert True\n",
        },
    ]
    for i in range(30):
        nodes.append(
            {
                "id": f"tests.test_extra.test_{i}",
                "name": f"tests.test_extra.test_{i}",
                "text": f"def test_{i}():\n    pass\n",
            }
        )
    edges = [
        {
            "source": "pkg.routing.APIRouter.include_router",
            "target": "pkg.routing.APIRouter",
            "type": "calls",
            "key": 0,
        },
        {
            "source": "tests.test_routing.test_include",
            "target": "pkg.routing.APIRouter.include_router",
            "type": "calls",
            "key": 0,
        },
    ]
    g = graph_from_node_link(
        {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}
    )
    patch = (
        "\n".join(
            [
                "diff --git a/docs_src/pkg/routing.py b/docs_src/pkg/routing.py",
                "--- a/docs_src/pkg/routing.py",
                "+++ b/docs_src/pkg/routing.py",
                "@@ -1,4 +1,5 @@ class APIRouter:",
                " class APIRouter:",
                "     def include_router(self):",
                "-        return None",
                "+        return self",
            ]
        )
        + "\n"
    )
    matches = pp.map_changes_to_symbols(g, pp.parse_unified_diff(patch))
    ids = {m.node_id for m in matches}
    assert "pkg.routing.APIRouter" in ids
    assert "pkg.routing.APIRouter.include_router" in ids
    assert not any(m.node_id.startswith("tests.") for m in matches)
    anchors, source = pp.select_anchors(g, patch)
    assert source == "patch"
    assert "pkg.routing.APIRouter.include_router" in anchors


def test_select_anchors_falls_back_to_structural_when_patch_empty():
    g = graph_from_node_link(
        {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [
                {"id": "pkg.a", "name": "pkg.a", "text": "def a():\n    b()\n"},
                {"id": "pkg.b", "name": "pkg.b", "text": "def b():\n    pass\n"},
                {"id": "tests.t", "name": "tests.t", "text": "def t():\n    a()\n"},
            ],
            "edges": [
                {"source": "pkg.a", "target": "pkg.b", "type": "calls", "key": 0},
                {"source": "tests.t", "target": "pkg.a", "type": "calls", "key": 0},
            ],
        }
    )
    anchors, source = pp.select_anchors(g, "")
    assert source == "structural"
    assert anchors and anchors[0] == "pkg.a"  # both directions preferred
    assert "tests.t" not in anchors
