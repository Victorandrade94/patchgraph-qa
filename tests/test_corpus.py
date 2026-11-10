from __future__ import annotations

from patchgraph_qa import corpus, questions
from patchgraph_qa import io as pio


def test_build_corpus_and_score(data_dir):
    tasks = pio.load_tasks(pio.find_tasks_file(data_dir))
    graphs = data_dir / "graphs"
    items, summary = corpus.build_corpus(tasks[:1], graphs, max_anchors=2, path_pairs=2, seed=0)
    assert summary["n_graphs"] == 1 and summary["n_items"] == len(items) > 0
    assert "path_exists" in summary["by_template"]
    out = data_dir / "qa.jsonl"
    assert questions.write_jsonl(out, items) == len(items)
    by_id = {t.instance_id: t for t in tasks}
    scores = corpus.score_corpus(
        items, graphs_dir=graphs, emb_dir=data_dir / "embeddings", tasks_by_id=by_id
    )
    assert scores["oracle"]["exact"] == 1.0
    assert scores["n_emb_files_loaded"] >= 1
