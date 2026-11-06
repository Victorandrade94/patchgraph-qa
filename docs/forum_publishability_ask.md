# Forum ask — publishability of derived PatchGraph-QA (DRAFT — do not post automatically)

**Where to post:** Gemma 4 Developer Agent **Paper Track** discussion/forum (same competition hosts).  
**Poster:** Víctor Andrade (manual paste after review).  
**Language:** English (hosts).

---

## Subject

May we publish a derived PatchGraph-QA JSONL (graph-relative Q&A) for the Paper Track?

---

## Body

Hi hosts — thanks for the competition and the code-graph tools.

I’m preparing a **Best New Resource** Paper Track writeup for **PatchGraph-QA**: a small, verifiable question–answer set generated from the competition code graphs (templates such as callers / callees / 2-hop callers / path_exists). Answers are computed with NetworkX from the **provided graphs** (explicitly “correct w.r.t. the graph,” including known async limitations). Items are anchored on symbols touched by each task’s gold patch.

Competition rules forbid redistributing Competition Data to non-participants. Before I ship anything public, could you please clarify what is allowed?

1. **Derived JSONL only** — questions + answers + task ids / template metadata, **without** shipping raw `graphs/*.json`, embeddings, patches, or snapshots. OK to publish (e.g. public GitHub / public Kaggle Dataset)?
2. **Graph summaries** — aggregate stats (node/edge counts, edge-type histograms, coverage rates) in the writeup. Assumed OK; please confirm.
3. **Private Kaggle Dataset** — code + derived JSONL visible only to me / reviewers. OK even if (1) is “no”?
4. If (1) is not allowed, is a **writeup-only** description of the generator (no released items) acceptable for Best New Resource?

Happy to follow whatever constraint you prefer (including releasing nothing until after the competition). Deadline awareness: Paper Track writeup due **12 Nov 2026 23:59 UTC**.

Thanks!  
Víctor Andrade  
Kaggle: TODO_USERNAME

---

## Notes for Víctor before posting

- Replace `TODO_USERNAME` with your Kaggle handle.  
- Link the main-track slug only if you want; keep the ask focused on **derived** artifacts.  
- Do **not** attach competition files to the forum post.  
- After they reply, paste the answer into `docs/writeup_draft_v0.md` §6 / data card.
