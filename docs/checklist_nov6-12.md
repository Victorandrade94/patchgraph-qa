# PatchGraph-QA Paper Track — day checklist (Nov 6–12, 2026)

Deadline: **Thu 12 Nov 2026, 17:59 America/Mexico_City** (= 23:59 UTC).  
Synced to `docs/paper-track-recovery-2026-11-05.md`.  
Scope: Best New Resource writeup ≤3000 words. Gemma E2B optional.

Legend: `[ ]` todo · `[x]` done · `(V)` needs Víctor · `(K)` needs Ken/agent locally

**Remaining TODOs:** host answers (embedding model, publishability forum), related-work polish, data-card license, code/dataset link, writeup word-count pass, upload day7 zip to Drive, submit Thu 12.

---

## Tonight (Wed 5 Nov) — prep without competition data

- [x] (K) Inspect repo at day-3 (`99e77fa`)
- [x] (K) Writeup skeleton → `docs/writeup_draft_v0.md`
- [x] (K) This checklist → `docs/checklist_nov6-12.md`
- [x] (K) Forum publishability ask draft → `docs/forum_publishability_ask.md` (**do not post yet**)
- [x] (K) Branch `paper-track-recovery` + small Colab/export fixes
- [x] (V) Confirm Drive has a fresh code zip matching `patchgraph-qa-day*.zip` (day6 used; day7 next)

---

## Fri 6 Nov — real data + forum ask

- [x] (V) Open Colab with secret `KAGGLE_API_TOKEN` (2026-11-09)
- [x] (V) Put `patchgraph-qa-day4.zip` in Drive; download completed (nested CLI path quirk noted)
- [x] (V) Run All on notebook 01 — sections 8–11 pasted (129/129 graphs; mean cov 0.827)
- [x] (K) Fix anchor `docs_src` mismatch + nested download flatten + baselines cell → day5 zip
- [x] (V) Day-5 Colab: demo fastapi_15588; callers≡callees; baselines 11 items; cov 0.825
- [x] (K) Day-6: reciprocity/neighbors templates, fast download, full §12 corpus export
- [x] (V) Upload **day6** zip; Run All; paste §8 reciprocity + §10–13 (2026-11-09 evening)
- [x] (K) Fill writeup from §8/§12 paste: reciprocity 1.000, 613 items, full baselines
- [ ] (V) Post `docs/forum_publishability_ask.md` on the **paper-track** forum (after a quick personal read)
- [ ] (K) Fill README “Unverified facts” from those outputs (embedding model still UNKNOWN)

**Done when:** notebook 01 completed on real data + forum ask posted.

---

## Sat 7 Nov — coverage table + JSONL v0

- [x] (K/V) Patch→symbol coverage table (mean 0.825; per-repo in writeup §4)
- [x] (V/K) Full corpus JSONL on Colab: 613 items (neighbors 272, neighbors_2hop 195, path_exists 146)
- [x] (K) Templates after reciprocity: `neighbors`, `neighbors_2hop`, `path_exists` (callers/callees collapsed)
- [x] (K) Record item counts in writeup §4
- [ ] (V) Keep JSONL **private** until hosts answer publishability

**Done when:** one JSONL file + counts written into `writeup_draft_v0.md`.

---

## Sun 8 Nov — edge reality + related work

- [x] (K) Reciprocity measured: min/median/max = 1.000/1.000/1.000 over 129 graphs (loader adds none)
- [ ] (K) Related-work bullets expanded (GraphQA, LocAgent, RepoGraph, CG-Bench, TraceEval)
- [ ] (K) Read hosts’ “Note on Code Graphs for Async Functions”; cite in §6

**Done when:** writeup §6 and §8 have real sentences (not only TODOs).

---

## Mon 9 Nov — baselines freeze

- [x] (K) Baselines cell added to notebook §11 (oracle + embedding)
- [x] (K/V) Score **graph oracle** + **embedding heuristic** on full corpus (n=613); seed=0; k=max(1,|gold|)
- [x] (K) Freeze numbers into writeup §5 table — FROZEN 2026-11-09
- [ ] ( ) Optional: Gemma E2B smoke on ~50 items (skip if behind)

**Done when:** §5 table filled and marked FROZEN.

---

## Tue 10 Nov — writeup draft v1

- [x] (K) Day-7: fill v0 in place with frozen numbers + results table + limitations
- [ ] (K) Word count ≤3000; cut encodings / E2B / extra plots first
- [ ] (V) Read once for tone / overclaims

**Done when:** v1 readable end-to-end under the word cap.

---

## Wed 11 Nov — harden + data card + package

- [ ] (K) Data card (§9) completed (size filled; license TBD)
- [ ] (K) Private Kaggle Dataset of code (+ JSONL only if hosts OK)
- [ ] (K) `make zip` → upload fresh **day7** archive to Drive
- [ ] (V) Join/confirm Paper Track competition entry ready for writeup paste

**Done when:** submission package assembled offline.

---

## Thu 12 Nov — submit (before 17:59 CST)

- [ ] (V) Final pass on writeup (limits + publishability caveats visible)
- [ ] (V) Submit Kaggle Writeup
- [ ] (V) Attach / link code zip or private dataset
- [ ] (K) Tag git commit / note submission time in America/Mexico_City

**Hard stop:** 17:59 America/Mexico_City.

---

## Must-have vs nice-to-have

**Must have by Nov 12**

1. Real-data stats from notebook 01 ✅  
2. JSONL for templates on all 129 (613 items) ✅  
3. Two non-LLM baselines with measured scores ✅  
4. Data card + graph-relative-answer caveat + publishability ask (ask drafted; post pending)  
5. Kaggle Writeup submitted  

**Nice:** one Gemma E2B smoke on ~50 items.
