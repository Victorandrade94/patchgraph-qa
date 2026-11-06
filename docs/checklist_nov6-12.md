# PatchGraph-QA Paper Track — day checklist (Nov 6–12, 2026)

Deadline: **Thu 12 Nov 2026, 17:59 America/Mexico_City** (= 23:59 UTC).  
Synced to `docs/paper-track-recovery-2026-11-05.md`.  
Scope: Best New Resource writeup ≤3000 words. Gemma E2B optional.

Legend: `[ ]` todo · `[x]` done · `(V)` needs Víctor · `(K)` needs Ken/agent locally

---

## Tonight (Wed 5 Nov) — prep without competition data

- [x] (K) Inspect repo at day-3 (`99e77fa`)
- [x] (K) Writeup skeleton → `docs/writeup_draft_v0.md`
- [x] (K) This checklist → `docs/checklist_nov6-12.md`
- [x] (K) Forum publishability ask draft → `docs/forum_publishability_ask.md` (**do not post yet**)
- [x] (K) Branch `paper-track-recovery` + small Colab/export fixes
- [ ] (V) Confirm Drive has a fresh code zip matching `patchgraph-qa-day*.zip` (or upload new `make zip` output)

---

## Fri 6 Nov — real data + forum ask

- [ ] (V) Open Colab (or Kaggle notebook) with secret `KAGGLE_API_TOKEN`
- [ ] (V) Put newest `patchgraph-qa-day*.zip` in Drive → My Drive (≤3 folders deep) **or** Add Input on Kaggle
- [ ] (V) Open `notebooks/01_explore_graphs.ipynb` → **Run All**
- [ ] (V) Re-run download cell until it prints nothing missing (429 backoff is normal)
- [ ] (V) Save / copy outputs from sections **8, 9, 10, 11** (stats table, embedding check, QA sample, checklist)
- [ ] (V) Paste section-11 checklist into a note or `docs/real_data_checklist_nov6.md`
- [ ] (V) Post `docs/forum_publishability_ask.md` on the **paper-track** forum (after a quick personal read)
- [ ] (K) Fill README “Unverified facts” from those outputs once Víctor shares them

**Done when:** notebook 01 completed on real data + forum ask posted.

---

## Sat 7 Nov — coverage table + JSONL v0

- [ ] (K/V) Finish patch→symbol coverage table (per-repo + overall mean)
- [ ] (K) Run `python scripts/export_qa_v0.py --out artifacts/patchgraph_qa_v0.jsonl` (needs local/Colab data)
- [ ] (K) Confirm all 4 templates present: `callers`, `callees`, `callers_2hop`, `path_exists`
- [ ] (K) Record item counts + refuse rates in writeup §4
- [ ] (V) Keep JSONL **private** until hosts answer publishability

**Done when:** one JSONL file + counts written into `writeup_draft_v0.md` TODOs.

---

## Sun 8 Nov — edge reality + related work

- [ ] (K) Edge-reality sample (edge backed by caller source text) + short negative-edge note
- [ ] (K) Related-work bullets expanded (GraphQA, LocAgent, RepoGraph, CG-Bench, TraceEval)
- [ ] (K) Read hosts’ “Note on Code Graphs for Async Functions”; cite in §6

**Done when:** writeup §6 and §8 have real sentences (not only TODOs).

---

## Mon 9 Nov — baselines freeze

- [ ] (K) Score **graph oracle** on frozen JSONL (expect ~perfect)
- [ ] (K) Score **embedding heuristic** (cosine top-k / threshold); log seed + k
- [ ] (K) Freeze numbers into writeup §5 table — no more metric churn after today
- [ ] ( ) Optional: Gemma E2B smoke on ~50 items (skip if behind)

**Done when:** §5 table filled and marked FROZEN.

---

## Tue 10 Nov — writeup draft v1

- [ ] (K) Promote skeleton → `docs/writeup_draft_v1.md` (or edit v0 in place) with all numbers
- [ ] (K) Word count ≤3000; cut encodings / E2B / extra plots first
- [ ] (V) Read once for tone / overclaims

**Done when:** v1 readable end-to-end under the word cap.

---

## Wed 11 Nov — harden + data card + package

- [ ] (K) Data card (§9) completed
- [ ] (K) Private Kaggle Dataset of code (+ JSONL only if hosts OK)
- [ ] (K) `make zip` → upload fresh archive to Drive
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

1. Real-data stats from notebook 01  
2. JSONL for 4 templates on a documented subset (or all 129)  
3. Two non-LLM baselines with measured scores  
4. Data card + graph-relative-answer caveat + publishability ask  
5. Kaggle Writeup submitted  

**Nice:** one Gemma E2B smoke on ~50 items.
