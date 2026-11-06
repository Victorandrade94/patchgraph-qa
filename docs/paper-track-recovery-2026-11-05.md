# PatchGraph-QA → Gemma Paper Track (recovery, 2026-11-05)

Deadline (from project README + Kaggle "ends in a week" mail to victorandradem94@gmail.com):
Paper Track writeup due **2026-11-12 23:59 UTC = 17:59 America/Mexico_City**.

Live check today:
- Local `/workspace/patchgraph-qa` at `99e77fa` (day-3 zip).
- pytest: 1 skipped, rest passed (venv `.venv`).
- README roadmap still lists Sun 18 → Sun 25 mostly unchecked. No writeup draft found in the repo.
- Recent Drive activity today was the quantum `invsqrt` baseline zip, not PatchGraph.

## Minimal scope that can still ship

Aim: Best New Resource style writeup (≤3000 words on Kaggle) about a **verifiable graph-QA resource** over competition code graphs. Drop Gemma E2B leaderboard runs if time dies.

Must have by Nov 12:
1. Real-data stats from notebook 01 (graph sizes, edge types, patch→symbol coverage).
2. Generated JSONL for at least `callers`, `callees`, `callers_2hop`, `path_exists` on a documented task subset (or all 129 if cheap).
3. Two non-LLM baselines with measured scores (graph oracle + embedding heuristic).
4. Data card + clear "answers are w.r.t. provided graph" caveat + publishability ask to hosts.
5. Kaggle Writeup submitted; code zip or private dataset link.

Nice if time: one Gemma E2B smoke on ~50 items. Not required to submit.

## Day plan (Nov 6–12, light evenings)

| Day | Outcome |
|---|---|
| Fri 6 | Run notebook 01 on real data (Colab/Kaggle). Fill README unverified checklist. Ask hosts about publishing derived QA. |
| Sat 7 | Finish patch→symbol coverage table; export v0 JSONL for 4 templates. |
| Sun 8 | Edge-reality sample + negative-edge note; related-work bullets. |
| Mon 9 | Baselines (oracle + embedding) scored; freeze numbers. |
| Tue 10 | Writeup draft v1 (problem, method, stats, baselines, limits). |
| Wed 11 | Harden, cut scope, data card; private Kaggle Dataset if allowed. |
| Thu 12 | Final pass + submit writeup before 17:59 CST. |

## Writeup skeleton (sections)

1. Problem: agents get code graphs; we lack a label-free check that models use them.
2. Resource: PatchGraph-QA (templates, refuse rules, patch anchors).
3. Construction: graphs from competition data; NetworkX answers; coverage of gold patches.
4. Stats: tasks/graphs/edges (fill from notebook).
5. Baselines: oracle vs embedding heuristic (fill numbers).
6. Limits: async gaps, publishability, answers relative to provided graph.
7. Use: how a Gemma agent paper could cite this as a diagnostic.

## Blockers only Víctor can clear

- Confirm he still wants the Paper Track (vs drop and keep main track only through Dec 2).
- Kaggle/Colab run with his token for real `tasks.jsonl` / graphs.
- Submit the Writeup on Kaggle (and any host forum ask).
