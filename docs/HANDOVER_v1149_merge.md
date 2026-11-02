# Handover — the v1.1.49 upstream merge

Written 2026-11-02 after the job stopped at **gate 2** with the merge not started. The service
is online and healthy on the **pre-merge** build; nothing here is urgent, and nothing here should
be rushed.

## State you are starting from

| | |
|---|---|
| serving now | Maya on **par4**, pid 14988, port 8080, plan `0 1 2 3 4 5 6 7 8`, `retimes 0` |
| deployed commit | `0f50931` = **`v1.1.42-125-g0f50931`**, tag `deployed/v1.1.42-lane-plan-20261101-1834` |
| backup tag | `backup-pre-v1149-20261101` -> `a6e9c34` -> `0f50931` |
| target base | **`v1.1.49` = `26aacb6`** (confirmed local) |
| pushed | branch + both tags on **`sia` only**; `origin` (mw00, upstream) untouched |
| config | `STRATA_GLM_CPU_RETIME=0`, `MAYA_GLM_EXPERT_USED=7` |

## Two findings that change the plan — read before step 1

1. **`series-v1142/` holds 261 patch files, not ~96.** Earlier notes (and `FORK_LAYOUT.md`)
   assume the smaller number. **The paring target is 261 -> N**, and N is unmeasured.
2. **`tools/patch_invariants.txt` CANNOT be run unmodified against a `v1.1.49` base.** Its own
   header says the entries describe the **release-branch (`cd52717`) lineage** and must be
   re-derived from the gated tree. Some entries reading `0` belonged to the replay tree, where
   upstream had already replaced our `slot_load`/butterfly. **Run as-is it fails closed on a stale
   expectation and the failure reads as a merge fault.**

   **Consequence: re-deriving the invariant expectations for the v1.1.49 base is now step 1 of the
   merge, not part of gate 5.** Do it against `0f50931` (verified good) before any patch is applied,
   so the gate has a correct reference to fail against.

## Order (gates 1-2 already done)

1. **Read `docs/FORK_LAYOUT.md` in full** — it is stale in its base and series identity (see below)
   but authoritative on method. Then **re-derive `patch_invariants.txt` for the v1.1.49 base.**
2. Branch from `v1.1.49`. **Base the series on `0f50931`, never on a previous replay.**
3. Regenerate and **pare the series into logical patches; report before/after (261 -> N).**
4. Apply. Collision surface is **5 files**: `glm_model.hpp`, `serve/server.py`, `glm_fast_path.cu`,
   `glm_fast_state.hpp`, `generate.cpp`. **A union is valid only if both sides are pure insertions;
   compiling is not evidence.** Justify each resolution.
5. **GATE: `tools/patch_invariants.py`, fail-closed, blocks everything downstream.** It is the only
   instrument that catches a *removed* clamp — the v1.1.41 replay dropped or thinned **seven**
   changes and **all seven compiled**, one a removed `std::min<size_t>(8, F->stage.size())` that was
   a real host-side out-of-bounds write (invisible to `compute-sanitizer`: a `std::vector` overrun,
   not a device fault).
6. **Grep deliberately for the recurring loss shape**: per-row behaviour reduced to per-call — `n`
   where we passed `1`, `i` where we used `R(i)`, one shared slot where we read `r`-indexed state.
   It has paid out **five times**.
7. Build; verify plan `0 1 2 3 4 5 6 7 8` and `retimes 0`.
8. n=4 smoke: plan coherent, tokens produced, no wedge, no faults.
9. Tag `deployed/v1.1.49-lane-plan-<date>`; release id from **`git describe --tags --match 'v1.1.*'`,
   NEVER `STRATA_VERSION`** (`CMakeLists.txt:11` is `project(strata VERSION 0.1.27)` and its own
   comment says that string never moves with a Maya release — it produced a mislabelled tag on
   2026-11-01). Verify the tag resolves to the built commit.
10. Repoint config, bring Maya up on **par4**, verify **serving** (a real request returning tokens),
    not merely launched. **No `llama-server8080` restart, no `config_drift.py`** — llama.cpp lab only.
11. Record (`FORK_LAYOUT.md` rewrite, `STATE.md`, append-only `TESTING.md`, regenerate
    `maya-patches/` keeping the old series labelled historical), commit logically, push **`sia` only**
    with tags explicit, and verify with `ls-remote`.

**Leave the current build serving until gates 5-8 pass.** Do not take Maya down to do this work.

## Standing honesty requirement

**+6.8 / +11.8 / +23.9% at n=1/2/4 steady are measured on the PRE-merge tree.** After the merge they
are **inherited, not re-verified**. The smoke test is not a substitute. A paired re-measure is its
own job.

## Why `FORK_LAYOUT.md` was not edited yet

Deliberate, and correct: its stored series identity (`96 commits, e16dec85a..cd52717, 39 files`)
describes a merge that does not exist. Writing a "new series" before the series is produced would
leave **two answers in the file** — the exact fault that cost this session. The superseding facts
live in `STATE.md`; `FORK_LAYOUT.md` gets its rewrite **with** the merge, at step 11.
