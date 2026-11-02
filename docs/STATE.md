
## 2026-11-01 - v1.1.49 MERGE: NOT STARTED. Stopped at gate 2.

**State:** backup tag `backup-pre-v1149-20261101` on `0f50931` (the verified-good, currently serving
reference tree). Base confirmed local: **`v1.1.49` = `26aacb6`**. **No branch created, no rebase, no
cherry-pick, no checkout, nothing merged.** Maya was never taken down and is still serving the v1.1.42
build throughout.

**Why it stopped:** the merge was gated on reading `docs/FORK_LAYOUT.md` in full first. Not satisfied in
the available window, and starting a series regeneration without it is the failure that document exists to
prevent (v1.1.41 dropped seven changes; all seven compiled).

**Two facts found before stopping, both needed by the next attempt:**
1. **`maya-patches/series-v1142/` holds 261 patch files**, not the ~96 the earlier notes assumed - so the
   paring target (261 -> N) is bigger than planned and the before/after count must be measured.
2. **`tools/patch_invariants.txt` (42 lines) cannot be run unmodified against a `v1.1.49` base.** Its own
   header says the expectations describe the *release branch* (upstream merged onto `cd52717`) and must be
   **re-derived from the tree being gated, not from memory**; some entries that read `0` there belonged to
   the replay tree where upstream had replaced our `slot_load`/butterfly. Gate 5 therefore needs its
   expectations derived for the new base FIRST, or it will fail closed on a stale expectation and look
   like a merge fault.

**Superseded for the next merge (do not leave two answers in FORK_LAYOUT.md):** the upstream base is no
longer `e16dec85a`, and the series is no longer "96 commits, e16dec85a..cd52717, 39 files". The current
reference is **`0f50931`** (v1.1.42 + 126 commits, `git describe` -> `v1.1.42-125-g0f50931`), and the
conflict surface measured against **v1.1.49** is **5 files of ours (41 changed) against upstream's 54
changed**: `include/strata/core/glm_model.hpp`, `serve/server.py`, `src/core/glm_fast_path.cu`,
`src/core/glm_fast_state.hpp`, `src/program/generate.cpp`.

**Also carried here so it is not only in DECISIONS.md:** the deploy release id comes from
`git describe --tags --match 'v1.1.*'`, never `STRATA_VERSION` (a static strata string that never moves
with a Maya release); "engine left running and verified serving" is a required final deploy step; and
**speculation is closed on Maya by actual test** - MTP *and* DSpark both run, same outcome as llama.cpp,
DSpark being DeepSeek's drafting algorithm (not DGX Spark).

**The +6.8 / +11.8 / +23.9% figures are INHERITED from the pre-merge tree, not re-verified.** A paired
re-measure is a separate job; the smoke test is not a substitute.

## 2026-11-02 - v1.1.49 MERGED, DEPLOYED AND SERVING (supersedes the "NOT STARTED" entry above)

**Live state:** Maya runs **v1.1.49 + 127 commits** -- `90cbc11`, `git describe --tags --match
'v1.1.*'` -> **`v1.1.49-127-g90cbc11`**, tagged **`deployed/v1.1.49-lane-plan-20261101-1954`**
(verified to resolve to that commit), on branch `feat/v1149-from-good`, config **par4**
(`maya-v3q-q4-par4.json`), `MAYA_GLM_CPU_RETIME=0`, `MAYA_GLM_EXPERT_USED=7`, engine
**running and verified serving** (plan `0 1 2 3 4 5 6 7 8`, `retimes 0`, faults 0, a real
request returning tokens). The planner is **adopted**, not queued: `lane_plan`'s wave count
shipped in this build, and `RETIME_V2` and the residency term remain absent (the term is
preserved at `40df967` on `feat/residency-term`, queued as *broken, so fix it*).

**Series identity (the number to quote):** 127 commits, `v1.1.49..90cbc11`, **42 files,
+5925/-325**. Upstream `v1.1.42..v1.1.49` was 50 commits; the colliding surface is **5
files** -- `glm_model.hpp`, `serve/server.py`, `glm_fast_path.cu`, `glm_fast_state.hpp`,
`generate.cpp` -- of which only `serve/server.py` conflicted textually (24 hunks, resolved
by whole-file token-stream comparison: it was v1.1.42's content reformatted, so upstream's
side was taken and our one `[engine env]` print re-applied).

**Evidence (all on the merged build):** `tools/patch_invariants.py` **PASS 15/15** after
re-deriving the manifest for this base and adding four lane-plan entries; per-row audit
across all 41 owned files, **0 markers fell**; build clean; n=4 smoke **4/4 answered, 258
tokens, rotor 59, faults 0**; real request served.

**Note, recorded because the brief assumed otherwise: there is no `docs/TESTING.md` in this
repo.** The append-only evidence log named in AGENTS.md is the **llama.cpp lab's**
(`G:\AI\llama-server\TESTING.md`) and must not be used for Maya. Merge evidence lives here
and in `FORK_LAYOUT.md`; `docs/DECISIONS.md` holds the closed decisions.

**The +6.8 / +11.8 / +23.9% at n=1/2/4 steady were measured on the PRE-merge tree. On this
build they are INHERITED, not re-verified.** The smoke test shows plan and no wedge, not a
rate. A paired re-measure is its own job.

**Next lever, and only one:** `moe_cpu_wait`, ~33% of decode stage time -- the largest
remaining stage, and the one the all-CPU plan itself created. Not opened here.
