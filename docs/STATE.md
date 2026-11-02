
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
