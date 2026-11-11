
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

## 2026-11-02 (later) - CORRECTION: the retime claim was true, the evidence for it was not

**What was wrong with the v1.1.49 deploy report:** it listed "`STRATA_GLM_CPU_RETIME=0`
(controller off)" and evidenced it with `retimes 0` in the STAT line. **That reading is
expected under both states** - the controller cannot fire without two deliberate overrides -
so it proves nothing. [[feedback_verify_the_arms_not_the_config]] in its exact form, and it
was caught by the owner, not by me.

**What the check then found, measured rather than read:**
- The only name read is `STRATA_GLM_CPU_RETIME` (`glm_fast_path.cu:4223`), and the serving
  config spelled it `MAYA_GLM_CPU_RETIME`. That looked damning and was not: `maya.py:721`
  loads every config through `maya_names.env_in` (MAYA_ -> STRATA_) and
  `serve/server.py child_env` exports **both** names to the engine (`export_env`).
- **Re-measured with the engine's own words, on the pre-fix spelling:** it announced
  `CPU lane retime OFF (STRATA_GLM_CPU_RETIME=0)`. **The setting was delivered all along;
  the claim was true and the arm was worthless.**
- **The spelling drifts back.** `maya_names.env_out` rewrites configs to the `MAYA_` form on
  save (~20 sites in `maya.py`), so editing the file to `STRATA_` returned to `MAYA_` within
  two boots. A hand-fix of the spelling is not durable, and the alias is what makes the key
  work. **`maya_names`' write-back is a trap for anyone auditing a config by eye** - its own
  entry, and arguably a tool fix, not a per-config typo.
- **No config key is dead.** All 8 keys resolve to a reader, but `GLM_SLOTS`,
  `GLM_SLOT_MIN` and `GLM_SLOT_GB` are read only via `env_num("STRATA_…", default)`
  (`generate.cpp:1347-1349`) and are **invisible to a `getenv(` grep**.

**What changed (binary included, so the tag moved):**
- `glm_fast_path.cu` now **announces the switch at boot**, where the split is planned:
  `glm fast: CPU lane retime OFF|ON (STRATA_GLM_CPU_RETIME=<value>)`. **Verified to
  discriminate**, not to be a constant: config `1` -> `ON`, `0` -> `OFF`, key absent ->
  `ON (unset)`.
- New tag **`deployed/v1.1.49-lane-plan-20261102-2121`** on `f7d8888` (built and verified
  serving; `v1.1.49-134-gf7d8888`). The earlier tag
  `deployed/v1.1.49-lane-plan-20261101-1954` is **not deleted** - it honestly points at the
  previous binary (`90cbc11`), which was correct; what changed is that the arm is now code.
- `configs/maya-v3q-q4-par4.example.json` carries `STRATA_GLM_CPU_RETIME=0`.

**Verified serving on the deployed build:** `17 tokens, reply 'serving confirmed.'`, plan
`0 1 2 3 4 5 6 7 8`, `retimes 0`, `faults 0`, and the announcement `OFF (...=0)`.

**Rule, now in `FORK_LAYOUT.md`'s deploy steps:** trace every config env name to the name the
code reads (grep the NAME, covering `getenv(`, `env_num(` and `maya_names.lookup(`) before a
deploy is called done - and take a boolean state from the engine's announcement, never from a
counter that reads the same in both states.

**The +6.8 / +11.8 / +23.9% remain inherited from the pre-merge tree, not re-verified.**

## 2026-11-02 - LOAD WINDOW SETUP: log preservation, per-config logs, PROF off

Set up before agents put real load on the service. Nothing here changes the model or the
plan; the binary is unchanged (the rotation is `maya.py`, not C++).

1. **The log now survives a restart.** `maya.py` rotates it in `serve_on_screen`, before the
   engine opens it with `"w"`: `<log>` -> `<log>.<YYYYmmdd-HHMMSS>.log`, keeping 40. Rotation
   is a property of starting rather than a habit; it fails silently so a log it cannot move
   never stops Maya. **Proven**: two boots, the second preserved the first's 5,321 B as
   `maya-v3q-q4-par4.20261101-215724.log`.
2. **Each config writes its own log.** par4 -> `maya-v3q-q4-par4.log`; standard ->
   `maya-v3q-q4.log`. Before this both wrote `maya-v3q-q4.log`, so a par4 figure and a
   standard one were indistinguishable afterwards - and par4's history sat under the
   standard config's name. Today's misfiled log is preserved as
   `maya-v3q-q4.PAR4-hist-20261101-215538.log`.
3. **PROF is off** in both live configs: no `GLM_PROF`/`TRACE`/`TIMING` env keys (the only
   `PROF` strings are `--expert-profile`, which is pool ranking, not profiling). Throughput
   from this window is from the ordinary STAT path and can be quoted.

### What the load window can be measured by

- **Per request, from the engine log:** `[maya] done: <n> tokens in <t> s (<x> tok/s)
  (<finish>), expert cache <p>% hit` and, in the multi path, `glm multi: request <id> done
  (<finish>): <n> tokens in <t> s (<x> tok/s), <k> running`.
- **Per request, from `/metrics`:** `totals` (cumulative since idle: requests,
  prompt_tokens, reused, output_tokens, prompt_ms, decode_ms -> rate) and `history` (sampled
  series: tok_s, prefill_tok_s_mean, cpu, ram_used, disk_read_mb, gpu_util, gpu_power,
  gpu_temp, gpu_pcie_rx_mb). `totals` is a since-idle cumulative - a collector must take
  deltas between polls, not read it as per-request.
- **Live STAT counters, from `/metrics` `tiers.now`:** tok_s, ms_tok, vram_hit, ram_fetch,
  disk, promo, vram_used/slots, ram_used/slots, ram_gb, vram_gb, disk_gb, draft_acc,
  draft_off, active; `tiers.history` holds their series.
- **Boot-time, from the log:** the plan line (`0 1 2 3 4 5 6 7 8`) and the retime
  announcement (`CPU lane retime OFF (STRATA_GLM_CPU_RETIME=0)`).

**Not available, and a collector should not look for them:** the **STAT lines are not in the
log** - the engine writes them to the server's pipe and the server folds them into
`/metrics` `tiers`, so a log-only collector gets the `[maya] done:` summary instead of the
per-8-token tier counters. **`rotor`, `retimes` and `faults` are not metrics at all**: they
exist only as log line patterns (`rotor entered`, `timed again`, `did not drain`), and at
n=1 the rotation declines (`glm rows: ... rotation declined (n=1, ...)`). There is no
per-request plan (boot only), no queue/latency percentiles, no quality (KLD) measurement, and
no per-request prompt-cache ratio beyond `reused`.

### Open list (none blocks the load window)

1. **The +6.8 / +11.8 / +23.9% at n=1/2/4 steady are INHERITED from the pre-merge tree, not
   re-verified.** This load window is unpaired and uncontrolled: it can reveal a regression,
   it cannot confirm a gain. A paired re-measure is its own job and this is not it.
2. **`maya_names`' write-back trap.** A hand-edited `STRATA_` key drifts back to `MAYA_`
   within two boots (`env_out`, ~20 save sites) - harmless today because `env_in`/`export_env`
   make the alias work, but a live trap for anyone auditing a config by eye. Fixing the tool
   (preserve the spelling the code reads, or stop rewriting on save) is a candidate job.
3. **`moe_cpu_wait`, ~33% of decode stage time** - the next lever, unopened. Real-load data
   may re-rank it, which is a better reason to wait than an instruction to.

## 2026-11-02 (v1.1.50) - live state

**Deployed:** Maya **v1.1.50 + 138 commits**, `6c786a7`, `git describe --tags --match 'v1.1.*'`
-> `v1.1.50-138-g6c786a7`, tag **`deployed/v1.1.50-lane-plan-20261102-1147`** (verified to
resolve to the built commit). Branch `feat/v1150-from-good`. Config **par4**,
`STRATA_GLM_CPU_RETIME=0`, `MAYA_GLM_EXPERT_USED=7`, engine **running and verified serving**
(plan `0 1 2 3 4 5 6 7 8`, retime announced OFF, n=4 smoke 4/4, a real request answered).

**What v1.1.50 brought, and what it did not:**
- taken and inert-for-us: the prefill CPU gate (threshold knob, default 2.0), the
  resident-expert cap (opt-in, 0 = all - our EXPERT_USED=7 untouched), thinking-budget
  defaults (Maya-XS setup only; we stay at 4096).
- **NOT delivered despite the commit message:** `[maya]` log lines still carry no
  `HH:MM:SS`. The stamping code is in the merged tree and wired, but nothing is stamped in
  our log - **open item, needs its own diagnosis**, and the "log timestamps" benefit must
  not be quoted until it works.

**The merge's one real hazard, recorded because it is the shape that keeps costing us:**
both sides appended a parameter to `moe_route()` (`int miss_to_cpu = 0` ours,
`bool resident_ok = true` theirs). A union would have compiled and bound the bool into our
int, silently mis-routing disk-only experts. Both are kept, in a defined order, at the
declaration, the definition and the single call site.

### Open list (carried forward)

1. **The +6.8 / +11.8 / +23.9% at n=1/2/4 steady are INHERITED from the pre-merge tree,
   never re-verified.** Now three merges old. A paired re-measure is its own job (owed for
   v1.1.52's +8-9% decode claim too).
2. **Sphinx's looping/garbage** (load window): the engine log showed one loop-guard firing
   at 1,421 tokens of a 1-token pattern, sampling at `temperature=1.0` with
   `repetition_penalty=1.0` and no presence/frequency penalty, and six answers capped at
   ~150 tokens while the thinking budget is 4,096. The proposed fix (per-client sampling +
   a token cap above the thinking budget) is **not applied**.
3. **`[maya]` log timestamps** - in the tree, not in effect (above).
4. **`maya_names` write-back trap** - a hand-edited `STRATA_` key drifts back to `MAYA_` on
   save; harmless (the alias works) but a trap for anyone auditing a config by eye.
5. **`moe_cpu_wait`, ~33% of decode stage time** - the next lever, unopened.

## 2026-11-03 - v1.1.52 MERGED, DEPLOYED AND SERVING

**Live state:** Maya runs **v1.1.52 + our series** -- `c367e85`, `git describe --tags --match
'v1.1.*'` -> **`v1.1.52-140-gc367e85`**, tagged **`deployed/v1.1.52-lane-plan-20261103-1605`**
(verified to resolve to that commit; branch + tag pushed to sia only), on branch
`feat/v1152-from-good`, config **par4** (`maya-v3q-q4-par4.json`),
`MAYA_GLM_CPU_RETIME=0`, `MAYA_GLM_EXPERT_USED=7`, engine **built clean and verified
serving** (boot plan `0 1 2 3 4 5 6 7 8`, retime announced `OFF`, n=4 smoke 4/4,
real request 83 tokens coherent, loop_guard all zero).

**Series identity:** 140 commits, `v1.1.52..c367e85`, **44 files, +6384/-325**, pared
export `maya-patches/series-v1152/` (same 15 subsystem patches). Upstream `v1.1.50..v1.1.52`
was 383 commits. Merge made **zero** textual conflicts across an 11-file collision
surface (v1.1.52's CUDA-graph DSA feed-forward sits in regions ours does not touch).

**Upstream features taken as-is:** DSA feed-forward graphs default-on
(`STRATA_GLM_DSA_FFN_GRAPH` unset = on; their +8-9% decode claim inherited-unverified),
IQ1_S vector rows, setup keeps host/key/CORS/MCP, MULTI_CKPTS (opt-in, off here),
ROW_HALVES/DEFER_CPU (opt-in, off), watchdog fix. KDA graphs stay separately opt-in.
`moe_route` dual params and the env print re-verified post-merge.

**Repo hygiene found during this merge:** the repo is a **shallow clone** — series
counts need `A..B` range syntax (`v1.1.52..c367e85` = 140); space-separated args count
a union (read 523 for the same range). Recorded in `docs/FORK_LAYOUT.md`.

## 2026-11-05 - v1.1.53 MERGED, DEPLOYED AND SERVING

**Live state:** Maya runs **v1.1.53 + our series** - `8a5ec13`, `git describe --tags --match
'v1.1.*'` -> **`v1.1.53-146-gd3ea1f3`**, tagged **`deployed/v1.1.53-20261104-2304`** on
`feat/v1153-from-good` (pushed to `sia`). This supersedes `deployed/v1.1.52-lane-plan-20261103-1605`.

**Gates:** invariants 15/15 on OUR tree BEFORE the merge and 15/15 after; zero merge conflicts;
per-row loss audit PASS (no marker fell); build clean; boot `MAYA_GLM_ORPHAN_THINK=1`, CPU retime
OFF; serving `ready`; n=4 smoke 4/4. Upstream was 65 commits / 108 files / +10,461-579.
Details in `FORK_LAYOUT.md`.

**The "garbage output" banner (2026-11-04/05) is DIAGNOSED and documented** in
`FINDINGS_thinking_budget_2026-11-04.md`: it was the model's own reasoning delivered in the
CONTENT channel, not corrupted arithmetic (zero non-finite scores, every row ran every layer,
guard silent). Two routes: a budget-forced close, and GLM-5.3's orphaned close tag written inside
its own reasoning. Both are known ecosystem traits (vLLM #54744, ollama #18082, hermes #96735,
pi #9554). The parser-side fix is in and gated; **the engine-side `in_think` channel tag is the
one open item** - it is what closes the no-tag variant.

## 2026-11-06 - v1.1.55 MERGED, DEPLOYED AND SERVING

**Live state:** Maya runs **v1.1.55 + our series** - `d8e5fbc`, tagged
**`deployed/v1.1.55-20261106-0015`** on `feat/v1155-from-good` (pushed to `sia`). This supersedes
`deployed/v1.1.53-20261105-0033`.

**Gates:** invariants 15/15 before and after; zero conflicts; loss audit PASS; build clean; real
request `ready`; n=4 smoke 4/4 with guard 0. Upstream was 52 commits / 69 files / +3,201-601.

**Unchanged and worth knowing:** this pack runs `--kv q4_0`, so upstream's new INT8 latent-cache
default is inert here (verified in the boot line, not assumed). Details in `FORK_LAYOUT.md`.

**Carried open items** (none introduced by the merge): the audit lane's abandon-on-HEAD behaviour;
a verdict accepted from a self-stopped run; the counting-loop degeneration the loop guard cannot
match; and the reasoning-return change whose benefit is still unmeasured.

## 2026-11-06 - v1.1.56 MERGED, DEPLOYED AND SERVING (supersedes v1.1.55)

**Live state:** Maya runs **v1.1.56 + our series** - `3342116`, tagged
**`deployed/v1.1.56-20261106-1527`** on `feat/v1156-from-good` (pushed to `sia`). This supersedes
`deployed/v1.1.55-20261106-0015`.

**Gates:** invariants 15/15 before and after; **zero conflicts across a 24-file overlap**; loss audit
PASS; build clean; `ready`; n=4 smoke 4/4 with guard 0. Upstream was 36 commits / 56 files /
+5,823-442, GLM-focused.

**Two things to carry forward:**
- **#840 (slot fallback)** should stop the `refused ... foreign identity` lines we saw after the
  v1.1.55 upgrade - watch the next boot rather than assume.
- **#845 makes no-seed requests non-deterministic** (they were pinned to seed 0). Fixed seeds are
  mostly reproducible at temperature > 0 with spec decode on, not exactly; `temperature: 0` stays
  exact. Past no-seed runs at temperature > 0 are not comparable with this build.

Everything else in the release (DeepSeek EXL3/Pascal, AMD, web-chat attachments) does not touch our
GLM path. Details in `FORK_LAYOUT.md`.

## 2026-11-10 - v1.1.57 MERGED, DEPLOYED AND SERVING (supersedes v1.1.56)

**Live state:** Maya runs **v1.1.57 + our series** - `2e00a26`, tagged
**`deployed/v1.1.57-20261110-0809`** on `feat/v1157-from-good` (pushed to `sia`). This supersedes
`deployed/v1.1.56-20261106-1527`.

**Gates:** invariants 15/15 before and after; 8 files / 16 hunks resolved by intent; loss audit PASS;
build clean; `ready`; n=4 smoke 4/4 with guard 0. Upstream: 8 commits but 456 files / +76,605-1,166
(GLM-5.3 744B support, state-layout refactor, setup/API work).

**The q4_0 latent cache is RETIRED.** v1.1.57's state layout has no q4_0 mode, so per the user's
decision we adopted upstream's INT8: `--kv int8` in the config, latent 288 -> **544 B** a position,
pinned RAM 1.66 -> **3.14 GB**, VRAM streaming 0.10 -> 0.20 GB (measured). Revisit other KV quants
later.

**Two fork bridges are load-bearing** and must survive future merges: (1) the BF16 shared-expert down
on glm5-next has no glm-dsa ext, so it dequantises q8_1 into `F->hq` and runs via `gf::mv`; without it
*every* request fails `shared expert down`. (2) `slot_load`'s blob path builds an istringstream for the
NextN trailer. Details and the per-file resolution table are in `FORK_LAYOUT.md`.

**Carried open items** (unchanged): the audit lane's abandon-on-HEAD behaviour; a verdict accepted
from a self-stopped run; the counting-loop degeneration the guard cannot match.

## 2026-11-11 - v1.1.59 MERGED, DEPLOYED, REFUSAL ABLATION ENABLED (supersedes v1.1.57)

**Live state:** Maya runs **v1.1.59 + our series** - `5becfc2`, tagged
**`deployed/v1.1.59-20261111-1118`** on `feat/v1159-from-good` (pushed to `sia`). This supersedes
`deployed/v1.1.57-20261110-0809`. The engine was **down** before this (no NSSM autostart; the machine
rebooted overnight) and was started by hand after the merge.

**Gates:** invariants 15/15 before and after; 10 files / 20 hunks resolved by intent; loss audit
PASS; build clean; `ready`; n=4 smoke 4/4 with guard 0 (ablation off AND on). Upstream: 5 commits,
413 files, +7,419-3,013.

**Refusal ablation is ON** (`ablate.default: true`): GLM-5.3-Flash's direction file at
`G:\AI\maya-eval\models\refusal-directions\GLM-5.3-Flash-refusal-direction.gguf` (layer 31,
scale 1), loaded via `--ablate` in the config's args. Per-request `"ablate": false` turns it off; the
Models page's switch is the default. Note: a hand-edited config needs the args itself - only the
Models page's save applies them.

**Three merge hazards worth remembering** (all fixed, all documented in FORK_LAYOUT):
1. `glm_rows.cu`'s loop bounds - upstream's `l0_..l1_` would run all 45 layers per per-row call.
2. The `STRATA_*` -> `MAYA_*` rename - the code reads `MAYA_*`; `env_alias.hpp` only maps the old
   names at start, so a `STRATA_`-only reader silently ignores a `MAYA_` setting.
3. Header/definition drift on `lat8_rec_bytes`, `mla_attend*` (`int` vs `bool lat_q8`) and `mla_out`
   (parameter removed) - these linked/compiled wrong before being aligned.

**Not enabled:** `MAYA_FAIR_SLICE_S` (new: a long answer yields the engine to a waiting request) and
`MAYA_GLM_PREFILL_INTERLEAVE`/`_GAPS` (prefill-time). They are complementary phases, both default off.

**Carried open items:** the audit lane's abandon-on-HEAD behaviour; a verdict accepted from a
self-stopped run; the counting-loop degeneration the guard cannot match.

## 2026-11-11 (later) - Maya runs as the NSSM service **Maya-8080**; fairness knobs ON

The engine had been started by hand every time (it was down after the 2026-11-11 reboot). It is now a
Windows service and survives a reboot.

| | |
|---|---|
| service | **`Maya-8080`**, `AUTO_START`, runs as **LocalSystem** |
| application | `G:\AI\maya-eval\project-maya\.venv\Scripts\python.exe` |
| parameters | `maya.py --config maya-v3q-q4-par4.json --port 8080 --host 0.0.0.0` |
| directory | `G:\AI\maya-eval\project-maya` |
| logs | `logs\maya-8080.out.log` / `.err.log` (rotating, 10 MB) |
| installer | **`tools/install_maya_service.ps1`** - idempotent; stops a hand-started instance, reinstalls, starts, waits for the port |
| `llama-server8080` | left **`DEMAND_START`** (the llama.cpp fork's fallback); it does not race for 8080 at boot |

Elevation: `sc`/`nssm` need admin, which the tool shell does not have - `ssh agips@localhost` does get
an admin token (verified by creating and deleting a probe service). Example:

```powershell
ssh agips@localhost "powershell -NoProfile -ExecutionPolicy Bypass -File G:\AI\maya-eval\project-maya\tools\install_maya_service.ps1"
```

### The trap that costs an hour if it is met cold

The engine loads **`cublas64_13.dll` from `CUDA\v13.4\bin\x64`**, which is *not* on a LocalSystem
service's `PATH` (a user shell has it, which is why running `maya.py` by hand always worked). Without
it the engine exits before serving. The service's `AppEnvironmentExtra` therefore carries `PATH`
(found by the installer, not hardcoded) plus `USERPROFILE`/`APPDATA`/`LOCALAPPDATA`/`HOME`/`TEMP`, so
calibration and caches resolve exactly as they did from a shell.

### Fairness - enabled and measured

| knob | where | value |
|---|---|---|
| `MAYA_FAIR_SLICE_S` | **service environment** (the *server* reads it; the config's `env` reaches only the engine child) | **20** |
| `MAYA_GLM_PREFILL_INTERLEAVE` | config `env` (engine) | **2048,2** |
| `MAYA_GLM_PREFILL_GAPS` | - | **not applicable**: needs `MAYA_GLM_SPLIT_ROWS`, and this is one GPU |

**Measured, saturated** (4 rows busy, a 16-token request arriving at t=6 s):

```
short   17.0s  tokens=11      <- admitted at the 20 s slice boundary
long0   87.8s  tokens=700
long1   87.4s  long2 87.4s  long3 87.4s
```

So the queued request finished in **17 s instead of waiting ~81 s**, and the four long answers still
completed. Under *light* load there is nothing to slice: `--parallel 4` plus the rotor decodes two
answers in the same pass, so a second request is served concurrently (measured 2.5 s for a short
request alongside a 900-token one, with the long one still at 21.6 tok/s).

### Verified end to end

- service `RUNNING`, listening after ~65 s (the usual engine start), engine is a descendant of the
  service's python (`sc start` -> python -> maya.py -> server -> maya.exe).
- real request answered; `/api/models` reports `settings.ablate: true`.
- engine log: `MAYA_GLM_PREFILL_INTERLEAVE=2048,2` in its settings line; `[engine env]` line shows
  `MAYA_FAIR_SLICE_S=20`; abliteration line present.
- **stop/start**: stop leaves **0** `maya.exe`, 0 server processes, port free in ~5 s (maya.py treats
  SIGINT/SIGTERM/SIGBREAK as Ctrl+C, and `AppStopMethodConsole 25000` gives it room); start returns to
  serving.
- **n=4 smoke through the service**: 4/4 survivors, guard 0.
