# Fork layout — branches, tags, patches, and the conflict surface

**Read this before answering any question about where our changes live, and before any
upstream merge or rebase.** Written 2026-10-31 after a full day was spent re-deriving
facts that `git branch -a` would have printed in one second.

## The rule this document exists to enforce

> **Inventory before answering.** `git branch -avv`, `git tag`, `git stash list`,
> `git worktree list`, `git remote -v`. A structural claim about this repo made without
> running those is a guess, and two were made on 2026-10-30 — "our changes are not in
> their own branches" (false: 22 of them are, 7 pushed to Sia) and a bisect across a
> rewritten lineage (the range was empty; `git merge-base --is-ancestor` would have said so).

## Upstream base — CURRENT (2026-11-03): v1.1.52

**Supersedes the `v1.1.49` / `v1.1.50` identities this section used to carry. Read only
this one — a document with two bases is how a merge got started against the wrong tree.**

Our tree is now **upstream `v1.1.52` (`3c935e4`)** with our work merged onto it, on branch
**`feat/v1152-from-good`**. The merge commit is `c367e85` (parents `615e36b` +
`3c935e4`); the built and deployed commit is `c367e85`
(`git describe --tags --match 'v1.1.*'` → **`v1.1.52-140-gc367e85`**), tagged
**`deployed/v1.1.52-lane-plan-20261103-1605`** (verified to resolve to that commit;
branch and tag pushed to **sia only**).

**"Our changes", as of this merge: 140 commits, `v1.1.52..c367e85`, 44 files,
+6384/-325.** That is the series the export is generated from and the number to quote.
(Count with `A..B` range syntax: this repo is a **shallow clone**, and space-separated
args count a union, which once read 523 for this same range.)

The backup tag for this merge is `backup-pre-v1152-20261103` → `615e36b` (the deployed
v1.1.50 tree plus docs). The running binary it preserves is kept at
`build/rollback-maya-v1150.exe` until the new build verifies serving.

## The reference tree

**`0f50931` is the verified-good pre-merge tree** — `v1.1.42-125-g0f50931`, deployed and
observed serving (plan `0 1 2 3 4 5 6 7 8`, `retimes 0`, 44-token reply, pid 14988 on
port 8080). Tagged `backup-pre-v1149-20261101`. **Base every future series on it, never on
a previous replay.** It supersedes `cd52717` as the reference: `cd52717` was the verified
`n=2` de-fused rotor (41-47 rotor entries, ~24 tok/s, 3/3 boots, tagged
`backup-pre-20261030-0835`), and is now historical — still the right reference for any
claim about *that* lineage, but not the base for a new merge.

## Branches (2026-10-31)

| Branch | Purpose |
|---|---|
| `feat/rows-de-fused-pf` | the de-fused decode rotation — the main line of work |
| `feat/v1142-from-good` | the v1.1.42 merge, deliberately based on `cd52717`, not on the failing v1.1.41 replay |
| `feat/prefill-align` | prefill alignment + the layer-stationary assessment |
| `fix/rows-bf16-serial` | per-row serialisation when the KDA output is BF16 |
| `fix/mvnt-bf16-xf` | `mv_nt` q8_1 dequantise for BF16 jobs |
| `fix/rows-mvnt-fallback` | **WIP, NOT SHIPPABLE** — wrong output for n>1 |
| `fix/step-mutex` | **WIP, does NOT fix** the KDA output failure |
| `homelab/v3q-support` | v3q pack support + the prompt-cache `.meta`/`.bin` fix |
| `integrate/picks` | integration of the picks below |
| `pick/118-fetch-overlap` | **feature DEFECTIVE, not enabled** |
| `pick/487-iq-mmq`, `pick/560-iq-mmvq` | upstream PR picks |
| `pr/118`, `pr/487`, `pr/560`, `pr/582` | what we sent upstream |
| `backup/*` (4, dated) | restore points — do not delete without checking for unique work |

Remotes: `origin` = github.com/mw00/project-maya (upstream), `sia` =
git.ankh-labs.com/ankh-labs/project-maya (**ours — 7 branches pushed**).
`main` is at v1.1.29 and 254 behind; it is not where our work lives.

## Exported patches — `G:\AI\maya-eval\maya-patches\`

Regenerate after significant work. `README.md` there is the index.

- **`series-v1149/` — CURRENT.** The series against the v1.1.49 base, **pared into logical
  patches by subsystem** instead of one patch per commit: **261 -> 14** (14 subsystem
  patches covering all 43 changed files). The count is the point — a patch per commit is a
  replay log, and replaying one is what dropped seven changes on the v1.1.41 attempt.
  `maya-patches/README.md` is the index and lists every patch.
- `by-file/` — one squashed diff per touched file, to review or re-apply one subsystem
- `reference/CONFLICT-SURFACE.txt`, `CONFLICT-FORECAST-*.txt` — the tables below
- **`series-v1142/` (261 files, `e16dec85a..f383c28`) and `series-cd52717/` (96 files) are
  HISTORICAL.** Keep them; they are the record of the two previous lineages. Do not apply
  them to anything: they are based on trees that v1.1.49 has moved past.

## The conflict surface — the number that decides how we organise

**CURRENT (2026-11-02, measured against v1.1.49): 5 of our 42 touched files are also
touched upstream across v1.1.42 -> v1.1.49.** Ours changed 42 files, upstream 54.

| File | upstream lines | our changed lines | resolution this merge |
|---|---|---|---|
| `src/core/glm_fast_path.cu` | 7747 | 339 / 41 | auto-merged; upstream added 3 lines (Monitor's `all_bytes`) |
| `src/program/generate.cpp` | 9255 | 262 / 16 | auto-merged; upstream added 65 lines (STAT fields, spec counters, serve cache floor) |
| `src/core/glm_fast_state.hpp` | 961 | 193 / 2 | auto-merged; upstream added 1 line (`all_bytes`) |
| `include/strata/core/glm_model.hpp` | 1087 | 34 / 4 | auto-merged; upstream added 1 line (`all_gb`) |
| `serve/server.py` | 6186 | 5 / 0 | **the one textual conflict, 24 hunks — see below** |

**Superseded:** the earlier table ("14 of our 39 touched files ... across v1.1.39 -> v1.1.42",
with `glm_fast.cu`, `glm_prefill.cu`, `CMakeLists.txt` in the set) described the previous
lineage. It is kept only as history. **The instruction it earned still stands and is the
reason this merge was cheap: extract logic behind seams rather than editing upstream files
in place — `glm_rows.cu`, our largest change, collides with nothing because it is ours.**

### `serve/server.py` — why "theirs" was the correct resolution, not a shortcut

The 24 hunks looked like a large merge and were **formatting**: our `server.py` was
v1.1.42's content reformatted (1255 lines over 100 characters -> 675). Resolved by
**whitespace-collapsed token-stream comparison of the whole file**, not by the compiler and
not by hunk-by-hunk guessing:

- it found exactly **two** ours-only segments: the `[engine env]` print (genuine — re-applied
  after `maya_names.export_env`) and one statement upstream had merely *moved*;
- therefore taking upstream's file whole loses **no behaviour**, and drops our reformatting,
  which also deletes the conflict surface for the next merge.

Upstream's file was taken; the env print was re-added at the same site; `py_compile` passes.
**The general lesson: when a conflicted file is a reformat of the other side, the token
stream is the instrument, and a union of 24 hunks would have been 24 chances to reintroduce
the loss shape this document exists to prevent.**

## What the v1.1.41 rebase cost, and the gate against a repeat

The replay silently dropped or thinned **seven** of our changes. **All seven compiled.**
Five were found only by reading three versions side by side (ours at `cd52717`,
upstream's, the replay's).

- Four per-row hunks reverted to per-call behaviour
- The mover's `std::min<size_t>(8, F->stage.size())` batch clamp — a real host-side
  out-of-bounds write at `MAYA_GLM_EXPERT_USED=7`, invisible to `compute-sanitizer
  --tool memcheck` because it is a `std::vector` overrun on the host
- `slot_load`'s parallel-read path and the `kv_norm` butterfly — recorded, not yet restored

**`rerere` does not help**: it reproduces a *wrong* resolution faithfully, and
recordings from a forward replay reinstate the newer side if you later go backwards.

**The gate is `tools/patch_invariants.txt` plus its fail-closed checker.** Patterns and
expected occurrence counts for the expressions our fork depends on, derived from
`cd52717`, including entries marked as acknowledged-divergent so a known difference reads
as acknowledged rather than as a failure. **Run it before tagging anything**, and treat a
missing pattern or an unreadable file as a failure, never a skip. A count-based audit is
what catches a *removed clamp*, which adds and removes nothing and so defeats every other
instrument.

## Merge procedure

1. Branch from the **last verified-good tree**, not from the previous replay. Note the
   consequence: tooling added *after* that tree (the invariant manifest, any later
   guard) does **not** come with it and must be cherry-picked across explicitly. A
   manifest entry that post-dates the base is legitimate but must be labelled
   `POST-BASE` with its commit, never silently dropped.
2. Resolve each known site from the three-way reading, not from the compiler. The
   recurring loss shape is **per-row behaviour quietly reduced to per-call** — anything
   passing `n` where we passed `1`, indexing by `i` where we used `R(i)`, or reading one
   shared slot where we read `r`-indexed state.
3. Run the invariant checker. It must pass.
4. Three boots at `n=2`, rotor entry count and tok/s read from the log. No single-run
   cross-build comparison: this failure is a Heisenbug (see `LAYER_PIPELINE.md`).
5. Only then tag and deploy. **A Maya deploy is NOT the llama.cpp lab's deploy** (verified
   2026-10-31, and the mistake was made once): there is **no `maya` NSSM service**
   (`sc query maya` → 1060); the only service on the box is `llama-server8080`, which runs the
   **llama.cpp** fork's binary. `config_drift.py` lives in `G:\AI\llama-server\tools\` and is
   **that** lab's checker — Maya has none. So a Maya deploy is: build, the config `.json`
   (keeping `MAYA_GLM_EXPERT_USED=7`), a `deployed/<version>-<date>` tag on the exact commit
   that was built (not on a later docs commit), and the docs plus the regenerated export.
   **Do not restart `llama-server8080` for a Maya change, and do not run the llama.cpp drift
   checker.** Carry deploy rules across projects only after checking they apply — that is the
   same failure as inferring repo structure without running the inventory command.
   **The deploy is not finished until the engine is up and verified SERVING** — see step 6.
6. **Start it, then prove it serves.** `maya.py` on the deployed config, then a real request
   returning tokens, the `plan` line, and `retimes`/`faults`. **A process existing is not
   readiness.** On 2026-11-01 a deploy ended with a verified artefact and an **idle box**:
   tag, commit, config and smoke all done, engine stopped. The contrast that must not be
   confused: **`llama-server8080` is `DEMAND_START` and STOPPED is its normal resting state
   (the owner arbitrates the GPU with it); Maya is the opposite — it is the thing that
   should be UP**, on the latest build, except while restarting or fixing a failed restart.
   Note the practical consequence for a merge: **the build cannot relink `maya.exe` while
   the engine holds it** (`LNK1104: cannot open file 'maya.exe'`). Copy the running binary
   aside as a rollback, stop, build, then bring it back up and verify — the old build serves
   until the gate has passed and must not be taken down before it.
7. **Trace every config env name to the name the code READS, and get the state from the
   engine's own announcement.** Added 2026-11-02 after a deploy was signed off with an
   invalid arm:
   - **`retimes 0` is NOT evidence that the CPU-lane retime controller is off.** It is
     expected whether the controller is OFF or merely DORMANT (it needs two deliberate
     overrides to fire). The evidence is the boot line
     `glm fast: CPU lane retime OFF|ON (STRATA_GLM_CPU_RETIME=<value>)`, and it was verified
     to **discriminate**: config `1` -> `ON`, `0` -> `OFF`, key absent -> `ON (unset)`.
     Boot it both ways once, so the arm is known to move.
   - **Grep the NAME, not `getenv(`.** Three keys in the live config
     (`GLM_SLOTS`, `GLM_SLOT_MIN`, `GLM_SLOT_GB`) are read only through
     `env_num("STRATA_…", default)` (`generate.cpp:1347-1349`) and are invisible to a
     `getenv(` search; `maya_names.lookup(` reads env without `getenv` too. A config key
     that looks unread is usually a missed read path — and the reverse mistake (assuming a
     spelling is what is read) is what produced this rule.
   - **The on-disk spelling does not decide what the engine reads, and a hand-fix will not
     survive.** Configs are written in the `MAYA_` form (`maya_names.env_out`, ~20 save
     sites in `maya.py`), read as `STRATA_` (`env_in`), and `serve/server.py child_env`
     exports **both** names to the engine (`export_env`). Measured: with the file spelling
     `MAYA_GLM_CPU_RETIME`, the engine announced `OFF (STRATA_GLM_CPU_RETIME=0)`; editing
     the file to `STRATA_` drifted back to `MAYA_` within two boots. So the alias layer is
     what makes it work, and `maya_names`' write-back is a live trap for anyone auditing by
     eye — its own entry, not a typo to fix per config.

## 2026-10-30 (late) — v1.1.42 merged onto the verified-good tree: a working n=2 rotor

**Branch `feat/v1142-from-good`, merge commit `f383c28`, tagged `deployed/v1.1.42-20261030-2106`.**
Base: `cd52717` (the verified working `n=2` rotor), NOT the v1.1.41 replay - that one fails 0/3 for reasons we do not
understand, and merging onto it would make every new symptom ambiguous between three origins.

### The arm (three boots, `n=2`, 512 tokens, config-only)

| boot | tokens | finish | time | rotor entries | drain | non-finite |
|---|---|---|---|---|---|---|
| 1 | 46 + 117 | stop / stop | 7.7 s | **45** | 0 | 0 |
| 2 | 45 + 128 | stop / stop | 8.0 s | **44** | 0 | 0 |
| 3 | 69 + 220 | stop / stop | 32.4 s | **287** | 0 | 0 |

**Clean 3/3.** ~21 tok/s on the two normal boots (163 tokens in 7.7 s; 173 in 8.0 s); the long third turn ran 289 tokens
over 287 rotor entries at 8.9 tok/s - more layers published per token means more misses, which is the mover doing its job.

**Boot 3 is a different regime, not a larger boot 1.** It ran 289 tokens in 32.4 s (112 ms/token) against boot 1's 163 in 7.7 s (47 ms/token), and published 287 rotor steps against 44-45. Its request B generated 220 tokens where boot 1's generated 117, so the generation extended past what the resident expert set covers and the per-token cost should rise: the shape of cache saturation, not of a schedule regression. One measurement settles it and it is already instrumented: run the pair with
`STRATA_GLM_MISS_LOG=1` and read experts-over-PCIe per step; if the misses grow with the token index, it is the cache, and
the short boots simply never exercised it.

`tools/patch_invariants.py` **PASSES** on the merged tree (11 invariants, `rotor_will_run` acknowledged as absent by design
on this lineage).

### What this is, and what it is not

**This routes around the wedge; it does not fix it.** v1.1.42 touches neither the MoE routing ring, `wait_ch`, the
response channels nor ring publication (only `fast_service`'s idle gate, + config env naming). The `ch = 4` consume and
the fully-accounted wedge remain unexplained, and the ring-entry instrument has since reported **"ring entries consistent
with the host's record"** - so the entry the service reads is the one published, and the fault is downstream of the read.

### The eight conflict resolutions (three-way, not compile-driven)

`serve/server.py` took upstream's `maya_names.export_env` and kept our engine-env print. `glm_fast_path.cu` 1017/4119
were unions of additions (plus the `resp_h` separation from `bf3b16f`, which our per-channel indexing needs); 4751 took
upstream's streaming block but kept **our `rq->ch` indexing** of the response. `glm_fast.cu` 4007/6777 unions, with
upstream's ROUTE_FAST pre-write kept but **indexed by the route's own `hseq`** instead of `*a.d.seq + 1` (a shared counter
would let a later row write its 16 KB FFN input into a slot another row had published); 4047 ours plus their `!rank_sel`;
4138 **ours** - `*moe_wseq(a.d)`, not upstream's `*a.d.wait_seq` (channel 0), the same reduction as `wait_ch`.

Two compile fixes that were themselves informative: a duplicate `s_near`, and `q->pad_`, which no longer exists because
the merged `MoeRequest`'s slot is `int ch` - the waiting row's channel.

### Export regenerated

`G:\AI\maya-eval\maya-patches\series-v1142\` - 261 `format-patch` files (4.7 MB) for `e16dec85a..f383c28`, plus
`by-file/src_core_glm_fast_path.v1142.diff` and `by-file/src_kernels_cuda_glm_fast.v1142.diff` as the three-way reading
material for the next merge.

### Not done, deliberately

Repointing the services: **there is no Maya NSSM service** (`sc query maya` -> 1060) and `config_drift.py` lives under
`G:\AI\llama-server\tools\` - both belong to the llama.cpp lab, not this project. The Maya engine is launched from the
config (`exe: build\maya.exe`) by `maya.py`, so the deploy step is the config plus that server, not an NSSM restart.
Confirm before anything is restarted.

## 2026-11-02 — v1.1.49 merged: the plan survived upstream's seven releases

**Branch `feat/v1149-from-good`.** Merge commit `6edde5d` (parents `0f50931` + `26aacb6` =
`v1.1.49`), built and deployed commit `90cbc11`, tagged
`deployed/v1.1.49-lane-plan-20261101-1954`, verified to resolve to that commit.
Upstream `v1.1.42..v1.1.49` was **50 commits**; ours is **127 commits, 42 files, +5925/-325**.

**Procedure that worked, and is now the template:** base on the verified-good tree, merge
upstream INTO it (`git merge v1.1.49`), so our side is the base and theirs is the incoming
side. 41 of 42 files merged without a textual conflict; the 5-file collision surface above
was resolved from the token stream, not the compiler.

### Gates

| gate | result |
|---|---|
| `tools/patch_invariants.py` (re-derived first, see below) | **PASS 15/15** |
| per-row loss-shape audit, all 41 owned files | **0 markers fell** — no `R(i)`/`per_row`/`rows_`/`& 7`/`ch` count decreased |
| build | clean, `build\maya.exe` relinked |
| boot | plan **`0 1 2 3 4 5 6 7 8`**, `retimes 0` — planner and all-CPU plan survived |
| n=4 smoke | **4/4 answered, 258 tokens, 12.7 s, rotor 59, faults 0** — no wedge |
| serving | real request → 14 tokens, reply `ready`; engine left running |

### The invariant manifest had to be re-derived BEFORE the merge (step 1, not gate 5)

Its header described a merge onto `cd52717`; run unmodified against a v1.1.49 base it would
have **failed closed on a stale expectation and read as a merge fault**. Every entry was
re-checked against `0f50931` first (11/11 held there), so a failure during the merge is a
loss and not a stale reference. **Four entries were added for the lane plan (`d098fa7`)**,
which no earlier entry covered because the manifest predates the planner — including
`const int waves = (kk + waves_of - 1) / waves_of`, the units fix itself. A merge that
drops it silently reverts the planner to serial pricing and voids every timing claim made
against it.

### Honesty line

The **+6.8 / +11.8 / +23.9% at n=1/2/4 steady were measured on the PRE-merge tree.** On this
build they are **inherited, not re-verified**. The smoke test is not a substitute — it shows
the plan and no wedge, not a rate — and a paired re-measure is its own job.

## 2026-11-02 — v1.1.50 merged, deployed and serving

**Branch `feat/v1150-from-good`.** Merge `6c786a7` (parents `d7e15fb` + `10e69a0` = `v1.1.50`),
built and deployed commit `6c786a7`, tagged **`deployed/v1.1.50-lane-plan-20261102-1147`**,
verified to resolve to that commit. Series: **138 commits, `v1.1.50..HEAD`, 44 files,
+6,289/-325.** Upstream `v1.1.49..v1.1.50` was 32 commits, 51 files, +3,284/-165.

**Supersedes `v1.1.49` as the upstream base, and `d7e15fb` as the reference tree.** The
backup tag for this merge is `backup-pre-v1150-20261102` -> `6e92ebe` -> `d7e15fb`.

### Conflict surface, and the one decision that mattered

Six files collide (`glm_fast.hpp`, `maya.py`, `serve/server.py`, `glm_fast_path.cu`,
`glm_prefill.cu`, `glm_fast.cu`); three conflicted textually. Both sides appended a
parameter to **`moe_route()`**:

```
ours:   int win_rows = 0, int miss_to_cpu = 0
theirs: int win_rows = 0, bool resident_ok = true
```

**A union would have compiled and been wrong**: the surviving call site passes one
argument, `bool` binds into the `int miss_to_cpu` slot, so `miss_to_cpu` would have become
`(il != mtp_il_)` while `resident_ok` kept its default. Both parameters are kept, in that
order, in the declaration, the definition and the single call site:
`..., int miss_to_cpu = 0, bool resident_ok = true)` / `win_rows, miss_to_cpu(), il != mtp_il_`.

### Upstream's overlapping work, judged by what it does

- **`29c075f` prefill CPU gate** — a hardcoded `cpu_p_ms > 2.0 * cpu_c_ms` became
  `STRATA_GLM_PREFILL_CPU_GATE` (default 2.0, so unchanged unless set). It is the **prompt**
  path; our `lane_plan` is decode-only and shares no env with it. Taken as-is; nothing of
  ours is reinterpreted. **If prefill is ever reopened these two must be reconciled, not
  stacked** — they are the same decision at different stages.
- **`86bb00b` resident-expert cap** — `STRATA_GLM_RESIDENT_TOPK` / `_SKIP_W` keep the top-k
  heaviest experts of a route, renormalise, and drop the rest to a null plan entry (no
  fetch, no CPU work). **0 = all, and our config sets neither**, so `MAYA_GLM_EXPERT_USED=7`
  is untouched. It is a Turbo preset knob that *would* fight our 7 if enabled: decide
  deliberately, never by inheriting a preset.
- **`9b2a07e` keep-every-expert** — Maya-XS presets; our model is GLM-5.3-Flash, inert.
- **`24e2101` thinking budget 8192** — written by *setup* into a Maya-XS config. The server
  default stays 0 and our config carries no `thinking_budget`, so the engine keeps 4096.
- **`a6a4fc5` normalise line endings** — both sides' blobs were already LF; noise-free here.

### Gates on the merged build

`patch_invariants` **PASS 15/15** (run on the pre-merge tree first, so a failure would mean
a loss); per-row audit across all 44 files, **0 markers fell**; build clean; boot plan
`0 1 2 3 4 5 6 7 8`; retime announced `OFF`; **n=4 smoke 4/4, 266 tokens, rotor 31,
retimes 0, faults 0**; real request served.

### One claim retracted

`1fb69f5` adds `HH:MM:SS` before each `[maya]` line. The code is present and wired
(`StampedStdout` then `ServerLogTee`), **but no timestamps appear in our log** — so the
"timestamps fix last night's blind spot" benefit is **not** delivered yet and must not be
quoted. The log rotation did preserve the load-window log, so that evidence survived.

## 2026-11-03 — v1.1.52 merged: eleven files collide, zero conflict

**Branch `feat/v1152-from-good`.** Merge `c367e85` (parents `615e36b` +
`3c935e4` = `v1.1.52`), built and deployed commit `c367e85`, tagged
**`deployed/v1.1.52-lane-plan-20261103-1605`**, verified to resolve to that commit.
Series: **140 commits, `v1.1.52..c367e85`, 44 files, +6384/-325.** Upstream
`v1.1.50..v1.1.52` was **383 commits** (v1.1.51 mostly DeepSeek/dashboard, v1.1.52 mostly GLM).

**Supersedes `v1.1.50` as the upstream base, and `615e36b` as the reference tree.** The
backup tag for this merge is `backup-pre-v1152-20261103` -> `615e36b`.

### Conflict surface, and why there were no textual conflicts

11 of our files collide with upstream's 383-commit span (vs 6 last time). Every one
auto-merged; the merge made no manual resolution at all:

`glm_model.hpp`, `glm_fast.hpp`, `maya.py`, `serve/server.py`, `glm_fast_path.cu`,
`glm_fast_state.hpp`, `glm_model.cu`, `glm_window.cu`, `glm_fast.cu`,
`generate.cpp`, `CMakeLists.txt`

Notably `glm_fast_path.cu` — where v1.1.52 lands its CUDA-graph DSA feed-forward —
merged cleanly: upstream's graph code sits in regions ours does not touch. The general
lesson from v1.1.49 holds (seam extraction over in-place edits), and `glm_rows.cu`,
our largest change, collides with nothing for the third merge running.

### Upstream's overlapping work, judged by what it does

- **CUDA-graph DSA feed-forward, default ON** — `STRATA_GLM_DSA_FFN_GRAPH` unset means
  on (`glm_fast_path.cu:6362`). Taken as-is. Upstream claims +8-9% decode with identical
  greedy answers; on this build that rate is **inherited, not re-verified** (same honesty
  rule as the lane plan). Our KDA graphs stay separately opt-in
  (`STRATA_GLM_KDA_GRAPH`, still unset here).
- **IQ1_S rows decode in vectors (#695)** — same-results per upstream; taken. No
  interaction with our paths beyond sharing the file.
- **`MAYA_GLM_ROW_HALVES=1`, `MAYA_GLM_DEFER_CPU=2`** — opt-in, off here. Untouched.
- **Setup keeps host/key/CORS/MCP on re-setup (#720)** — taken; our `serve/server.py`
  env print re-verified present after the merge.
- **`MAYA_GLM_MULTI_CKPTS=1` (#714)** — opt-in per-conversation checkpoints; our
  `--parallel 4` runs without it, unchanged.
- **GLM engine watchdog fix** (short back-to-back requests no longer trip "no progress")
  — taken; it guards the path our loop-guard watches, without touching it.
- **AMD Windows fixes, DeepSeek/MiniMax/dashboards/i18n** — inert for this box.

### Gates on the merged build

`patch_invariants` **PASS 15/15** on the pre-merge tree *and* on the merged tree (a
post-merge failure would have meant a loss, not a stale expectation); per-row audit
across all 25 owned source files, **0 markers fell** (606 → 615, five patterns);
`moe_route` dual params verified at the declaration (`int miss_to_cpu = 0, bool
resident_ok = true`) *and* the single call site (`win_rows, miss_to_cpu(), il !=
mtp_il_`); env print verified; build clean (`[152/152]`, warnings only); boot plan
**`0 1 2 3 4 5 6 7 8`** (CPU takes experts 0–8); retime announced **OFF**;
**n=4 smoke 4/4**; real request 83 tokens, coherent; serving verified; tag pushed sia
only.

### Honesty lines carried forward

The +6.8 / +11.8 / +23.9% are now **three merges old**, still inherited. v1.1.52's +8–9%
decode joins them as inherited-unverified. A paired re-measure is its own job and is
owed for both.

### Shallow-clone trap, found during this merge

This repo is a **shallow clone** (`git rev-parse --is-shallow-repository` → true).
`git rev-list --count A B` (space-separated) counts a **union**, which read 523 for a
range whose true series is 140 — and shallow grafts break the set algebra that would
have caught it. **Count series with `A..B` range syntax** (`v1.1.52..c367e85` = 140),
and treat any space-separated count as suspect.

## 2026-11-05 - v1.1.53 merged, deployed and serving

**Branch `feat/v1153-from-good`.** Merge commit **`d3ea1f3`** (parents = our `a06ad60` +
`v1.1.53`). Upstream `v1.1.52..v1.1.53` is **65 commits, 108 files, +10,461/-579** - the largest
release we have crossed, mostly new backends (`src/metal/`, `mm3` serve, DS4 tests, `tools/agent_bench.py`).

### Gates, in the order the procedure requires them

| gate | result |
|---|---|
| `tools/patch_invariants.py` **on OUR tree, before the merge** | **PASS 15/15** (so a post-merge failure is a loss, not a stale expectation) |
| merge conflicts | **none** - upstream did not touch our lines |
| `tools/patch_invariants.py` after the merge | **PASS 15/15** |
| our additions present after the merge | `orphan_think` x9, budget WARNING, `STEP REFUSED`, `svc.cfg` |
| per-row loss-shape audit (6 markers over src/include/serve) | **PASS** - none fell; `rows_` 946->987, `& 7` 80->82, `R(i)`/`per_row`/`rows_per_row_`/`row_base_` unchanged |
| build | clean, `build\maya.exe` relinked |
| boot | `MAYA_GLM_ORPHAN_THINK=1` live, CPU retime OFF as configured, `git describe` = `v1.1.53-146-gd3ea1f3` |
| serving | real request -> `ready` |
| n=4 smoke (4 rows, temp 1.0, 200 tokens) | **4/4 coherent, guard detected 0** |

The merge was clean enough that no three-way judgement was needed; the only manual step was the
documentation. Series: **211 commits `v1.1.52..HEAD`**.

### What this release does not contain

The `in_think` channel-tagging fix (the no-tag variant of the reasoning-leak, see
`FINDINGS_thinking_budget_2026-11-04.md`). The parser-side orphan split is in, gated by
`MAYA_GLM_ORPHAN_THINK=1`; the engine-side tag is still open.

## 2026-11-06 - v1.1.55 merged, deployed and serving

**Branch `feat/v1155-from-good`.** Merge commit **`d8e5fbc`**. Upstream `v1.1.53..v1.1.55` (two
releases: 1.1.54 and 1.1.55) is **52 commits, 69 files, +3,201/-601**. Series `v1.1.53..HEAD`: **207
commits**. Tag **`deployed/v1.1.55-20261106-0015`**, branch and tag pushed to `sia`.

### Gates, in the procedure's order

| gate | result |
|---|---|
| `tools/patch_invariants.py` on OUR tree **before** the merge | **PASS 15/15** |
| merge conflicts | **none** |
| invariants after the merge | **PASS 15/15** |
| our additions present | `orphan_think` x9, budget WARNING, `STEP REFUSED`, `_note_replay` x3, `MAYA_LOG_REASONING_REPLAY` x4 |
| loss audit (6 markers over src/include/serve) | **PASS** - none fell (`rows_` 987, `& 7` 82, `R(i)` 76, `per_row` 165, `rows_per_row_` 21, `row_base_` 31 all unchanged) |
| build | clean, `maya.exe` relinked |
| boot | listener in 85 s, settings line unchanged |
| serving | real request -> `ready` |
| n=4 smoke (4 rows, 200 tokens, low effort) | **4/4 coherent, guard detected 0** |

**Overlap surface was 9 files** (`.gitignore`, `CMakeLists.txt`, `docs/SETTINGS.md`, `maya.py`,
`serve/frontend.py`, `serve/server.py`, `src/kernels/cuda/glm_batch.cu`, `glm_fast.cu`,
`src/program/generate.cpp`). Every one auto-merged; no three-way judgement was needed this time.

### Upstream's INT8 latent-cache default does NOT apply here

v1.1.55 makes "the INT8 latent cache" the default ("quality within FP16's spread, no loops"). This
pack runs **`--kv q4_0`** from its own config - our prompt-cache work - and the boot line confirms
both the mode and the layout (`--kv q4_0: the cache records q4_0 of the ROTATED rows`; `latent cache
q4_0 (288 bytes a position and layer)`). The merge did not change it. Their claim covers their INT8
branch, not our q4_0 branch; if that setting is ever removed, this is the paragraph to revisit.

The rest of 1.1.54/1.1.55 is Qwen3.8 serve work (not our path), Codex/Responses API fixes, and
default flips from their opt-in review (`MAYA_PROMPT_MEMO`, `DS4_AI_RUN`, `DS4_AR2`).

## 2026-11-06 - v1.1.56 merged, deployed and serving

**Branch `feat/v1156-from-good`.** Merge commit **`3342116`**. Upstream `v1.1.55..v1.1.56` is
**36 commits, 56 files, +5,823/-442**, and it is a GLM release. Tag
**`deployed/v1.1.56-20261106-1527`**, branch and tag pushed to `sia`.

### Gates

| gate | result |
|---|---|
| invariants on OUR tree **before** the merge | **PASS 15/15** |
| merge conflicts | **NONE**, despite a **24-file overlap** (vs 9 for v1.1.55) |
| invariants after the merge | **PASS 15/15** |
| loss audit (6 markers) | **PASS** - `rows_` 987 -> 995 (upstream added rows code), `& 7` 82 -> 83, the rest unchanged |
| our additions | `orphan_think` x9, budget WARNING, `STEP REFUSED`, `_note_replay` x3 |
| build | clean |
| boot | settings line unchanged; `CPU lane retime OFF (STRATA_GLM_CPU_RETIME=0)` still prints |
| serving | `ready` |
| n=4 smoke | **4/4 coherent, guard 0** |

The 24 overlapping files included our hot path - `glm_fast_path.cu`, `glm_fast_state.hpp`,
`glm_model.cu`, `glm_prefill.cu`, `glm_window.cu`, `kernels/cuda/glm_fast.cu`, `generate.cpp`,
`serve/server.py` - and every one auto-merged.

### What is in it, and what touches us

- **#860 speculative decode**: a draft with no finite logit counts as a valid token on every
  drafting path; a two-GPU split's spec decode is back to v1.1.51's behaviour; and
  `MAYA_GLM_NO_SPEC`/`NO_MTP`/`NO_PREFILL` now treat `0`, `false`, `off`, `no`, `nothing` as OFF.
  The last one matters generally: a plain `getenv() != nullptr` took `NO_SPEC=0` to mean "no
  drafting". We set none of those flags, and `CPU_RETIME=0` still reads OFF.
- **#845 no seed draws a new seed** (it was always seed 0). **This is a behavioural change for us** -
  see the determinism note below.
- **#840 GLM slots: try the next slot and the checkpoint when a parked conversation fails to load** -
  aimed straight at the "refused ... foreign identity, incomplete, or its checksum failed" lines
  this box was logging after the v1.1.55 upgrade.
- **#861 iq_pack keeps the GLM5-next selection bias `exp_probs_b.bias` F32** instead of rounding it
  to BF16 - pack tooling, not serving.
- The rest is DeepSeek EXL3/Pascal/AMD work and file attachments in the web chat (PDF/DOCX/XLSX...).

### Determinism after #845 - do not compare past no-seed runs with these

Measured on this box (temperature 1.0, no seed, then `seed=1234`):

```
no seed:      varies between runs   (intended: #845 draws a new seed)
seed=1234:    identical in 3 of 4 pairs; one pair differed (red, blue, green / Red, green, blue)
```

So an explicit seed is **mostly** but not fully reproducible at temperature > 0 with speculative
decoding on - the draft-acceptance path consumes RNG state differently. **`temperature: 0` remains
the exact mode** and is what the fork's own harnesses use. Any benchmark or eval that sent no seed
at temperature > 0 is *not* reproducible against this build, and that is upstream's change, not the
fork's.

## 2026-11-10 - v1.1.57 merged, deployed and serving (the q4_0 decision)

**Branch `feat/v1157-from-good`.** Merge commit **`2e00a26`**. Upstream `v1.1.56..v1.1.57` is 8
commits but **456 files, +76,605/-1,166** - it adds **GLM-5.3 (744B) on the GLM engine** (`glm_dsa`),
replaces state allocation with `glm_state_carve`/`GlmStateLayout`, and reworks setup/API/docs. Tag
**`deployed/v1.1.57-20261110-0809`**, branch and tag pushed to `sia`.

### Gates

| gate | result |
|---|---|
| invariants on OUR tree **before** the merge | **PASS 15/15** |
| files upstream touched / ours / **overlap** | 456 / 140 / **42** |
| **conflicts** | **8 files, 16 hunks** - all resolved by intent, none by taking a side wholesale |
| invariants after the merge | **PASS 15/15** |
| loss audit (6 markers) | **PASS** - none fell |
| build | clean |
| boot + serving | listener 75 s, `ready` |
| n=4 smoke | **4/4 coherent, guard 0** |

### q4_0 latent cache RETIRED - upstream's INT8 adopted

v1.1.57's `GlmStateMode` is **`SlowF32` / `Fp16` / `Int8` only** - there is no q4_0, and the callers
already thread the new layout (`m->lay_`, `c.kda_S = L.kda_S`). Keeping our tri-state carve would
have meant re-implementing it inside a brand-new upstream module. **Decision (user, 2026-11-10): adopt
upstream's INT8 and revisit other KV quants later.**

Measured after the build, not assumed:

| | q4_0 (v1.1.56) | INT8 (v1.1.57) |
|---|---|---|
| latent record | 288 B a position and layer | **544 B** |
| latent pinned RAM | 1.66 GB | **3.14 GB** |
| KV streaming in VRAM | 0.10 GB | **0.20 GB** |

The config's `--kv q4_0` was changed to `--kv int8`. Our stage-A/B q4_0 kernels remain in the tree as
dead code; nothing selects them.

### Two bridges the merge needed (both are OURS, both documented in the code)

1. **A BF16/F32 shared-expert down on glm5-next.** Upstream's fused launch refuses those types
   (`ext_shared_down_ok` in `glmd_mv_ext.cuh`: they read a FLOAT input and the fused path only has the
   q8_1 activation), and serves them through `glmdsa::mv`'s fallback - which needs the glm-dsa ext.
   This build's glm5-next path has **no ext** (`FastState::dsa` unset), so every request failed with
   `glm fast: shared expert down` until bridged. Our pack stores the shared down as **BF16 (type 30)**,
   so this was not an edge case - it was all traffic. Bridge: dequantise the q8_1 activation once into
   `F->hq` (float, large enough; the fused gate/up overwrites it after this down on the same stream)
   and run the down through `gf::mv`. Uses upstream's own `glmdsa::q8_1_to_f32`.
2. **`slot_load`'s blob path.** After the merge it carried the *file-handle* form of the NextN trailer
   call (`mtp_slot_get(f, ...)`) inside the function that reads the slot into a blob - `f` undefined.
   It now builds an `istringstream` over the blob, matching upstream's own blob path.

Our **warm-up batch clamp** (clamp the batch to `F->stage.size()`, else a fixed batch of 8 reads
`stage[7]` past the end when `MAYA_GLM_EXPERT_USED` is below 8) was re-applied over upstream's new
file-order warm-up loop, with `max(1, ...)` so an empty stage cannot spin.

### Conflict resolutions, by file

| file | hunks | resolution |
|---|---|---|
| `serve/frontend.py` | 3 | both: our `_note_replay` + their `_clear_thinking`; signature takes `deepseek` and `glm_dsa`; parser takes `orphan_think` **and** `unions` |
| `serve/server.py` | 1 | both: our `orphan_think` gate + their `unions=` |
| `include/strata/kernels/glm_fast.hpp` | 3 | our `int ch` kept with their `x[kMaxEmbd]`; both comments blocks; `moe_route` takes `miss_to_cpu` **and** `norm_mode` |
| `src/kernels/cuda/glm_fast.cu` | 2 | their guard + signature union; both assignments (`miss_to_cpu`, `norm_mode`, `rand_seed`) |
| `src/core/glm_fast_path.cu` | 4 | theirs for the layout/route work, plus the two bridges above |
| `src/core/glm_model.cu`, `glm_multi.cu`, `glm_prefill.cu` | 1 each | theirs (the layout refactor; q4_0 retired) |

**The lesson worth carrying:** the hard conflicts were not textual, they were *assumption* conflicts -
upstream's new code presumes a dsa ext exists and that packs store the shared down as a quant type.
Neither held here. A merge that only resolves markers would have built and then failed every request.

## 2026-11-11 - v1.1.59 merged, deployed, ablation enabled

**Branch `feat/v1159-from-good`.** Merge commit **`5becfc2`**. Upstream `v1.1.57..v1.1.59` is
5 commits but **413 files, +7,419/-3,013**: GLM-5.3-Maya-XS (the 744B quant), **refusal ablation for
GLM-5.3-Flash**, **fair slices under `--parallel`**, and a repo-wide **`STRATA_*` -> `MAYA_*` rename**.
Tag **`deployed/v1.1.59-20261111-1118`**, branch and tag pushed to `sia`.

### Gates

| gate | result |
|---|---|
| invariants before / after | **PASS 15/15 / 15/15** |
| conflicts | **10 files, 20 hunks** - resolved by intent |
| loss audit | **PASS** |
| build | clean |
| boot + serving | `ready` |
| n=4 smoke, ablation **off** | **4/4**, guard 0 |
| n=4 smoke, ablation **on** | **4/4**, guard 0 |

### The rotor's first real conflict - and the one that mattered

`glm_rows.cu` conflicted for the first time (4 hunks). One was silent and serious: upstream's loop
header is `for (int il = l0_; il < l1_; ++il)` where ours is `for (int il = ilf; il < ilt; ++il)`.
The rotor calls `rows_layers()` for **one layer at a time** (`il, il + 1`); taking upstream's bounds
would have run all 45 layers on every per-row call. Restored `ilf/ilt` with a comment naming why.

### The `MAYA_` rename (a real correctness point, not cosmetics)

Upstream renamed every setting `STRATA_*` -> `MAYA_*`. `include/strata/platform/env_alias.hpp` maps
any `STRATA_X` with no `MAYA_X` to `MAYA_X` **at process start**, and the code reads `MAYA_X`. Our
conflict blocks had restored the old names, so each affected file was renamed here - a
`STRATA_`-only reader would have silently ignored a `MAYA_` setting.

### Duplicated and drifted symbols the merge produced

- `lat8_rec_bytes` was defined in **both** `glm_lat_types.hpp` (new, upstream) and `glm_fast.hpp`
  (ours) -> linker error; `glm_lat_types.hpp` is the single definition now.
- `mla_attend` / `mla_attend_rows` / `mla_attend_batch` had a **header saying `int lat_q8` and a
  definition saying `bool`** -> unresolved externals; both sides now `bool`, matching upstream.
- `mla_out` kept our `int lat_q8` parameter and its q4_0 rotation branch in the definition while the
  header had already dropped the parameter -> removed, matching upstream (q4_0 retired in v1.1.57).
- `model.kv_name()` (our tri-state helper) no longer exists; `generate.cpp` now uses upstream's
  `model.kv_int8() ? "int8" : "fp16"`.

### Refusal ablation - ENABLED (user request)

- File: **`G:\AI\maya-eval\models\refusal-directions\GLM-5.3-Flash-refusal-direction.gguf`**
  (16.5 KB; `maya.ablate.model = GLM-5.3-Flash`, `fit_layer = 31`, `scale = 1`), from
  `peasantsmith/GLM-5.3-Flash-Maya-GGUF`.
- Config: `"ablate": {"file": ..., "default": true}` **and** the engine args
  `--ablate <file> --ablate-default off`. Both are needed: the engine loads the direction once
  (`--ablate`), keeps it off (`--ablate-default off`), and the **server** decides per request from the
  request's `ablate` field or `cfg["ablate"]["default"]`.
- A hand-written config does NOT get the flags automatically - `models.apply()` (the Models page's
  save path) is what adds them, so a config edited by hand needs them in `args`.
- Verified: the engine logs `glm: abliteration: ... direction.1, scale 1, projected out of every
  layer's attention and FFN writes`; the server reports `settings.ablate: true`; requests omitting
  `ablate`, `ablate: true` and `ablate: false` all answer; and the 4-row smoke is clean with the
  projection on (the rotor's per-conversation runs go through `abl_seqs`).
- Per-request off remains available always: `"ablate": false` on any API.

### `MAYA_FAIR_SLICE_S` (new in 1.1.59) - and how it relates to our earlier work

From `serve/server.py`: *"a request that has been answering this long while others wait gives the
engine to the next one and queues again behind them (0, the default: it keeps the engine to the end)"*.
v1.1.59 also fixes it so one waiter pauses **one** running answer, not every one.

**It does not replace what we were doing - it is a different phase.**
- `MAYA_GLM_PREFILL_INTERLEAVE` / `PREFILL_GAPS` (our earlier work): other requests **decode while a
  prompt is being read** - prefill-time fairness, so one long prompt does not stall everyone.
- `MAYA_FAIR_SLICE_S`: a **long answer** yields the engine to a waiting request after N seconds -
  decode-time fairness, so one long generation does not starve everyone.
- The rotor makes two rows run **at the same time** at all; the lane plan prices where their experts
  run. Neither of those is a fairness knob.

Both fairness settings default to 0/off, so nothing changed by this merge.

## 2026-11-11 - the prompt path was gated off for GLM-5.3-Flash i-quants (cold prompts 44 tok/s)

**Symptom.** A cold 82,305-token prompt was read at **24 tok/s** (and 14k prompts at **44-48 tok/s**),
while the same box had done **2,370 tok/s** on 2026-11-10 07:54.  Warm conversations hid it: with a
slot hit only the new tokens are read, so every long conversation that was already cached felt fine.

**What it was NOT** (each ruled out by measurement, not argument):
- **Not the KV latent type.** `--kv int8` 44 tok/s, `--kv q4_0` 48 tok/s, `--kv fp16` no prompt path
  either.  Restoring q4_0 (dropped in v1.1.59) would not have helped.
- **Not the new fairness knobs.** With `MAYA_GLM_PREFILL_INTERLEAVE` and ablation both off, a cold
  13,910-token read was still 44 tok/s.  (`PREFILL_INTERLEAVE` is inert without a prompt path.)
- **Not the repack on its own.**  The repack is a necessary ingredient but the refusal was a policy
  gate, not a missing tensor.
- **Not `deq_ok`'s empty-slot test.**  Patching it to accept `f32`-stored weights changed nothing
  (reverted).

**What it was.** `f16_covers()` in `src/core/glm_prefill_dsa.cu`:

```cpp
if (!dsa) return false;                      // Flash: Maya's set, unchanged (no i-quant)
if (type == 16 || 17 || 18 || 21 || 22 || 29) return true;   // glm-dsa only
```

GLM-5.3-Flash (arch != GlmDsa) therefore treated **any i-quant dense weight** as uncovered.  The
v3q repack is built from `GLM-5.3-Flash-UD-IQ3_XXS`, so a handful of layers carry **IQ4_XS (type 21)**
attention/shared-expert tensors - and `prompt path: has no prompt path (not covered: blk.17.attn_k
(type 21) ...)` disabled the batched read for the whole model.  The older `pack` dequantised those
tensors to BF16, so it never hit the gate (that is why the pre-repack logs show
`chunks of 65536 tokens` and 2,370 tok/s).

**The change** (`glm_prefill_dsa.cu`, +4/-2): the i-quant list is honoured before the `dsa` gate.
`dequant_f16` already routes i-quants to `iq_dequant_f16`; the gate was a policy, not a capability.

**Measured after** (v1.1.59, int8, ablation on, interleave on):

| | before | after |
|---|---|---|
| cold 14k prompt | 44 tok/s (314 s) | **1,822-1,964 tok/s (~7 s)** |
| cold 82k prompt | 24 tok/s (~57 min) | not re-run (see below) |

**Correctness** (the reason this could not be judged on speed alone - the last fast pack was the
KNOWN-BAD one): needle-in-a-haystack 11,837-token prompt retrieved `ZQ-4471`; third-largest planet
`Uranus`; Great Red Spot `Jupiter`; 17x23 = `391`; n=4 concurrent smoke 4/4, guard 0.

**Worth telling upstream:** the gate reads as deliberate ("Flash: Maya's set, unchanged"), so this is
a local divergence - but a pack that is otherwise stock-validated silently loses ~50x on cold prompts.
Either the gate should be a setting or Flash should get the same i-quant coverage glm-dsa has.
