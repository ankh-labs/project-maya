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

## Upstream base — CURRENT (2026-11-02): v1.1.49

**Supersedes the `e16dec85a` / `cd52717` identity this section used to carry. Read only
this one — a document with two bases is how a merge got started against the wrong tree.**

Our tree is now **upstream `v1.1.49` (`26aacb6`)** with our work merged onto it, on branch
**`feat/v1149-from-good`**. The merge base with upstream was `v1.1.42`; the merge commit is
`6edde5d`; the built and deployed commit is `90cbc11` (`git describe --tags --match
'v1.1.*'` → **`v1.1.49-127-g90cbc11`**), tagged
**`deployed/v1.1.49-lane-plan-20261101-1954`**.

**"Our changes", as of this merge: 127 commits, `v1.1.49..90cbc11`, 42 files,
+5925/-325.** That is the series the export is generated from and the number to quote.

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
