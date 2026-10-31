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

## Upstream base — one base for every recent release

`e16dec85a` is the merge base of our tree with **every** release from v1.1.39 through
v1.1.42. So "our changes" is one well-defined series regardless of which release we
target: **96 commits, `e16dec85a..cd52717`, 39 files, +4776/-345.**

## The reference tree

**`cd52717` is the verified working `n=2` de-fused rotor** — 41-47 rotor entries per
boot, ~24 tok/s, six coherent replies, 3/3 clean boots. Tagged
`backup-pre-20261030-0835`. Every quality or liveness claim is referenced against it.
It is the base for `feat/v1142-from-good`.

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

- `series-cd52717/` — the 96 commits as `format-patch` files, re-appliable with `git am`
- `by-file/` — one squashed diff per touched file, to review or re-apply one subsystem
- `reference/CONFLICT-SURFACE.txt`, `CONFLICT-FORECAST-*.txt` — the tables below

## The conflict surface — the number that decides how we organise

**14 of our 39 touched files** are also touched by upstream across v1.1.39 -> v1.1.42.
The **v1.1.42 delta alone reaches only 5**: `glm_fast_path.cu`, `glm_prefill.cu`,
`generate.cpp`, `serve/server.py`, `CMakeLists.txt`.

| File | upstream lines | our changed lines |
|---|---|---|
| `src/program/generate.cpp` | 8804 | 278 |
| `src/core/glm_fast_path.cu` | 7562 | 339 |
| `src/kernels/cuda/glm_fast.cu` | 7006 | 411 |
| `src/core/glm_prefill.cu` | 4899 | 64 |

**`src/core/glm_rows.cu` — our LARGEST change at 902 lines — is not in the conflict set.**
Upstream has not touched it in four releases. Our biggest change is our cheapest to
carry, because it lives in a file that is ours. **That is the case for extracting more
logic behind seams rather than editing upstream files in place.**

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
5. Only then tag and deploy.

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
