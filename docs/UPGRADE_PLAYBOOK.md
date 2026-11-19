# Our Maya patches, and the upgrade playbook

**Read this before merging a new upstream release.** It exists because the same four or five
things break or silently vanish on every upgrade, and rediscovering them costs a session each time.
Everything here is verifiable with a command; the expected numbers are measured, not guessed.

Last updated: 2026-11-19 (tree `v1.1.66-170-gf281841`, upstream at **v1.1.68**).

---

## 1. What we run

| | |
|---|---|
| engine | our fork of `mw00/project-maya`, branch `feat/vNNNN-from-good`, deployed tag `deployed/<version>-<date>` |
| model | **v3q pack** `G:\AI\dsquant-work\gguf\v3q\pack-repack-20261027` (UD-IQ3_XXS-derived, experts native) |
| service | NSSM `Maya-8080` (LocalSystem), `tools/install_maya_service.ps1` |
| config | `maya-v3q-q4-par4.json` (gitignored - **this doc is its record**) |
| expected speed | prefill **1,800-2,000 tok/s**, decode **20-25 tok/s** |

---

## 2. Our patches, what each buys, and how to verify it survived

Every row here has been silently lost at least once. If a verification fails, the patch was dropped
in a conflict resolution - do not "fix" the symptom, re-apply the patch.

| # | patch | files | what it buys | verify | symptom when lost |
|---|---|---|---|---|---|
| 1 | **i-quant prompt path** - the i-quant list (`16 17 18 21 22 29`) is honoured *before* the `if (!dsa) return false` gate in `f16_covers()` | `src/core/glm_prefill_dsa.cu` (4 lines) | the batched prompt path for a pack with any i-quant dense weight: **44 tok/s -> ~1,900 tok/s** | log line `glm prefill: CUDA0 chunks of 65536 tokens`; cold 8k prompt >=1,500 tok/s | `has no prompt path (not covered: ...)`; every prompt at decode speed |
| 2 | **host-ordered route seq** (`hseq`, `need_seq`, `moe_wseq`, `per_row_bufs`) | `glm_fast_path.cu`, `glm_fast.cu`, `glm_fast_state.hpp`, `glm_model.cu` | two concurrent rows cannot publish out of order; deterministic answers | `git grep -l hseq` -> 3 files; n=4 smoke 4/4 | answers swap between rows; wrong text under concurrency |
| 3 | **the rotor** - `rows_layers(rw + r, 1, err, il, il + 1)` called **one layer per row**, `ilf/ilt` loop bounds, `admit_row0` | `glm_rows.cu` (+799), `glm_rows_state.hpp`, `glm_batch.cu` | two conversations decode in one pass (~4 rows / 4 slots) | invariant `if (!rows_layers(rw + r, 1, err, il, il + 1))` present; n=4 smoke | upstream's `l0_..l1_` runs all 45 layers per row call - 45x work, wrong state |
| 4 | **mover staging clamp** `std::min<size_t>(8, F->stage.size())` | `glm_fast_path.cu` | prevents a host-side OOB write at `MAYA_GLM_EXPERT_USED=7` (we run 7) | invariant `std::min<size_t>(8, F->stage.size())` | silent heap overrun; compute-sanitizer cannot see it (host vector) |
| 5 | **serve-side thinking handling** - `MAYA_GLM_THINK_TOKEN_ONLY`, `MAYA_GLM_ORPHAN_THINK`, `_note_replay`, `STEP REFUSED`, budget WARNING | `serve/server.py`, `serve/frontend.py`, `serve/test_orphan_think.py` | reasoning separated from the answer; orphaned `<｜end▁of▁thinking｜>` handled | `git grep -l orphan_think` in `serve/`; `python -m serve.test_orphan_think` | reasoning leaks into answers; truncated answers |
| 6 | **merge gate** `tools/patch_invariants.{py,txt}` (15 invariants, fail-closed) | tools/ | mechanical proof nothing above was dropped | `python tools/patch_invariants.py` -> `PASS: 15 invariants hold` | you find out days later, from wrong answers |
| 7 | **service installer** `tools/install_maya_service.ps1` | tools/ | NSSM service that survives reboots; the CUDA-on-PATH trap handled | `sc query Maya-8080` -> RUNNING | Maya dies on reboot; engine exits at start with no log reason |
| 8 | **q4_0 KV, stage B** (`LatType::q4_0`, rotated records, `fwht512`, `lat4_rec_bytes`) | `glm_lat_types.hpp`, `glm_lat4.cuh`, `kv_q4.*`, `glm_batch.cu` branches | q4_0 latents = 288 B/pos instead of int8's 544 | **BROKEN - see §3** | asking for q4_0 silently runs fp16 |

---

## 3. The KV mode trap (this is the one that keeps biting)

> **CORRECTED 2026-11-19 - it is a DROPPED PATCH, not an upstream limitation.** q4_0 was never
> "silently fp16" as a property of the codebase: our selector was removed by **my v1.1.59 merge
> resolution**, which took upstream's `bool lat_q8_` and deleted `lat_type_`. The feature still
> exists and still works when the selector is restored. This is the exact content of
> `homelab/v3q-support:src/core/glm_model.cu`:
>
> ```c
> const char* kv_int_env = getenv("MAYA_GLM_KV_INT8");          // (STRATA_ on that branch;
> const bool want_int8 = kv_int_env && atoi(kv_int_env) != 0;   //  env_alias maps it at start)
> const char* kv_q4_env  = getenv("MAYA_GLM_KV_Q4");
> const bool want_q4    = kv_q4_env && atoi(kv_q4_env) != 0;
> lat_type_ = want_q4 ? LatType::q4_0 : want_int8 ? LatType::int8 : LatType::fp16;
> ```
>
> **Restoring it is a tri-state job, not a one-liner:**
> 1. `glm_model.hpp`: `int lat_q8_` (0 fp16 / 1 int8 / 2 q4_0), not `bool`; accessors that state the
>    type (`kv_int8()`, `kv_q4()`) for the printf/slot identity.
> 2. `generate.cpp`: `--kv q4_0` must set `MAYA_GLM_KV_Q4=1` **and** clear `MAYA_GLM_KV_INT8`
>    (today `-kv q4_0` only calls `qsa_set_kv_q4`, the Qwen path).
> 3. `glm_model.cu`: the selection above, plus the state-layout mode (q4_0 records are 288 B/pos).
> 4. The `int lat_q8` signatures we changed to `bool` in the v1.1.59/1.1.66 merges
>    (`glm_fast.hpp`, `mla_attend*`, `mla_out`) must go back to the tri-state where the stage-B
>    branch `if (lat_q8 == 2) fwht512_rows_cuda(...)` lives (`glm_batch.cu`).
>
> **Verify:** the boot log must say `latent cache q4_0 (288 bytes a position and layer)` - if it
> says INT8 (544 B) or FP16 (1024 B), the selector is still missing. Then cold bench >=1,500 tok/s
> and decode 20-25 tok/s.

`--kv q4_0` is **not** handled by GLM's engine any more:

```c
// generate.cpp:5433
qsa_set_kv_q4(o.kv == "q4_0");            // the Qwen/qsa path only
// glm_model.cu:3563  - GLM's latent is bool: int8 or fp16, nothing else
lat_q8_ = fast_mode_ && ... MAYA_GLM_KV_INT8 ...;
```

So on GLM, **`--kv q4_0` silently selects FP16**, and FP16 is measurably unreliable on this build
(see §4). Our stage-B q4_0 code is still in the tree but nothing selects it any more.
**To restore q4_0 (the "works great once patched" configuration):** in `glm_model.cu` where the latent
type is chosen, select `LatType::q4_0` when `--kv q4_0` was asked for (carry it through the same
`qsa_kv_q4`-style flag that `generate.cpp` already sets), and confirm the stage-B branches in
`glm_batch.cu` / `glm_lat4.cuh` still compile and run. Verify with the log line naming the latent
type plus a cold prompt bench.

### Measured KV behaviour on v1.1.66 (8 short requests each)

| `--kv` | decode failures | prompt path | notes |
|---|---|---|---|
| **int8** | **0 / 8** | 1,841 tok/s | use this today |
| fp16 | **5 / 8** (also 2/8 with the prompt path on) | ~2,000 tok/s | do not use: non-finite decode at random layers |
| q4_0 | **= fp16 today** | - | needs patch 8 re-wired |

Pairs with `MAYA_GLM_SLOTS=20`, `MAYA_GLM_SLOT_KEEP=1`, `MAYA_GLM_SLOT_MIN=1024` - if the slot
identity changes, every conversation is cold once (that is a 44 tok/s prompt, not a broken engine).

---

## 4. Known open bugs (upstream, not ours)

- **int8 + prompt path + ~97k-token prompt** -> `glm router: prompt layer 7 invalid expert ID or
  non-finite score (315 of 72758 routes)`, deterministic. This is Sphinx's failure. Not the weights
  (the layer's only suspect tensor is already BF16 in the pack), not ablation (vector is unit-norm).
- **fp16 latents** -> `glm router: layer N decode NON-FINITE SCORE` at random layers, ~60 % of
  requests. Do not run fp16.
- **opencode server memory** (CT 915): unbounded SSE subscriber queues / per-session listeners; a
  68-minute run peaked at 4.3 GB. Cap the unit's lifetime (`RuntimeMaxSec`) rather than chase it.

---

## 5. The upgrade playbook (do it in this order)

```powershell
# 0. baseline: the current tree must pass its own gate BEFORE the merge
python tools\patch_invariants.py            # expect: PASS: 15 invariants hold

# 1. branch + merge
git checkout -b feat/v<ver>-from-good
git merge v<ver> --no-edit
git diff --name-only --diff-filter=U        # conflicts

# 2. resolve by INTENT (see §6), then re-check the gate
python tools\patch_invariants.py            # must still be 15/15

# 3. build - the SERVICE MUST BE STOPPED (maya.exe is locked)
sc stop Maya-8080 ; # wait for STATE STOPPED
cmd /c C:\Users\agips\AppData\Local\Temp\opencode\build_maya.bat

# 4. deploy + verify (never skip; a clean build proves nothing)
sc start Maya-8080
#   a) prompt path:  log -> "chunks of 65536 tokens (--prefill"
#   b) cold prompt:  python ...\prefill_bench.py 8000   -> >=1,500 tok/s
#   c) decode:       a 128-token request               -> 20-25 tok/s
#   d) concurrency:  cancel_churn_v4.ps1 -Rows 4        -> 4/4, guard 0
#   e) KV mode:      int8; confirm the latent line, never trust the flag alone
#   f) ablation:     /api/models -> settings.ablate true
#   g) remember:     a bench run straight after a restart reads ~390 tok/s because the
#                    expert cache is cold. Let it warm, or run the bench twice.

# 5. record + publish
#   docs/STATE.md + docs/FORK_LAYOUT.md entry, deployed/v<ver>-<stamp> tag, push to sia
```

**Restart discipline:** `sc stop` then **poll until `STATE : STOPPED`** before `sc start`. Issuing
the start during `STOP_PENDING` silently leaves the service stopped (cost a serving outage
2026-11-17). On the Linux side, `pkill -f "maya.py --config ..."` matches the SSH command line
itself - put it in a script file.

---

## 6. Conflict patterns that recur (and the resolution)

| pattern | example | resolution |
|---|---|---|
| env rename | `STRATA_GLM_*` (ours) vs `MAYA_GLM_*` (theirs) | the code reads `MAYA_*`; `env_alias.hpp` maps the old names at start. Adopt theirs. |
| allocation-order trap | upstream's `F->md.resp = dp;` inside the `wait_diag` hunk | **drop that line**: in our order `dp` is the channel-seq buffer there, `resp` was set earlier from `resp8`. Taking it points the routing response at the wrong memory and still compiles. |
| loop bounds | `for (int il = l0_; il < l1_; ++il)` vs ours `ilf/ilt` | keep **ours**: the rotor calls `rows_layers` one layer at a time; `l0_..l1_` runs all 45. |
| type plumbing | `int lat_q8` (ours, tri-state) vs `bool lat_q8` (theirs) | with q4_0 retired upstream they use bool. Keep the definition and the declaration in step - a header/definition mismatch surfaces as an unresolved external, not a compile error. |
| duplicated symbols | `lat8_rec_bytes` in both `glm_lat_types.hpp` and `glm_fast.hpp`; `mla_attend*` declared `int` but defined `bool` | one definition, matching signatures. |
| our helpers removed | `model.kv_name()` | `model.kv_int8() ? "int8" : "fp16"`. |
| init ordering | their `fast_prefill -> DS4_YOCO_CONE` vs our `[engine env]` print | apply theirs **first**, so our diagnostic prints the key it claims to show. |

---

## 7. Tools we keep

| tool | what it does |
|---|---|
| `tools/patch_invariants.py` + `.txt` | the fail-closed gate; run before and after every merge |
| `tools/install_maya_service.ps1` | NSSM service; sets the CUDA PATH a LocalSystem service lacks |
| `tools/iq_pack.py` | builds v3q packs from a GGUF (`--compat-bf16` for the BF16-read projections) |
| `tools/strata_tokenizer.py`, `mtp_from_gguf.py` | tokenizer export; MTP drafter pack (needed by the Qwen serve path) |
| `prefill_bench.py` (scratch) | cold-prompt benchmark; the number that answers "is the prompt path on?" |
| `cancel_churn_v4.ps1` (scratch) | n-row concurrency smoke + guard check |

Scratch tools live in `C:\Users\agips\AppData\Local\Temp\opencode\`; if they disappear, the methods
in §5 are the reproducible part.
