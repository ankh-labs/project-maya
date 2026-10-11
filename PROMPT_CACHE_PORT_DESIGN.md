# Prompt-cache port: fork → Maya — critique, tiered design, stage plan

Branch `homelab/v3q-support` (5218aeb). Static analysis only; you build and test.
Companion patch: `PROMPT_CACHE_PORT_PROPOSED.patch` (stage 1).

## Status (2026-10-27, rebased to fe090f5)

Committed on the branch: stage 1 hardening, the marker fix (b2cfc87), and stage 2A accounting
(fe090f5 - `PROMPT_CACHE_STAGE2A.patch` is redundant, delete it from the set). Three commits landed
after the first handover: 8d200b4 (parallel 8-stream slot_load - measured no gain; the read was ~15%
of restore, as the analysis said), 1179d7b (the v1.1.31 merge, 44 commits), fe090f5 (2A).

Pending patches, each verified with `git apply --check -C1` against fe090f5 (all earlier patch files
were corrupt as diffs and were hand-applied - these are regenerated as real diffs):

1. `KV_IDENTITY_TRUTHFUL.patch` - `kv_name()` accessor; both `kv=` call sites (identity, INFO
   line) read it; `--kv q4_0` on the GLM path refused loudly (it silently fell to fp16). Parse-time
   `--kv` validation already existed - the gap was GLM-path-only.
2. `PACK_IDENTITY_CONTENT_HASH.patch` - pack identity by content (FNV-1a over `index.txt` + dense.bin
   size/mtime folded in), and the restore-side size+mtime fast path for the `.bin` checksum (`.meta`
   magic bumps to GMAYESS2; old metas refuse loudly and are re-saved). Startup indexing still hashes
   fully. **Correction 3:** the fast path was NOT in the tree before this patch - both call sites hashed
   the whole `.bin`; the earlier write-up recorded it as done, which was wrong.
3. `STATE_SIGNATURE_DIGEST.patch` - **correction 4:** `engine=` in the identity is STRATA_VERSION, the
   static `project(strata VERSION ...)` string, which never moves with a Maya release; the four slots
   survived the v1.1.31 merge because that field is inert, not because `the version string didn't move`.
   Combined with the size-derived pack identity, NOTHING invalidated on a layout-changing rebuild. The
   digest (build-time SHA over the state's meaning sources; see CMakeLists, mirroring the fork's
   `llama_state_signature()`) is therefore the only layout guard there is, and `unknown` falls back to
   the compile stamp (over-strict), never permissive.
4. `GLM_KV_Q4_STAGE_A.patch` (applies after 1) - `LatType` + `glm_lat_types.hpp` (shared by model
   and kernels), truthful `kv=q4_0` identity, parser wiring via `STRATA_GLM_KV_Q4`, arena/streaming
   sizing from `lat_records_` (what is actually stored - INT8 in stage A, so the layout never lies).
   Byte counts at kv_lora 512, 262144 positions x 11 DSA layers (log-measured where marked): fp16
   1024 B/row = 2.95 GB (measured), int8 544 B/row = 1.57 GB (measured), q4_0 288 B/row = 0.83 GB.
   q4_0 saves 47% of the int8 cache (288/544) = 0.74 GB = ~84 slots at 8.85 MB/slot. Stage B (the
   ~15-25 kernel read ternaries + `lat_put`) flips `lat_records_` to q4_0; the digest then invalidates
   stage-A q4_0 slots because glm_batch.cu changes.

   **Stage B basis (correction from the user: the machinery already exists)** - q4_0 and k8v4 are live
   for the QSA path (`kv_q4.hpp`/`kv_q4.cu`, PR #21): `block_q4_0` (18 B: fp16 scale + 16 nibble
   bytes, value = (q - 8) * d) and a Walsh-Hadamard rotation (`fwht256_cuda`) that spreads outlier
   channels before quantizing, with the query rotated identically (scores unchanged: <Hq, Hk> = <q, k>)
   and the attention output rotated back (H is self-inverse). Stage B reuses this rather than inventing a
   parallel format: `glm_lat_types.hpp`'s `lat4_rec_bytes` matches `block_q4_0` exactly (kv_lora
   512 = 16 blocks = 288 B). The latent-specific work is a 512-point rotation (two fwht256 halves), applied
   to the latent at write and to `q_abs` per head - both live in the same 512-dim absorbed space, so the
   score dot product is preserved - and to the gathered context before the `wv_b` output projection. Plain
   q4_0 COMBINES with streaming on the QSA path (`kv_q4.cu` writes the host copy and the resident pages);
   k8v4's "not with --kv-resident" (generate.cpp:4776) is a QSA-path implementation gap, and k8v4 does not
   map to the GLM latent at all: the latent is a single shared record serving both scores and context - there
   is no separate V to hold at q4_0 while K stays exact.

   **Stage-B work plan (exact sites; `glm_lat4.cuh` written, inert until the sweep includes it):**
   - Record helpers: `src/kernels/cuda/glm_lat4.cuh` - `lat4_rec` / `lat4_f` / `lat4_h4` / `lat4_h8`
     (decode, same uint2/uint4 shapes as the `lat8_h4`/`lat8_h8` the ternaries already produce) and
     `lat4_group` / `lat4_store_warp` (store, mirroring kv_q4.cu's tie-safe `q4_group`).
   - Store: `lat_put` gains the q4 arm in BOTH `dsa_prep_kernel`s (glm_batch.cu ~499 and glm_fast.cu ~2928),
     covering lat/lat_host/lat_slots; the rotation is a shared-memory butterfly in the kernel (blockDim 512 =
     kv_lora 512, guarded loudly at load), then `lat4_store_warp` per warp.
   - Read arms: the eight ternaries in glm_batch.cu's attention kernels (~721, 836, 965, 1132, 1334, 1531,
     1674, 1819) gain `lat4_h4`/`lat4_h8`; glm_fast.cu's `lat_any`/`lat8_f` (~2877) likewise.
   - Rotation, fused path: `mla_kernel` (glm_fast.cu ~3092) computes q_abs, ctx AND the wv_b output in one
     kernel's shared memory - rotate `s_qa` after the wk_b GEMV, decode q4 records for the scores, rotate
     `s_ctx` back before the wv GEMV.  No extra launches.
   - Rotation, split path: `mla_q_abs` (glm_fast.cu ~6140) rotates its output rows in place after the GEMV
     (`fwht256_inplace_cuda` over [n_head*ncol*2] contiguous 256-halves - q_abs is dense).  The attention
     kernels rotate their ctx back IN KERNEL (each after the P·V accumulation, before writing ctx out):
     `mla_out`'s ctx is per-token strided, so a host-side rotation there does not batch.  Each of the seven
     glm_batch.cu attention kernels (mla_attn_kernel, _tc_, _tc2_, _v70_, _tc_reg_, _mma_, _wmma2_) and
     glm_fast.cu's mla_split2_kernel needs the butterfly + the q4 read arm.
   - Flag: a process-global device flag (the `qsa_set_kv_q4` pattern) instead of threading a third bool
     through ~40 call sites - one model, one kv type per run, so the ternaries read `q4() ? lat4_... :
     q8 ? lat8_... : fp16`.  glm_fast.hpp signatures unchanged.
   - Gate: q4_0 requires kv_lora == 512 exactly (the butterfly's shape) - loud refusal at load otherwise.
   - Flip: `lat_records_ = lat_type_` (glm_model.cu) ONLY after the whole sweep lands; until then stage A
     keeps recording INT8.  Stage B is all-at-once - a partial path (some kernels reading records, others
     fp16 rows) is silent corruption.
   - Acceptance: the `<H512 x, H512 y> == <x, y>` orthonormality test (1e-6, random vectors); fp16-vs-fp16
     control byte-identical, then fp16-vs-rotated-q4_0 same continuation while near-tied; q4_0 slot
     round-trip byte-identical; latent bytes/slots and the streaming-window arms reported.

   **STAGE B LANDED (`GLM_KV_Q4_STAGE_B.patch`, verified `git apply --check -C1 --cached` against 56f1b72;
   two deviations from the plan above, both deliberate):**
   - Plumbing is an INT `lat_q8` (0 fp16 / 1 int8 / 2 q4_0), not a device flag: the model member `lat_q8_`
     becomes int and every caller (prefill/window/rows/mtp/fast_path) passes it unchanged - no ~40-site
     sweep, no `extern __device__` (no precedent in the tree; per-TU device copies would silently diverge).
     Kernel params go bool->int; ternaries dispatch 3-arm.
   - ctx rotation is HOST-side post-passes, not per-kernel butterflies: every attention kernel writes ctx
     DENSELY (`ctx + (t*n_head + h0)*KV`), so one `fwht512_rows_kernel` (kv_q4.cu: rows at
     `base + (i/rows_per_col)*col_stride + (i%rows_per_col)*512`, one block a row, 256 threads, two H256
     halves at 1/16) covers every site: `mla_attn`/`mla_attn_wmma2` rotate q_abs at ENTRY and ctx at EXIT
     (prefill's ctx->wv goes through cublas, NOT mla_out); `mla_out` gains a trailing `int lat_q8 = 0`
     and rotates ctx at entry (window batched + rows paths); `mla()` rotates its dense scratch after
     `mla_split_attend`; `mla_split_attend` rotates q_abs at entry and tells `mla_attend_batch` (new
     trailing `bool q_abs_rotated = false`) it is already done - no double rotation on any path.
     EXCEPTION: the fused `mla_kernel` does everything in one smem buffer - its `s_qa`/`s_ctx` get
     in-kernel butterflies (only it, plus both dsa_prep stores).
   - HIP: q4_0 loudly falls back to INT8 records at load (the f16q prompt attention has no rotation for its
     FP16 queries; `mla_attn_f16q` refuses loudly as belt-and-suspenders).  Production is CUDA.
   - Gate: kv_lora must be 512 exactly for q4_0 - loud refusal at load otherwise.
   - Flip: `lat_records_ = lat_type_` IS in this patch (all-or-nothing); the content-derived
     STRATA_STATE_SIGNATURE changes with glm_batch.cu, so stage-A int8 slots refuse by identity+digest.
   - Test: `tests/glm_lat4_test.cu` (registered beside glm_lat8_test, TIMEOUT 60) - host-FWHT-reference
     agreement, self-inverse, `<Hx,Hy> == <x,y>` to 1e-6, store->decode round-trip bit for bit plus a
     one-quantization-step error bound.

   **Stage-B invariants and the streaming-window win (user corrections, all verified):**
   - The Hadamard must stay ORTHONORMAL (scale 1/16 per 256 half = 1/sqrt(256); `fwht256_cuda` already
     applies it - kv_q4.hpp). Unscaled, every score multiplies by 256^k and the logits shift systematically -
     which looks like a precision problem, not a bug, and would be blamed on q4_0 forever. Hard test:
     `<H512 x, H512 y> == <x, y>` to 1e-6 on random vectors.
   - The failure mode is silent: a rotation applied to q_abs but not the stored latents (or a gather rotated
     back twice) costs a few quality points and nothing else - no crash, no log line. Differential test, not
     "the answers look right": fp16-vs-fp16 control must be byte-identical (defines what "same" means), then
     fp16-vs-rotated-q4_0 must produce the same continuation while sampling stays near-tied.
   - q4_0 shrinks the STREAMING WINDOW too, not just the host cache: --kv-resident 32768 costs 32768 x
     1024 B x 11 = 0.37 GB (log-exact) today; at 288 B/row that same 0.37 GB holds ~116,500 positions -
     3.5x the window - or keeping 32768 hands ~0.26 GB (~30 slots) back to the expert pool. The window is
     what decides how many DSA reads miss VRAM, and residency is the lever that has moved the needle all
     night (42.03 tok/s at 90.2% hit vs 25.43 at 74.5%). Measure as a separate arm; the knob is --kv-resident.
5. `PROMPT_CACHE_STAGE2B.patch` (rebased, real hunks) - the EWMA persists beside the slots keyed kv/k/ctx;
   `saved~?` when unmeasured.
6. `SLOT_ASYNC_SAVE.patch` (rebased, real hunks) - D2H inline (~25 ms), NVMe write + rename + meta on a
   side thread; `slot_drop`, a superseding save and both shutdown paths join first.

Stage 3 (break-even retention) and the pinned-RAM tier are not written.

---

## 0. The premise correction (read this first)

The task brief said "there is no file I/O for state anywhere in the tree". That is wrong, and
it changes the port: Maya **already has** the whole machinery, in two layers:

1. **RAM checkpoints** (`Glm5Model::ckpt_save`, glm_fast_path.cu ~6962–7100; driven from
   generate.cpp `STRATA_GLM_CKPTS`/`STRATA_GLM_CKPT_EVERY`): up to 4 host-RAM states of the
   live conversation at the system-prompt boundary, the last-turn marker, and every 16384
   fresh tokens. This is the fork's `--ctx-checkpoints` feature, already ported, already
   keyed by turn markers (better than the fork's spacing-only rule).
2. **Conversation slots** (`Glm5Model::slot_save/slot_load`, MAYALST2, generate.cpp
   ~1266–1560): full state to disk or RAM buffers — KDA recurrent snapshot (`snap_`), the
   open pool rows (`snap_pool_`), DSA latents — with tmp+rename atomic publish, a `.meta`
   sidecar carrying a **model identity string** (engine version, MTP flag, pack index/dense
   shard sizes, all .gguf shard names+sizes), LRU eviction, a byte budget, an 8 GB free-disk
   guard, and `STRATA_GLM_SLOT_KEEP=1` persistence across restarts with age eviction.

So the port is not "invent a checkpoint cache". It is: **harden what slots validate, then
tier the storage like the expert cache**. The parts of the fork worth copying are its
*validation discipline* (per-blob hashes checked on use); the parts worth NOT copying are
its keying (slot-scoped, not content-scoped) and its append-mode non-atomic slot-file
writes.

---

## 1. CRITIQUE of the fork's checkpoint design (blunt, as requested — rev 3 corrections)

Rev-3 credit where it is due, because the header grep undersold it: the fork's disk entries
(`<entry>.lpcs` referencing `<entry>.lpcs.ck<i>`) already solve atomicity, torn-file
handling and dedup properly — published spill files are hard-linked into entries (no bytes
move; NTFS dedupes identical checkpoints across entries for free), anything unpublished is
tmp+rename, the boot sweep finds every `<entry>.ck*` **including `.tmp` leftovers by
naming alone** so it works even when a header cannot be parsed, spill blobs carry hashes
that `read_blob` verifies on use, and reads above a size threshold go parallel
(`cache_disk_read_parallel`). Restore-time checksum failure is handled by the
scar-tissue rule: the checkpoint is NOT applied, `n_past` is not advanced, the prompt is
reprocessed — the exact crash that rule prevents is documented in the code.

What is still weak, revised accordingly:

1. **LCP → checkpoint selection has a subtle wrong-depth risk (the fork's rule, audited).**
   The fork picks the newest checkpoint (reverse scan) whose entire span lies inside the
   shared prefix: `cur.pos_max <= pos_next` and `cur.pos_min < pos_min_thold || 0`. When
   the common prefix falls **between** two checkpoints, the deeper one is skipped (its
   `pos_max` covers tokens the new prompt does not share) and the shallower one restores,
   reprocessing the gap. That is correct — but the invariant doing the work is easy to
   break: *a checkpoint may only be restored if every token it holds is shared with the
   new prompt, and the restore must leave at least one token to process*
   (`[TAG_PROMPT_LOGITS]` clamps `n_past` for exactly this). Maya's equivalent rule is
   simpler and already correct: `slot_take` requires the stored tokens to be a strict,
   exact prefix of the prompt, longest match wins; a prompt that shares a *partial* prefix
   simply misses and prefills. For agent workloads (each turn extends the previous prompt)
   Maya's whole-conversation slots hit the common case exactly; what Maya lacks is the
   fork's property that a *new* prompt sharing a long prefix with a cached entry still
   hits — stage 4's content addressing + mid-prefix checkpoints is where that lands.
2. **Refcount lifetime across restart (fork's answer, ported).** The fork's entry files
   reference their checkpoints; a restore hands the files to live checkpoints whose
   backing deletes them when the last reference goes. Get that wrong and you leak or
   delete a live checkpoint. The restart case is solved from the filesystem alone:
   ownership is recoverable by naming (`.ck*` suffix sweep), never by parsing. Maya's
   slots sidestep refcounting entirely — each `.bin` is owned by exactly one slot, and
   the `.meta`-judged startup sweep deletes orphans by name+meta pair — but stage 4's
   shared entries (two conversations, one prefix file) MUST import the fork's
   reference-or-hardlink discipline: hard-link the published file into each entry, never
   inline bytes, and never delete a `.bin` whose meta still names it.
3. **Identity: the fork's header does NOT carry k / kv precision / context / pack path**
   (the open question — checked, it carries tokens, positions, sizes and per-blob hashes;
   it is protected only by being single-model-per-process and by the boot sweep). Maya's
   `.meta` carries pack shard names+sizes, engine version and the MTP flag — better — but
   **not `k`, not the context**, and kv precision only indirectly. In Maya these all
   change at launch: `MAYA_GLM_EXPERT_USED` (k=7 vs 8), `--kv` precision, `--max-context`,
   the pack directory. A checkpoint restored across a change in ANY of them is
   semantically wrong and would not look wrong — layout-identical, silently corrupting,
   the exact class that cost two days. Stage 1 adds `k` and `ctx` to the identity; kv
   precision is enforced by the existing run-size layout check (`lat_q8` changes
   `lat_pp` → run sizes differ → loud "the slot's layout differs from this model's"); the
   pack directory is enforced by shard sizes in the identity today (colliding sizes across
   two packs is the residual gap — stage 4 hashes the pack dir into the identity).
4. **Keying is per-slot, not per-content** — two conversations with the same prefix
   cannot share a checkpoint (stage 4 fixes with content addressing; the fork's hardlink
   dedup is the mechanism to copy).
5. **Eviction is oldest-first, not usefulness-first** — a system-prompt-boundary
   checkpoint ages out at the same rate as a mid-turn one; the measured policy (§2.3b)
   replaces this.
6. **No hit/miss accounting** — restores, bytes, seconds saved are uncounted, so the
   feature's value is believed, not measured. §2.6 closes this.

---

## 2. TIERED PORT DESIGN (VRAM / pinned RAM / NVMe, mirroring the expert tiers)

### 2.1 What is persisted

A checkpoint = what `slot_save` writes, and nothing else: `snap_` (KDA recurrent+conv),
`snap_pool_` (open ik/ig rows), DSA latents (fp16 or q8 per `fast_mode_`/`lat_q8_`).
The kv-streaming VRAM block maps are NOT persisted — `slot_load` already calls
`kv_stream_reset_maps()` and `dflash_reset()` after a restore, which is the correct
contract: a restored checkpoint invalidates streamed-KV bookkeeping and the drafter's
cached positions. Do not try to persist those.

**Size reality (measure before budgeting):** the recurrent-only part is fork-145-MiB-like,
but the DSA latents scale with position. At 128k tokens with a ~1 GB full-context KV
(the log's `state/KV (262144 ctx) 0.99 GB`), expect a 128k checkpoint to be **~0.4–0.6 GB**,
not 145 MiB. At 2142 tok/s prefill, ~60 s of recompute vs ~0.2 s of NVMe read at the
measured 3.15 GB/s — the disk tier is still ~three orders of magnitude cheaper. Even so,
budgets below are sized for GB-scale entries, not MiB-scale.

### 2.2 Keying — extend the fork's key, do not invent one (rev 4)

**First, the feature classification that changes what gets ported.** The fork has two
features and only one is a persistent cache:

- **(a) The prompt cache** (`--cache-dir`/`--cache-disk`): keyed by the signature below,
  LCP-matched, persists across restarts. Fast and reliable — **this is the port target**.
- **(b) Context checkpoints** (`--ctx-checkpoint-dir`): deliberately NOT keyed;
  server-context.cpp:2302–2321 sweeps every `ckpt-*` file at boot with the comment that
  "they are only ever valid for the process that wrote them (they key on nothing), so a
  stale one is pure garbage occupying disk".

Maya's `ckpt_buf_`/`ckpt_save` is (b) in nature — process-local scratch. Spilling it and
restoring it across restarts would reproduce the unkeyed variant, which is exactly the
dangerous one. **Port (a); let Maya's checkpoints become its payload.** The slot system
(`slot_save`/`slot_load`) is the (a)-shaped machinery already present — stage 1 hardens it
into a keyed, checksummed, persistent cache.

**The fork's key, exactly** (server-context.cpp:2263–2278):

```
sig = model.path | n_ctx | n_parallel | flash_attn_type | cache_type_k | cache_type_v
    | build_part
build_part = "--cache-key-epoch" set      ? "epoch:" + epoch
           : state signature available       ? "state:" + llama_state_signature()
           :                                   build_number + "|" + commit
h = FNV-1a(sig);  prompt_cache->key = h;
```

The reasoning worth copying verbatim into Maya: keying on the commit "invalidated every
cached prefix on ANY rebuild, including ones that only touched server I/O or telemetry —
while still not being the right guard, since what matters is whether the state's MEANING
changed". Key on the meaning of the serialized state, not a version string; if the digest
is unknown, fall back over-strict (build+commit), never permissive.

**Maya's key extends the sig with what changes at launch AND changes meaning:**

- **pack identity**: content-derived, not name-derived — a hash of `index.txt` plus the
  size+mtime of `dense.bin`. The `.meta`'s shard list is names+sizes only, which collides
  across two packs with identical filenames; the stage-1 patch adds the content hash.
- **`MAYA_GLM_EXPERT_USED` (k)** — production runs k=7 against a pack whose metadata
  says 8; k changes expert routing and therefore the state's meaning
- **the KV precision** (`o.kv`: fp16/int8/q4_0/k8v4) — in the identity, not inferred from
  size; it changes what the serialized latents mean
- **a Maya state signature** (the `llama_state_signature()` equivalent): a build-time
  digest of the recurrent/KDA state layout, the slot read/write path and the graph
  builders, so a rebuild that changes recurrence or state layout invalidates entries on
  its own — with the same over-strict fallback if unavailable
- **`--max-context`** (bounds `n_pos`; already checked inside `slot_load` too, defense in
  depth)

**Explicitly NOT in the key** (same exclusion our sig applies to batch/ubatch): the
expert-cache / RAM-tier sizing, slot counts, `STRATA_GLM_SLOT_*` budgets — they change
*placement*, not state meaning. `STRATA_GLM_CKPT_EVERY` and checkpoint spacing likewise:
placement, not meaning.

### 2.3 Tier policy — TWO tiers, no VRAM (rev 2: VRAM tier dropped entirely)

The revised decision, from your measurements: a checkpoint's recurrent slice is
d_inner 8192 × kda_head_dim 128 × 4 B × 34 recurrent layers ≈ **136 MiB**; at
8.85 MB/expert-slot that is ~16 slots of proven decode leverage (42.03 tok/s @ 90.2% VRAM
hit vs 25.43 @ 74.5% — 1.65× on residency alone), spent on state that may never be
reused, while the NVMe read of the same 136 MiB costs **~45 ms** at the measured
3.15 GB/s. A cold checkpoint on disk is a first-class hit, not a fallback. This is also
why prompt caches are conventionally RAM-based and never VRAM-based. **The VRAM tier is
not a small budget — it is none.** The only VRAM residency a checkpoint ever gets is the
restore itself landing into the model's own state allocations, pinned by the existing
`snap_in_slot` logic until superseded. No new knob, no rent.

| Tier | Holds | Granularity | Eviction within tier |
|---|---|---|---|
| **NVMe (primary)** | everything | one file + meta per checkpoint | the measured policy in §2.3b, then age (`STRATA_GLM_SLOT_DAYS`), then byte cap |
| **Pinned RAM (bounded, newest only)** | the newest checkpoint's bytes in a host buffer | one `vector<uint8_t>` | newest-wins: a save or restore replaces the single RAM entry; everything older lives on NVMe only |

**Save path:** every checkpoint is written to NVMe first (tmp+rename, checksummed meta —
the disk tier is the source of truth). The newest checkpoint's bytes additionally land in
the one-entry pinned-RAM slot (write-through). The state is already in the model; nothing
else is copied.

**Restore path:** `slot_take` restores from the RAM entry if it matches (≈150 ms), else
reads NVMe (~45 ms for the recurrent slice, ~0.5–1 s with the DSA latents at 128k —
see the size reality note in §2.1). Either way the restored state is then resident in the
model exactly as a live conversation is; there is no extra pin timer — Maya already holds
the restored state until `slot_put` or the next prompt overwrites it.

### 2.3b The retention/eviction policy is MEASURED, not configured

Maya already decides how to operate from startup measurements, not constants
(`glm_fast_path.cu:3298/:3321` — per-expert CPU cost vs PCIe cost picks the CPU lane's
experts; `glm_cpu_bench.cpp:181` reports each shape in µs and GB/s; `maya.py`'s CAL
persists tuned settings per PC). Extend that habit to the cache:

a. **Measure restore cost.** At startup, one calibration read: write a throwaway
   checkpoint-sized file (or read the largest kept `.bin`'s first 136 MiB), time it,
   derive `restore_GBps`. Reuse the CAL pattern: persist the measured constant per PC
   (`<pack>/cal/cache-cal.txt` — engine version + GBps + the §2.1 measured bytes/token)
   so a restart does not re-learn. Real restores then update an EWMA of
   (bytes, wall-ms) — same shape as the CPU/PCIe probe, not a one-shot guess.
b. **Measure what it replaces.** The engine already reports prompt tok/s per request. The
   recompute cost of a prefix of N tokens is `N / tok_s EWMA` for the *actual* prefix
   length, never a constant.
c. **Derive break-even; decide retention.** Keep a checkpoint while
   `restore_ms + expected_miss_cost < recompute_ms × P(reuse)`; evict the entry with the
   worst margin first. `expected_miss_cost` is the cost of a wrong keep (disk space only
   — a kept-but-never-used checkpoint costs nothing at request time, so the real question
   is eviction order, not admission).
   **Reuse probability — the honest hard part, stated simply:** `P(reuse)` = EWMA of
   per-checkpoint hits with a Laplace-style prior: `P = (hits + 1) / (takes + 2)`, decayed
   by age so a checkpoint unused for `SLOT_DAYS` decays toward the prior. One explainable
   rule beats a clever opaque one; the audit trail (§2.6) is what catches it being wrong.
d. **Persist the constants per PC** like CAL: restore GB/s, bytes per checkpoint-token,
   prefill tok/s EWMA. On startup, load them; re-measure only if the file is missing or
   the engine version changed.
e. **Expose the accounting** (§2.6) in the metrics the Monitor already polls, so the
   policy is auditable rather than believed.

### 2.4 Budgets (knobs)

| Knob | Default | Meaning |
|---|---|---|
| `STRATA_GLM_SLOT_RAM_GB` | 0 (off; 16 when set) | pinned-RAM tier cap — exists today |
| `STRATA_GLM_SLOT_GB` | 16 | NVMe tier cap — exists today (raise to 65536 for the 128 GB budget if you want the full fork parity) |
| `STRATA_GLM_SLOT_KEEP` | off | cross-restart persistence — exists today |
| `STRATA_GLM_SLOT_DAYS` | 7 | age eviction — exists today |

No new VRAM knob. The one-entry RAM slot inherits `STRATA_GLM_SLOT_RAM_GB`'s cap.

### 2.5 One pool or separate allocations?

**Separate.** The expert pool's slots are 8.85 MB blobs with a residency table, admission
policy and refill machinery tuned to expert-shaped reads; a checkpoint is one contiguous
object with a different lifetime (whole-object promote/demote, pinned by an in-flight
conversation). The tiers share only the *object*: same bytes, same key, same validation,
copied between homes. The RAM tier is a host ring (already built: `slot_ram`), the disk
tier a directory (already built), and the restore lands in the model's own state (already
built). Nothing new is allocated.

### 2.6 Hit/miss accounting (so it is measured, not believed)

One counter block, logged on the existing `NOTE sessions:` channel and in a `glm slots:`
stderr line at each restore:

- `ckpt_hits_ram`, `ckpt_hits_disk`, `ckpt_misses` (a prompt with a prefix match that found
  nothing = a miss; a prompt with no prefix match is not a miss, it is a new conversation)
- `ckpt_bytes_read` per tier, `ckpt_restore_ms`
- `ckpt_seconds_saved` = `(prompt_tokens - restored_tokens) / measured_prefill_tok_s` —
  computed from the engine's own last-measured prefill rate, not a constant
- `ckpt_vram_rent_mib` (always 0 at default) and, if the opt-in is on, the decode delta it
  costs — the number that justifies turning the VRAM tier back off.

### 2.7 Validation: a bad restore CANNOT be silent (the stage-1 core)

Every tier, every entry, every restore checks, in order:

1. **Meta magic + identity** (`kMetaMagic`, `slot_identity`): engine version, pack shards,
   MTP flag — **plus k and ctx** (stage 1 adds these). Mismatch → the entry is deleted and
   the refusal is *logged by name*, not counted silently.
2. **Content checksum** over the whole `.bin` (FNV-1a 64, streamed) at startup indexing
   *and again at restore time* — a file that rots between index and use is caught. The
   meta's byte count alone cannot tell a torn file from a whole one; the fork's appendix
   gap, closed.
3. **Prefix match**: the stored tokens must be a strict prefix of the incoming prompt,
   `len < prompt.size()`, exact `std::equal` — existing `slot_take` rule, unchanged.
4. **Layout** (inside `slot_load`, already there): magic, halves, `n_pos`, per-half layer
   range, `n_rec`, `run`, per-run lengths. Mismatch → loud err, slot dropped, prompt
   re-read from token 0 — **never** a partial restore with `n_past` advanced (the fork's
   scar-tissue rule, already honored here).
5. **Post-restore**: `kv_stream_reset_maps()` + `dflash_reset()` (already there) — the
   restore declares its own invalidation instead of hoping the maps happen to align.
6. **Hash-cost discipline (rev 2)**: the `.meta` stores size + mtime alongside the full
   checksum. Startup indexing verifies the whole hash **once per boot** (the torn-file and
   bit-rot guard; seconds of I/O per restart, cheap next to a 60 s prefill). The restore
   fast-paths on size+mtime match and only re-hashes when the file changed — a restore
   costs ~0 hash I/O, not a second full read of a multi-GB file. Checksum presence is an
   explicit `has_sum` field, not a 0-sentinel (0 is also `file_sum`'s open-failure value).
   Residual risk, stated: bit-rot that preserves size and mtime passes the restore fast
   path until the next startup full-hash; accepted because writes are tmp+rename (a torn
   file never becomes a `.bin`) and rot is not a live failure mode on this box.
7. **Hard-fail vs degrade**: a corrupted entry is deleted and logged loudly (`glm slots:`
   ERR line with the path, reason, and the tokens that were lost), and the request falls
   back to full prefill. It never degrades quietly; it never half-restores.

What is logged at each refusal: `glm slots: refused <path> (<reason>) - removed`. What
hard-fails (request-level ERR): a restore that read back but failed `snapshot_restore()`.

---

## 3. Stage plan (what stage 1 implements vs defers)

- **Stage 1 (the patch): persist one checkpoint to NVMe, restore it, prove byte-identical
  continuation to a no-restart run — with loud validation.** Concretely, on the existing
  slot machinery (the (a)-shaped feature):
  - IMPLEMENTS: whole-file checksum in `.meta` (verified at startup indexing AND at
    restore time); `k` and context in the identity; loud by-name refusal lines at startup;
    checksum re-verification in `slot_take` before trusting the file. Cross-restart
    persistence via the existing `STRATA_GLM_SLOT_KEEP=1`; atomic tmp+rename already there.
  - DEFERS: the measured retention policy (§2.3b) — stage 1 evicts by the existing LRU +
    byte cap + age; the state-signature digest (stage 1 falls back over-strict: engine
    version already in the identity, so any rebuild changes it — acceptable until the
    digest exists); kv precision stays enforced by the run-size layout check; LCP
    mid-prefix hits and content addressing (stage 4); the RAM tier (stage 2).
- **Stage 2: the pinned-RAM tier.** `STRATA_GLM_SLOT_RAM_GB` exists; stage 2 adds
  write-through on save, the newest-only RAM entry, promotion-on-restore, and the
  accounting block (§2.6) plus the CAL-persisted measured constants (restore GB/s,
  bytes/token, prefill EWMA) and the EWMA-driven eviction order from §2.3b.
- **Stage 3: the measured retention policy fully live** — break-even margin per entry
  (`restore_ms + expected_miss_cost < recompute_ms × P(reuse)`, Laplace prior decayed by
  age), worst-margin-first eviction, accounting in the Monitor's polled metrics. Stage 2
  ships the measurements; stage 3 acts on them.
- **Stage 4 (optional): content-addressed entries** (`pc-<token-hash>.bin`) with the
  fork's hardlink/refcount discipline (published files hard-linked, never inlined;
  ownership recoverable from the filesystem alone; sweep by naming, never by parsing) and
  LCP mid-prefix checkpoint hits — the fork's property Maya lacks, that a new prompt
  sharing a long prefix with a cached entry still hits.

---

## 4. Acceptance test (run after you build stage 1)

**`STRATA_GLM_SLOT_KEEP=1` is off in every launcher today — every run below sets it
explicitly or the test proves nothing.** Also note: **prefix hits work today** —
`slot_take` matches stored tokens as a strict prefix of the prompt (`t.size() <
prompt.size() && std::equal(...)`), so a turn-2 prompt (turn 1's prompt + output + new
input) hits without stage 4. Stage 4 only adds the shorter-shared-prefix case (edited
turns). Step (d) below exercises the prefix hit; do not conclude "disk cache does not
help" from an exact-repeat-only test.

Env for every run: `STRATA_GLM_SLOT_KEEP=1`, `STRATA_GLM_SLOT_DIR=G:/AI/maya-eval/ckpt-test`,
`STRATA_GLM_SLOTS=4`, `STRATA_GLM_SLOT_MIN=512`, production values otherwise (k=7,
MTP off, kv fp16, ctx 262144).

**(a) Run A — populate.** Fixed prompt P (the 132k agent-style prompt), greedy tail
(temp 0, n=64). Capture the 64 tokens. Exit the engine cleanly (QUIT) so `slot_put("on
shutdown")` writes the slot; confirm `slot-*.bin` + `.meta` exist in the dir.

**(b) Restart.** Startup must log `folder ...: 1 conversation(s) kept from earlier runs`
and no `refused` lines.

**(c) Run B — exact repeat.** Same prompt + tail. The restore line to look for:
`took back a N-token conversation from disk (...); M new tokens to read` with N ≈ prompt
length, M ≈ 64. **Pass**: 64 continuation tokens byte-identical to run A; prefill wall
time drops from ~60 s to <2 s (PP line).

**(d) Run C — EXTENDED prompt (the real agent case).** P + 512 new tokens + tail.
**Pass**: same restore line (the slot's tokens are a prefix of the new prompt), M ≈ 576,
continuation byte-identical to a no-restart run of the same extended prompt. If this
misses, the matching limitation — not the cache — is the finding; do not report it as a
speed result.

**(e) Corrupt one byte** in the middle of `slot-*.bin`. Restart: startup must hash it
(verify_full path), log `glm slots: refused <path> (foreign identity, incomplete, or its
checksum failed) - removed`, delete the pair, and the next request must prefill from
token 0 with a NOTE line — no silent drop. Truncating by 4 KB must refuse identically.

**(f) Flip k=8** (`MAYA_GLM_EXPERT_USED` unset → pack default). Restart: the kept slot's
identity no longer matches (`k=7` vs `k=8`), so it is refused loudly at startup, removed,
and the continuation is still correct (prefilled). Also flip `--kv int8`: refused the same
way — and note the layout check inside `slot_load` is the backstop, not the primary guard.

**(g) Timing at 128k.** Baseline prefill ~2142 tok/s ⇒ ~60 s. Restored turn: PP line
≤ ~2 s; the restore itself ~0.2–0.5 s for a ~0.5 GB file at ~3.15 GB/s. Record both in
TESTING.md as full distributions (T6: no point estimates from single runs).

**Where I am unsure (so a confident wrong design doesn't cost you a night):**
- Exact checkpoint sizes at 128k (0.4–0.6 GB is an inference from the 0.99 GB full-ctx KV
  line, not a measurement) — measure with `STRATA_GLM_SLOT_TIMING=1` on the first save.
- The stage-1 patch is written against the code as I read it; line numbers in hunk headers
  are approximate, and `git apply` may need `-C1` fuzz. The `model.geometry()` accessor is
  real (glm_model.hpp:266) but I could not compile-check the call sites.
- Whether `slot_put`'s supersede rule interacts badly with `SLOT_KEEP` across restarts
  (anchor/prefix flags are not in the `.meta`) — stage 4 material, flagged here so it
  doesn't surprise you.

## 5. Record (from the stage-1 build/test, b2cfc87)

- Slot identity does NOT need reasoning_effort: changing it rewrites the system prompt via
  the template ("Reasoning Effort: High"), so the token prefix changes and old slots stop
  matching - implicitly keyed by the tokens.
- Restore cost split, measured: whole-file read of the 160 MB slot is 45 ms warm (3571 MB/s)
  out of ~300 ms - the read is ~15% of a restore; the remaining ~250 ms is the 16 MiB-chunk
  cudaMemcpy loop plus snapshot_restore(), split not yet measured. Parallel reads are kept
  for the cold case (no page cache) but no win is claimed for the warm path.
- The marker-free mark-slot fix works: turn-2 grown conversation reuses 98.8% (0.37 s);
  after a restart, "took back a 1198-token conversation from disk (0.3 s); 11 new tokens".
  Trade-off as predicted: the last turn's body is re-read.
- FOR BOTH ENGINES: the agent session's "please continue" stalling stopped at reasoning
  effort HIGH (it rewrites the system prompt via the template). Our llama.cpp production
  config runs --reasoning-effort low - likely wrong for agent workloads generally. Flag in
  any write-up; measure before changing production.

