# Layer pipeline (RxTy) — spec

Status: **spec only, nothing built.** Written against the current tree (`feat/rows-de-fused-pf` @ `84054cc`,
production config `--parallel 2`, `--prefill 65536`, `kv=q4_0` streaming).

## Notation

- **RxTy** = request `x`, token `y` of that request. Two requests decoding together are **R1T1** and **R2T1**;
  the scheme generalises to any set of requests and any tokens within them.
- Per (token, layer) there are three stages, and the engine already times each one (measured on a real decode,
  `STRATA_GLM_PROF`, ms per token of a 43.38 ms token):

| stage | what it is | engine mark | measured |
|---|---|---|---|
| **x** | fetch that token's routed experts RAM → VRAM | `moe_fetch` | **9.67 ms** (22.3%) |
| **y** | execute the token against those experts | `moe_gate_up` + `moe_down` | **4.31 ms** (9.9%) |
| **z** | route the *next* layer's experts for that token | `moe_route` + `router_shexp_mv` | **2.24 ms** (5.2%) |

So one token's layer costs `x + y + z ≈ 16.2 ms` when run serially, and the pipeline's goal is to make it
`max(x, y + z) ≈ 9.7 ms` by keeping the transfer engine busy while another token computes.

## The schedule

**Warm-up (first layer).** Run the first layer as the code does now for every request in the set — or batch them,
which costs slightly less than the sum of their x/y/z because the layer's expert set is shared. This is the only
stage that cannot be overlapped, because nothing is ahead of it yet.

**Then, per layer L, offset the two requests by one stage:**

```
x  (R1T1, L)
y  (R1T1, L)   ‖  x (R2T1, L)
z  (R1T1, L)   ‖  y (R2T1, L)
x  (R1T1, L+1) ‖  z (R2T1, L)
y  (R1T1, L+1) ‖  x (R2T1, L+1)
z  (R1T1, L+1) ‖  y (R2T1, L+1)
...
```

Read it as: R1T1 is never more than one stage ahead of R2T1, and at every step the two are doing *different* things
— a transfer beside a compute, a route beside an execute — so the PCIe link and the SMs are both busy. Note the
ordering constraint that makes this legal: `z(R1T1, L)` produces the routing that `x(R1T1, L+1)` consumes, so those
two stays ordered per token, while everything across tokens is independent.

With more than two requests the same pattern extends: the token set advances through the stages in a rotation, and
the leader is the one furthest ahead that the memory below allows.

## What bounds it: VRAM, and therefore how far the leader may run ahead

The leader must pause (wait for the follower) when the expert sets it wants resident no longer fit. So:

    max_lead_layers ≈ (free VRAM slots) / (experts per layer per token × tokens in flight)

With this box's numbers:

- expert set per layer per token: `n_exp_used = 8` experts; a slot is ~8.8 MB (31.17 GB / 3,541 measured slots);
- free VRAM with the production pool: **~1.2 GB** (pool 36.26 GB of 47.76 GB; 4,104 of 12,096 slots resident);
- so ~136 free slots ≈ **17 layers** of one token's set, or **~8 layers** with two tokens in flight.

That is the useful number: the bound is not one layer, it is several — so a 2-request pipeline has ample room and
the pause case is a fallback rather than the normal state. Adding requests divides the depth (3 requests ≈ 5
layers, 4 ≈ 4), which is exactly the trade to measure if this works.

## Why the earlier "17% batching cost" does not apply

The rows are no longer lockstep, so `hc_rows` and the KDA window batching stop being usable *as they are*. But that
is not a loss to accept — in this scheme the point is that the GPU executes R2T1's layer while R1T1's transfer is on
the wire, so the fetch (22.3% of the token) and the CPU-lane wait (34.6%) both move into overlap. The comparison is
not "per-row launches vs batched launches", it is `x + y + z` serial against `max(x, y + z)` pipelined.

## Implementation shape (the work)

1. **Extract the layer body** into a callable per (token, layer) — `rows_layer(seq, il, stage_mask)` — and prove it
   produces a **bit-identical** token when driven sequentially for a single token. That de-risks everything else:
   it shows the loop can be split without changing arithmetic. Gate: token-for-token equality on a single request,
   cold boot.
2. **Per-token layer cursors** so a request can be at layer L while another is at L−1.
3. **Stage scheduling** on two streams: fetch on the side stream (`F->ps`, already exists), execute and route on
   compute; per-token events so a token waits only on its own fetch.
4. **The scheduler** that advances tokens in the rotation above and pauses a leader when the VRAM bound is hit.
5. **Correctness gate before any timing**: single request, sequential drive, token-for-token bit equality against
   today's path; then two requests with the pipeline off (both bit-identical), then on.
6. **Measurement**: the same prompt pair at warm and cold tiers, aggregate tok/s and wall, against the current
   `--parallel 2` lockstep baseline (**35.8 tok/s aggregate, 6.6 s for 240 tokens**, measured today).

Env flag `STRATA_GLM_LAYER_PIPE`, default off, so the existing path stays byte-identical until the gates pass.


## Status (2026-10-29)

| # | increment | state |
|---|---|---|
| 1 | `rows_layers(..., il_from, il_to)` - one layer issued on its own; `multi_rows` drives it per layer behind `STRATA_GLM_LAYER_PIPE` | **done** `cd3ed9f`, bit-identical gate passed (cold boot, greedy, one request) |
| 2 | `RowsState::RowCursor cur[8]` - the map: per row its layer, stage, lead over the slowest row, done. Seeded in `rows_bind`, advanced as each layer finishes | **done** `d49c2f4`, no arithmetic effect (same 90 tokens as the pre-cursor build) |
| 3a | **per-channel handshake**: `MoeDev.wait_ch`, `wait_seq[ch]`, `resp[ch]`, `MoeRequest.ch`; the host service answers into the requesting channel's buffer; the rows path gives row *i* channel *i* | **in** (this commit) |
| 3b | per-row compute streams + per-token events | next |
| 4 | the rotation scheduler with the VRAM lead bound | after 3b |

### 3a - why the handshake had to change first

`wait_seq` was **one** device word and `resp` was **one** host-mapped answer buffer, both shared by every waiter:

```
F->md.wait_seq = (unsigned int*) fa_take_b(at, 64);   // the arena take was 64 bytes, only word 0 used
F->md.resp     = dp;                                  // one MoeResponse
glm_fast.cu:4239  *a.d.wait_seq = mask ? seq : 0u;    // route sets it
glm_fast.cu:4311  s_need = *d.wait_seq;               // waiter snapshots it
glm_fast.cu:4440  *moe_wseq(d) = 0u;                  // fused kernel resets it
```

Two rows in flight would set, snapshot and reset the same word - a waiter can miss its own seq and spin to the
~60 s bound. That is the same race that closed the `PF=1` structural arm, and it is why "add threads" alone does not
work. Now `moe_wseq(d)` and `moe_resp_ch(d)` index by `d.wait_ch`, the route stamps `rq->ch`, and the service writes
the answer into `resp_h[ch]`. Channel 0 is the lone token path, so a single request is unchanged by construction;
`n` rows take channels `0..n-1` (the arena take is 8 words of usable room; `rows` is capped at 8).

**Slot isolation was already correct** - expert slots are partitioned per layer (`slot_of[key]`, `spare_key(il, j)`,
77-119 slots/layer of 4,104), so two rows at *different* layers never alias. The handshake was the only shared word
that mattered.

**Not yet done in 3a:** the service still processes the ring strictly in seq order, which is the right ordering for a
single service thread and is what the per-channel answer buffers make safe. Whether one service thread is enough
bandwidth to feed two pipelined rows is a measurement, not an assumption - the stage table says the service path (x,
9.67 ms) plus the CPU lane (15.03 ms) is 24.7 ms of the 43.38 ms token, and the pipeline's prize is
`max(x, y + z)` rather than `x + y + z`.

### 3b - what the skew needs, found by mapping it

Three things are per row *already*, and one is not.

| shared thing | per row? | why it matters |
|---|---|---|
| expert slots (`slot_of[key]`, `spare_key(il, j)`) | **yes, per layer** | two rows at different layers never alias |
| plan / fetch_src / cpu_part / cpu_flag / cpu_seq | **yes** (`mplan_ptr + 8 i`, ...) | the rows path splits them in `row_dev` |
| `wait_seq`, `resp`, `rq->ch` | **yes now** (3a) | was the blocking race |
| **group scratch: `W.whq`, `W.wpart`, `W.wh`, `W.grp_ptr`, `W.grp_start`, `W.grp_ent`** | **NO** | one window indexed by *row position in the call* (`tmax * 8`). Two rows on two streams both land in slice 0 |

So 3b needs a `row_base_` that offsets those windows (they are already `8` slices wide per row; a per-row call should take the slice the row owns, not slice 0), plus a stream parameter on the layer path, plus two compute streams.

**And there is one more hazard the code already warns about** (`glm_rows.cu:846`): the tier tables (`tab`, the promo counters) are touched by the *route* kernels and by the host service. Two rows routing on two streams concurrently would write `tab` concurrently. The route is cheap (2.24 ms of 43.38) and the fetch is the long pole (9.67 ms), so the shape that avoids this is:

- **routes stay in global order** (serialized, one at a time - they are short),
- **fetches run on the transfer lane** `F->ps`,
- **computes run on per-row streams** and wait only on their own fetch event.

That gives the overlap the scheme is for (`fetch(BN) || compute(AN)`) without concurrent `tab` mutation. Serializing the *computes* would give nothing; serializing the *routes* costs ~2 ms a row and buys the whole transfer window.

### Measurement is blocked on the box, not the code (2026-10-29)

The gates ran during live fleet load (`active=2`, GPU 98%, 147 W of 300, and the request log showing 100k/245k-token conversations being restored and decoded at ~10-12 tok/s each). My first concurrent timing read **1.7 tok/s aggregate** and the warm pair read 6.5 concurrent vs 5.9 sequential - both are measurements of the workers' queue, not of the change, so neither is quoted as a result.

Consequences for the next measurement:

- An arm must be paired **within the same load window**, or taken when `live.active == 0`. Absolute tok/s across windows is not comparable on this box.
- `live.active == 2` is the *interesting* regime (it is the pipeline's whole point), but it is also exactly when the fleet owns the GPU - so use the **per-request `decode_tok_s` in `/metrics.requests[]`** and pair arms by interleaving them, rather than wall-clock aggregates.
- Cold-boot token equality gates are unaffected by load and remain valid (3a passed that way).

### The alias I nearly missed: the row's own activation slices

`row_at(S->x, E, i)` / `rowq_at(S->xq, E, i)` index `RowsState`'s buffers by **row position in the call**. A skewed drive calls `rows_layers(rw + r, 1, ...)`, so `i` is always 0 and *both* rows would use `S->x[0]`, `S->qq[0]`, `S->rlog[0]`... That is the same class of bug as the shared group window, one level up, and it would corrupt silently rather than fault.

So the 3b change set is, in order:

1. `row_base_` (the row's slot in the 8-wide buffers) and a `slice()` helper so `S->*` and `W.*` both take the row's own slice;
2. `rows_stream_` (compute stream per row) - `rows_layers`/`rows_dsa`/`rows_moe` use it instead of `F->cs`, which is a one-line change each because the body already goes through one local `s`;
3. the rotation: serialized routes, fetches on `F->ps`, per-row compute streams waiting on their own fetch event;
4. then the timing, with paired arms (`/metrics.requests[].decode_tok_s`, interleaved) rather than wall-clock aggregates.

Steps 1 and 2 are mechanical but wide (every `S->*` and `W.*` slice in the layer path). Step 3 is small. Nothing in 1-3 changes arithmetic, so the increment-1 bit-identity gate keeps its meaning: it must still pass afterwards, on `LEPIPE=1` with one row.

## RETRACTION (2026-10-29): token-equality is not a valid gate on this engine

Increments 1 and 3a were reported as "bit-identical" on the strength of one 90-token answer matching a stored
answer. **That evidence was not sufficient and the claim is withdrawn.** Repeated runs of the *same binary*, same
prompt, temperature 0, produce different answers every time:

| condition | runs | result |
|---|---|---|
| default config, three cold boots | 3 | all three differ (first divergence at char 132 / 348 / 132 across different pairs) |
| `STRATA_GLM_CPU_LANE=0` (no host float summation) | 3 | all three differ; one stopped at 46 tokens, two at 90 |
| CPU lane off, warm tier (hit 0.634 -> 0.639), 4 sequential runs | 4 | lengths 409 / 416 / 413, none equal |
| same, 3 more runs | 3 | lengths 409 / 416 / 413, none equal |

So the divergence is not the CPU lane, not cold tier, and not my slice change: **it is a property of the engine as
configured** (q4_0 latent KV, streaming experts with varying residency - the hit rate moves between runs, so the
set of experts served from VRAM vs RAM differs, and those two paths do not produce bit-identical values).

**Consequence: no future arm may be cleared on token equality.** What is still true and useful:

- `row_base_` is an arithmetic **identity by construction** at 0 (`ptr + 8 * 0`); the branch that reads it is the
  only one that changes. That is a code-level argument, verified by inspection, not a token comparison.
- The gates that DO work: no faults (`illegal`/`misaligned`/`route_error`), coherent non-empty output, and the
  next one to build - a **numeric parity harness**: log the top-k logits at fixed positions from both arms with
  residency frozen, and compare those, not the sampled tokens.

Until that harness exists, "the pipeline is correct" can only mean "it does not fault and its outputs look like the
model's". That is weaker than what was claimed earlier in this document and is stated here so nobody quotes the old
claims.

### What increment 3b's slice work is, then

Every per-row slice in the layer path goes through `R(i) = row_base_ + i` (~40 sites), and the group window's arrays
(`mw.plan_ptr/plan_w/cpu_part/cpu_flag`, `grp_ptr/grp_start/grp_ent/n_groups`, `hq`, `part`, `wh`, `sh_hq`, `sh_out`)
take the row's `[T * 8]` block. `row_base_` is 0 for an ordinary call, so behaviour is unchanged; the value is that a
driver can now run one row per call without two rows sharing a slice.

Still per-*call* (not per-row) and therefore still a hazard for concurrent rows: `S->aq` / `S->actx` (the MLA attention
scratch, sized for one row's heads), `W.scr` (a 64-byte claim word), and `F->rows_scr`'s per-row 8 slots (indexed by
`R(i)` already, but the *count* is per call). Those are the next mechanical step before streams.

## Increments 3b/4 (2026-10-29): streams, rotation, and the four alias classes it exposed

Landed in code (all behind `STRATA_GLM_LAYER_PIPE`, default off):

- `FastState::cs2` - a second compute stream; `rows_stream_` on the model, used by `rows_layers`, `rows_dsa`,
  `rows_moe`, `rows_moe_grouped`, `rows_head` (`s = rows_stream_ ? rows_stream_ : F->cs`).
- **the rotation** in `multi_rows`: for k in 0 .. L+n-2, row r runs layer k-r on its own stream, so row 1 is a layer
  behind row 0. `row_base_` = r for the call, `cudaStreamSynchronize(cs2)` before `rows_head`.

Getting it to *run* exposed four alias classes that a lockstep drive never touched. Fixed:

| what | how it showed | fix |
|---|---|---|
| `F->conv[3]`, `F->g1`, `F->g2` (KDA mixer) | used whenever `n == 1`, i.e. every per-row call - the token path's single scratch | per-row `kconv`/`kg1`/`kg2` allocated unconditionally and taken by any per-row call (`per_row_bufs`) |
| 21 n-row launches (`MvJob` for the KDA/DSA/FFN/head projections) | started at row 0 whatever row the call was | every base argument now `rows(S->X, stride, rb)` / `rows_q(...)` |
| `S->aq`, `S->actx` | one block per call, written at stride `ad` | `+ rb * ad` |
| `S->ascore`, `S->acells`, `S->aslot` (DSA score/selection) | base-0 | `R(i)`; these only exist with `STRATA_GLM_ROWS_ATTN=1`, so the rotation **requires** it and refuses otherwise |

Symptom of each: no fault, no `route_error`, coherent-looking but wrong text ("Ex4x.. I, ..s... la1....zaca.."), then
"# ZFS RAIDZZ1 Parityi" once most of them were fixed. Silent corruption, which is why they had to be found by design
review rather than by a test.

**Still shared, so the rotation is NOT certified:** the expert **spare claims** and the route counters (`tab`, `dcnt`,
the per-layer spare pools) are claimed per *call*, and two rows at different layers route concurrently - their claims
can collide, and a collided claim hands one row's blob to the other's plan. That is the corruption still visible.

Therefore:

- `STRATA_GLM_LAYER_PIPE=2` alone now **does nothing but say so** and runs the layers one at a time;
- `STRATA_GLM_LAYER_PIPE=2` + `STRATA_GLM_PIPELINE_UNCERTIFIED=1` runs the rotation for development;
- the shipped path (no flag) is unchanged - verified after every step: coherent output, 0 faults.

Next, in order: (1) make the spare claim per row (a claim word per row, or serialize routes and let the computes
overlap), (2) re-check `hc_part`/`hc_ctr` (allocated only with `STRATA_GLM_ROWS_BATCH`, and a per-row call reads
`rows(S->hc_part, ..., R(i))`), (3) the numeric parity harness, (4) the timing.

### Where the rotation actually stands (end of this pass)

The bisect is the important result: **the same rotation with both rows on one stream produces correct output**
("# ZFS RAIDZ1 Parity Distribution ## Variable-Stride Allocation Unlike traditional RAID 5..."). So the per-row
slicing is right, and the remaining corruption is a **genuine concurrency race**, not a mis-addressed buffer.

Six classes of per-CALL state were found and fixed on the way (all silent corruption, no faults):

1. `F->conv[3]`/`F->g1`/`F->g2` - used whenever `n == 1`, i.e. every per-row call.
2. 21 `MvJob` launches that started at row 0 whatever row the call was.
3. `S->aq`, `S->actx` (one block, written at stride `ad`).
4. `S->ascore`, `S->acells`, `S->aslot`, and - the one that hid behind `n == 1` - the whole `attn` predicate:
   `attn = rows_attn_on() && n > 1 && ...` meant a per-row call silently took the token path's SHARED `F->score` /
   `F->cells`. Fixed to `(n > 1 || row_base_ > 0)`.
5. `S->apart` - `take_all`, one buffer for `attn_tokens` rows, indexed by token inside the call: `+ rb * part_floats`.
6. `attn_q`'s q8 output in the batched path.

**Still wrong under overlap**, and the symptom has changed shape - repeated degenerate tokens (`of the of the`,
`((((((`, `RAID RAID RAID`) rather than word salad. That is a state crossing, not a bad pointer.

**The next step is instrumentation, not another guess:** `compute-sanitizer --tool racecheck` on the overlapped arm,
which names the buffers and lines with hazards. Six hypotheses by inspection have now been spent; one racecheck run
is worth more than the seventh. (The sanitizer needs the CUDA toolkit's `bin` on PATH; the box had `compute-sanitizer`
off PATH at the time of writing.)

Everything stays behind `STRATA_GLM_LAYER_PIPE=2` + `STRATA_GLM_PIPELINE_UNCERTIFIED=1`; the default and the shipped
path are untouched and were re-verified after each build.

### Two more real bugs, and the failure mode that replaced them

Chasing the non-finite router input (log: `glm router: layer 19/20 decode invalid expert ID or non-finite score`)
found two more classes, both now fixed:

7. **`fast_moe_experts` ignored the row's MoeDev.** It rebuilt `md = F->md` (the GLOBAL plan / fetch_src / cpu-lane /
   wait channel), so in a per-row drive every row shared one plan and one channel. The per-row MoeDev only ever
   existed in `rows_moe_grouped`. Fixed by letting a pipeline row (n == 1, row_base_ > 0) use that path: the
   `n < 2` gate is now `n < 2 && row_base_ == 0`, and its route passes the row as `win` with `win_n = 0`.
8. **The residual mixing (mHC) used the token path's shared scratch.** `h.part = F->part`, `h.counter = F->counter`,
   used on *every* row, and `hc_part`/`hc_ctr` were allocated only with `STRATA_GLM_ROWS_BATCH`. Two rows wrote one
   residual buffer - a non-finite residual is exactly what the router then reported. Now per row (`R(i)`), allocated
   unconditionally. Also `h.xq` still indexed by `i` rather than `R(i)`; fixed.

Plus the stream bug behind the earlier chaos: `fast_moe_experts` and `fast_moe_route` hardcoded `F->cs`, so in a
per-row drive the **experts ran on the engine stream while the layer's inputs were produced on the row's stream**,
with no ordering between them.

After these, one run of three produced a fully coherent answer ("# ZFS RAIDZ1 Parity Distribution A RAIDZ1 vdev
protects against the failure of a single disk by ..."). The rest failed, but **differently**:

> `the engine stopped itself because it had stopped making progress - a hang it caught`

That is **head-of-line blocking between the two channels**, and it is structural, not a bug in the slicing:

- the service answers the routing ring strictly in `seq` order;
- the route kernels take that `seq` from one shared atomic counter, so the two rows' sequences interleave in
  whatever order the GPU happens to run them;
- `moe_wait` / `moe_wait_fetch` occupy one block per SM and spin. If row B's route publishes first and spins, row A's
  route may never get an SM to publish the earlier sequence the service is waiting for. Deadlock, caught by the
  watchdog.

**So the next change is the protocol, not the buffers:** either a per-channel ring (the service answers each channel
on its own, no global order), or serialize the *routes* across rows with events while leaving the computes overlapped
- which is the shape originally proposed and is the cheaper of the two.

Everything remains behind `STRATA_GLM_LAYER_PIPE=2` + `STRATA_GLM_PIPELINE_UNCERTIFIED=1`. Shipped path re-verified
after each build (coherent output, probe answering).

## Item zero, measured: the need_seq publication, isolated (2026-10-29, later)

Three builds, one variable each, plain single-request prompt every time (`Explain what a ZFS block pointer stores`,
greedy, 200 max tokens):

| build | result |
|---|---|
| `130ef04` rebuilt untouched | **112 tok, finish=stop, 0 faults** (twice) |
| + `MoeDev.need_seq` field + 64-byte pinned allocation + **device** write in `moe_route_kernel` | **fails**: `prompt layer 20 invalid expert ID or non-finite score (105 of 189 routes)` |
| + field + allocation + **service-side** write (`need_seq_h[rq->ch] = next`), device write removed | **fails**, identical layer 20 |
| + field + allocation, **no writes at all** | **112 tok, finish=stop, 0 faults** |

So: **the pinned allocation and the new field are inert and safe; each of the two *writers* breaks the model on its
own.** Aaron's guess (the d2h write in the shared route kernel) was right about the first, and the same failure
survives when the write is moved to the host, which means it is not the d2h transfer - it is the **store itself**, and
the mechanism is unexplained. Writing one 4-byte seq into that pinned block, from either side, ends at layer 20 with
non-finite router scores.

Unresolved hypotheses (cheap to test, in this order):
1. the pinned block aliases something live (allocate 4 KiB with a guard, or `memcpy` from a stack copy, and see
   whether the mere store still breaks it);
2. `rq->ch` is not what we think for some route (a `MoeRequest` written by the older publish path) - the store would
   then land on the wrong channel word, which is harmless in itself, unless the *value* read back by something else
   matters;
3. the store perturbs timing enough to expose a pre-existing layer-20 race. (Layer 20 is suspiciously specific and
   appears in every failing build.)

**Consequence:** `need_seq` stays field+allocation only - no writer - until (1)-(3) are separated. The readiness
primitive is still unproven, so per the spec both the demotion ladder and the per-lane thread model stay blocked on
it. That is the honest state, and it is one instrumented run away from being answerable.

### CORRECTION to the "four builds" table above (same day, later)

The table said the store was the trigger. **That is withdrawn.** Two later builds disprove it:

- A build with the service write present **plus** a "first store" print and an out-of-range counter printed
  **neither** line - the store never executed - and it failed anyway, at a **different layer** (`layer 11`,
  63 of 189 routes) than the earlier failures (`layer 20`, 105 of 189).
- The committed build (`2b1d1c4`, field + allocation, no writers) then ran **6/6 clean**, 0 nan-guard lines.

So: the failing layer moves between builds and runs (11 vs 20), the store that "caused" it never ran in one failing
build, and the only differences in that build were two function-local `std::atomic`s and one `fprintf` in the service
loop. The instrumentation perturbed the outcome. **This is a Heisenbug, and single-run comparisons across builds are
not admissible for it** - every statement above that rests on one run per build is now suspect, including the
"130ef04 is clean" baseline (n=2).

What survives:

- the committed build has been clean 6/6 on identical prompts;
- the failure, when it happens, is a **prefill** non-finite at a variable layer with a large fraction of invalid
  routes (63-105 of 189), i.e. a whole layer's routing going bad, not one expert;
- it is **not** spatial (the layer moves), **not** the index (both store sites are masked to `[0,8)`), and **not** the
  store (it failed without one).

Method rule for whoever picks this up: **n >= 5 runs per build, and report the count**, or the comparison is noise.
This is the same lesson as the token-equality retraction, in a different place.

### Baseline flake rate (2026-10-29, `1d45044`, no rebuild, n = 12)

12 identical-shape prompts, each with a unique "(variant k)" suffix so **prefill is not cached** (identical prompts
would reuse the KV and never re-exercise the failing path):

    clean 12 / 12      nan-guard lines added: 0      tokens 80/80/80/80/80/80/80/80/80/80/80/80

So the committed build does **not** flake in this sample. Rule of three: with 0/12 the 95% upper bound on the per-run
failure rate is ~22%, which is not tight enough to call it zero - but it is enough to interpret the earlier builds.

**Re-reading the earlier builds against this baseline.** Counts of *clean runs observed* (this is the honest way to
state it, since several were n=1):

| build | observed | implication |
|---|---|---|
| field + allocation, no writer | **19 / 19 clean** (n=2, 1, 6, 12 across four boots) | consistent with the baseline |
| any build with a *writer* present | **0 / 4 clean** | at a 22% upper-bound per-run rate, 4/4 failures have p < 0.3% - the writers are implicated |

But the mechanism is **not** "the store executes and stomps memory": build 5 carried the writer *and* a first-store
counter, printed neither, and failed anyway. What every failing build had that the clean ones lack is the
**8-byte bigger `MoeDev`** (the field) *plus* a build in which something touches it. Builds 4 and 6 have the field and
are clean, so the field alone is not it either.

That leaves the two classes the instrumentation can distinguish, and neither has been ruled out:

1. **the store executes and perturbs timing** (the field is 8 bytes, the block is 16 sectors from the ring) - build 5
   would then be an instance where the store ran without its print, not one where it didn't run;
2. **the presence of the writer changes the service loop's code enough to reorder the handshake** - the same class as
   the fprintf finding, one step weaker.

Both are timing, not spatial. The n=12 baseline is what makes that statement testable: a writer build at n >= 12 either
shows ~0/12 (the writer is innocent and the earlier 4/4 was something else) or roughly 12/12 (it is decisive).

### Method rules this bought (both were already paid for once)

- **n >= 5 per build, report the count.** One run per build is noise; the four-build table was withdrawn for this.
- **No I/O inside the service loop.** Two function-local atomics are nearly free; an `fprintf` is a syscall and a lock,
  and it reordered a race by enough to change the layer from 20 to 11. Counters only, one dump at the end of a run.
- **Vary the prompt when measuring prefill.** Identical prompts reuse the KV and skip the path under test.

### The disk tier: the 14x was mine, and the counter closes the item (2026-10-29)

**Correction to the justification in commit `2ccbc2f` and to the spec.** The 14x compared CPU *compute* (0.214 ms)
against an NVMe *transfer* (2.98 ms) - but a disk-only expert has to come off disk whichever device computes it:

    GPU path:  disk->host 2.98 + host->VRAM 0.188 + GPU gemv ~0.01  =  ~3.18 ms
    CPU path:  disk->host 2.98 + CPU compute 0.214                  =  ~3.19 ms

**Dead even.** The two figures are only substitutes for a RAM-tier expert (0.214 vs 0.188, parity), and the lane
already takes those. So `STRATA_GLM_MISS_TO_CPU` is **not** a latency win and must not be sold as one.

What survives is second-order and is the part worth keeping: with the rule on, a disk-only expert takes **no spare and
no fetch**, so it does not evict a resident. Admitted to VRAM it costs a future miss, and hit rate is what decode is
most sensitive to. That is **pool preservation**, not first-order latency.

**Measured (production build, full pool, 200-token decode, slot dir cleared):**

    disk = 0        ram_fetch = 19.3 GB/s      vram_hit = 0.779      promo = 19.7

No disk reads during decode. The RAM tier (78 of ~110 GB) absorbs the misses; disk-only misses do not appear in this
regime at all. By the rule set for this item - "if disk-only misses are a small fraction of all misses it is noise and
closes" - **the item closes.** The disk->host read in the lane is not worth its build, and `MISS_TO_CPU` stays a
default-off knob whose only argument is pool preservation on a path we have not observed.

**The implication for the main line**, which is the useful part: if a disk read is unavoidable once the expert is
needed, no lane choice takes the SSD tier off the critical path. The only levers are fetching it earlier (prefetch
depth - the scheduler, which needs the primitive) or not needing it (lossy skip). **The disk tier is a scheduler
problem**, and the spec's "N = 5-8 of cover to hide a 2.98 ms miss" is the honest statement of what that costs.

### Fix: `-1` was overloaded, and it disabled the guard during spin-up (`f789a85` -> this commit)

The publish was `floor = l0_ + k - (n-1)` with the guard `if (pipe_f >= 0 && il >= pipe_f)`. At k = 0 with n = 3 that is
**-2**, so the guard's `>= 0` test failed and **every layer stayed evictable** while slots were live at layers
0..n-2 - the invariant was inert exactly when the slots are most tightly packed, because "no pipeline" and "the floor
arithmetic is still below the first layer" shared a sentinel.

Fixed: `pipe_active` is its own atomic, `floor` is clamped to `std::max(l0_, l0_ + k - (n-1))`, and the boundary tests
`pipe_on && il >= pipe_f`. The flag is cleared on **both** exits of the rotor, including the error return.

**Consequence, and it is the ladder's first case.** With the clamp, the early rounds have *no* evictable layers at all -
`il < floor` is empty at k = 0, and with n = 2 it stays empty through k = 1. That is semantically right (only layers
this step has finished with may be given up) but it means a miss in those rounds has no spare and no eviction
candidate. **That must demote, not fail.** So the ladder's first case is "no spare available -> demote", and its
acceptance test is sharper than the tiny-pool run: **force a miss at layer 0 with a full pool and confirm it demotes.**

**`S = n-1` is right, not a fudge** - the rotor's skew *is* the lead, so the cover window and the invariant's slack are
the same quantity by construction. The pin set is therefore `S x N x 8` slots = **1 x 2 x 8 x 8.80 MB = 141 MB at n = 2**
(0.45% of the 31.17 GB pool), not the 281 MB the spec originally estimated.

## 2026-10-30 — the overlapped arm works, and its first measurement

The de-fused rotor now produces correct full-length output with two rows overlapped (acd52d1): 30+30 tokens,
both requests ok, coherent text, and the fused path unchanged on the same binary (40 tokens, 2.5 s, 0 errors).

Root cause of the whole tail of the hunt: the per-row `MoeDev` (plan pointers, fetch_src, cpu_part, cpu_flag,
scratch, wait_ch, cpu_seq, cpu_ans) was built in `rows_moe_grouped`'s `row_dev`, and the path the rotor actually calls
is `rows_moe -> fast_moe_experts`, which started from `F->md`.  One unreachable builder produced: five null/offset
pointer bugs, a dead `transfers_done`, every CPU-lane route presenting ch=0, and the "row lane did not drain" stall.

### Measurement (interleaved boots, NOT one boot)

The arm is chosen at process start (`static const bool serial = getenv(...)`), so serial and overlap cannot be
interleaved inside one process without an engine-side runtime switch that does not exist yet.  Design used instead:
alternating boots S,O,S,O,S,O, two different prompts per boot (32 tokens each), first pair discarded.

| run | arm     | tokens | seconds | tok/s |
|-----|---------|--------|---------|-------|
| 1   | serial  | 64     | 4.69    | 13.6  |
| 2   | overlap | 64     | 4.14    | 15.5  |
| 3   | serial  | 64     | 4.46    | 14.3  |
| 4   | overlap | 64     | 4.09    | 15.6  |
| 5   | serial  | 64     | 4.45    | 14.4  |
| 6   | overlap | 64     | 4.11    | 15.6  |

Valid runs (first pair discarded): serial n=2, 14.4 tok/s; overlap n=2, 15.6 tok/s — about +8%, tight and consistent,
overlap 15.5-15.6 against serial 13.6-14.4.

### What that number is and is not

The copy is still issued on the compute stream; TRANSFER records an event but does not own the transfer.  So this
measures de-fused SCHEDULING without copy/compute overlap — a small win is the honest expectation, and ~0% would not
have been a failure of the design either.  The spec's copy-stream move is the change that would produce a real
overlap win, and it is still outstanding.

## 2026-10-30 (later) — the runtime arm switch, and what the old method was hiding

The arm was `static const bool serial = getenv(...)`, fixed at process start, so an A/B inside one boot was impossible.
It is now a one-character file read at most every 200 ms ("s" = serial, anything else = overlap), outside the service
loop.

### ONE boot, alternating arm per measurement (both prompts every time, 32 tokens each)

| run | arm     | tokens | sec  | tok/s |
|-----|---------|--------|------|-------|
| 1   | overlap | 64     | 4.15 | 15.4  |  (cold)
| 2   | serial  | 64     | 2.30 | 27.8  |  (discarded)
| 3   | overlap | 64     | 1.92 | 33.3  |
| 4   | serial  | 64     | 2.04 | 31.4  |
| 5   | overlap | 64     | 1.85 | 34.6  |
| 6   | serial  | 64     | 2.15 | 29.8  |

Valid (first pair discarded): serial n=2 **30.5 tok/s** (31.4, 29.8); overlap n=2 **34.0 tok/s** (33.3, 34.6).

### The cross-boot measurement was not just noisy - it was wrong by 2.3x

The earlier alternating-boot design reported 14.4 / 15.6 tok/s for the same arms and the same work.  Every one of those
runs paid a cold start inside its own measurement; inside a single warm process the same code runs at 30/34.  The delta
also changed: +8 % cross-boot, **+11 % in-boot**.  Both the level and the comparison were distorted, which is exactly
what "measure inside one boot" was meant to prevent.

### Still true

The copy is issued on the compute stream, so this is scheduling overlap only; the +11 % is in the range that argument
predicts, and the copy-stream move is the change that would show a real overlap win.

## 2026-10-30 (baseline) — n>=6 per arm, one boot, full distribution

One process, arm switched by file between measurements, the same two prompts (32 tokens each) every time, first pair
discarded.

| run | arm     | tok | sec  | tok/s |
|-----|---------|-----|------|-------|
| 1   | overlap | 64  | 4.40 | 14.56 |  (cold)
| 2   | serial  | 64  | 2.38 | 26.94 |  (discarded)
| 3   | overlap | 64  | 2.03 | 31.54 |
| 4   | serial  | 64  | 2.11 | 30.31 |
| 5   | overlap | 64  | 1.92 | 33.37 |
| 6   | serial  | 64  | 1.98 | 32.37 |
| 7   | overlap | 64  | 1.86 | 34.48 |
| 8   | serial  | 64  | 1.92 | 33.31 |
| 9   | overlap | 64  | 1.83 | 35.07 |
| 10  | serial  | 64  | 1.95 | 32.86 |
| 11  | overlap | 64  | 1.92 | 33.38 |
| 12  | serial  | 64  | 2.09 | 30.68 |
| 13  | overlap | 64  | 2.02 | 31.75 |
| 14  | serial  | 64  | 2.04 | 31.36 |

Distribution (valid runs, n=6 each):

    serial   min 30.31  q1 30.68  median 31.36  q3 32.37  p95 33.31  max 33.31  mean 31.82  sd 1.11
    overlap  min 31.54  q1 31.75  median 33.37  q3 33.38  p95 35.07  max 35.07  mean 33.26  sd 1.29

Marginals overlap heavily (serial p95 33.31 vs overlap min 31.54), so the honest statement is the PAIRED one, adjacent
runs in the same boot: +1.23, +1.00, +1.17, +2.21, +2.70, +0.39 tok/s - overlap faster in 6 of 6 pairs, median paired
delta about +1.2 tok/s (~+4%).  The n=2 reading of +11% was optimistic; the direction survives, the magnitude halves.

Baseline to judge the copy move against: **serial median 31.36, overlap median 33.37, paired +1.2 (+4%), n=6.**

## 2026-10-30 (copy lane) — the fetch moved to the copy stream, and it is slower

`STRATA_GLM_COPY_LANE=1` issues a per-row step's fetch on the copy stream (`F->ps`), ordered by one event a layer on
each side (plan written -> copy -> experts read).  Correct: 0 non-finite, 0 drain errors, coherent output.

Same method as the baseline: one boot, alternating arm, n=6 valid each, paired deltas.

    serial   min 26.91  median 28.86  max 29.53  mean 28.66      (baseline: median 31.36, mean 31.82)
    overlap  min 29.15  median 29.76  max 32.89  mean 30.88      (baseline: median 33.37, mean 33.26)

    paired deltas (overlap - serial): +0.82 +4.27 +2.61 +2.66 +0.04 +2.92   6/6 positive, median ~ +2.6

Both arms are ~8% SLOWER in absolute terms, while the row-vs-row margin widened from +1.2 to +2.6 tok/s (6/6 both
times).  Read: the copy lane serialises the two rows' fetches through one shared stream and adds two event round trips
per layer, and the overlap it buys back is smaller than that cost.  The spec's hoped-for copy/compute win did NOT
appear on this design - a null-to-negative result, which was a legitimate outcome given the copy had to share `F->ps`.

Default OFF; the code stays for the next attempt, which needs a copy lane per row (or a queue with priorities) rather
than one shared stream.

## 2026-10-30 (miss accounting) — what the copy lane would actually have to hide

Before building a per-row copy lane, the workload it serves:

    tiers warm: 12096 experts (114.0 GB)        = 42 MoE layers x 288 experts, ALL resident - but 114 GB
                                                  against ~48 GB of VRAM, so most of them are in the RAM tier
    expert cache hit: 71.0% / 70.9%             -> ~29% of routed experts are NOT in VRAM and cross PCIe
    engine's own figures: 0.567 ms per expert over PCIe; 114.0 GB / 12096 = 9.42 MB per expert
    geometry: 42 MoE layers, 8 experts per route

Per layer, per row, per token:

    8 x 0.29                       = 2.3 expert misses
    2.3 x 9.42 MB                  = ~22 MB over PCIe
    2.3 x 0.567 ms                 = ~1.3 ms of copy

And the layer's own wall time: 32 tokens per row in 4.05 s = 127 ms a token, over 42 MoE layers = ~3.0 ms a layer.

So the copy is roughly **40-45% of a layer's time**, and it is structural rather than incidental: 114 GB of experts
against 48 GB of VRAM means a large minority can never be resident at once.  The "the pool fits, so there is nothing to
fetch" hypothesis is false on this box - the pool fits in memory, not in VRAM.

Consequence: a per-row copy lane has real latency to hide (~1.3 ms a layer), so it is worth building; the shared copy
stream was the wrong shape, not a solution to a non-problem.

## 2026-10-30 (counted misses + per-row copy lane) — the lane does not pay

### Counted, not derived (STRATA_GLM_MISS_LOG=1, per step, both rows)

    MISS step 430: 140 experts over PCIe, 1417.33 MB  (3.11 experts, 31.50 MB a layer)
    MISS step 431:  94 experts,  948.22 MB            (2.09 experts, 21.07 MB a layer)
    MISS step 432: 117 experts, 1212.96 MB            (2.60 experts, 26.95 MB a layer)
    MISS step 433: 108 experts, 1074.63 MB            (2.40 experts, 23.88 MB a layer)
    MISS step 434:  88 experts,  884.41 MB            (1.96 experts, 19.65 MB a layer)

So ~2-3 experts and 20-31 MB per layer across BOTH rows: about 10-16 MB and 1-1.5 experts per row per layer.  The
derivation from the hit rate (2.3 experts / 22 MB per row) was about 2x high, and the hit rate itself moved 71% -> 83.5%
between requests, which is exactly why it should not be the figure of record.

### Per-row copy lane (a copy stream each, STRATA_GLM_COPY_LANE=1)

    paired deltas: +0.66 +0.96 -0.07 -0.19 -0.59 +0.51     3/6 positive, median ~ +0.24
    serial  n=6  median 30.95      overlap n=6  median 31.62
    baseline for comparison:       serial 31.36, overlap 33.37, paired +1.2

The per-row lane removes the shared-stream serialisation and the advantage still disappears: overlap is ~5% slower than
its own baseline and the paired delta collapses from +1.2 to ~0.  Reading: the two rows ALREADY hide each other's PCIe
latency in the baseline; moving the copy to a separate stream adds two event round trips a layer and buys back less than
it costs.  The row-vs-row overlap was doing this job all along.

Default off.  The copy lane would need a workload where the rows do not already cover each other - the disk-tier case
this design kept in - or a fetch that is a large fraction of a layer, which 1-1.5 experts/row is not.

## 2026-10-30 (gate) — where opening the rotation actually stands

1. The certification gate is retired: `STRATA_GLM_PIPELINE_UNCERTIFIED` is gone with its comment and warning, because its
   stated reason (shared per-row spare claims) was the per-row MoeDev bug and `floor_violations` / `spare_steals` read
   zero.  `pipe_mode` now defaults to 2 via `glmfast::layer_pipe_mode()`.

2. The rotation still cannot run on the default config: it needs the per-row DSA buffers, which are behind
   `STRATA_GLM_ROWS_ATTN`.  Making that flag follow the rotation, or allocating the buffers without the flag, both crash
   with illegal accesses in `mv` / `mla_q_abs` / `dsa_select_all` / `kv stream resolve` / `kv_q4`, because the flag is
   consulted OUTSIDE the rotor (prefill, the fused path).  That refactor is parked: 22 MB of the rows' 31 MB and a
   changed attention shape is a VRAM decision, not a cleanup.

3. With `STRATA_GLM_ROWS_ATTN=1` delivered from the config's own `env` block (live config is `maya-v3q-q4.json`, the one
   the launcher names), the engine's announcement confirms the gate opens: no 'rotation declined', the DSA line, and
   'rotor entered (n=2, layers 0..45)'.  That boot then died with the same illegal accesses.

4. Two hypotheses for that death were tested and FALSIFIED:
   - **k = 7 interaction**: shell `ROWS_ATTN=1 LAYER_PIPE=2 MAYA_GLM_EXPERT_USED=7` runs clean - 32+32 tokens, 4.4 s,
     0 errors, rotor on steps 1 and 2.
   - **config-env vs shell-env delivery**: `child_env()` is `dict(os.environ)` updated by `cfg["env"]`, and both spawn
     paths pass it (`Popen(..., env=env)`), so the two boots provably delivered the same values.

5. Remaining hypothesis, and it fits the engine's own last words ("exit code 1 ... the usual cause is running out of
   RAM"): the machine was downloading 156 GB and hashing it (Maya-L) at the time.  A repeatability run of four
   identical boots is in progress; if any of them crashes under an idle box, load is not the explanation either.

### Gate, continued: the difference is the CONFIG PATH, not the flag, the alias, k, or load

Measured, box idle, two concurrent requests every time:

| delivery | result |
|---|---|
| shell `STRATA_GLM_ROWS_ATTN=1` + `LAYER_PIPE=2` | works, 6+ boots, 32+32 / 24+24 tokens, 0 errors |
| shell `MAYA_GLM_ROWS_ATTN=1` (alias only) + `LAYER_PIPE=2` | works, 48 tokens, 0 errors, rotor on steps 1-2 |
| config `env` carrying the flag (server rewrites the key to `MAYA_GLM_ROWS_ATTN`) | **crashes 3/3**: illegal access in mv / mla_q_abs / dsa_select_all / kv_q4, rotor engaged first |

So: not the flag's value, not the `MAYA_`/`STRATA_` name (both work from the shell), not k=7, and not machine load
(the box was idle and the download finished).  The engine's `rows_attn_on()` is env-only and reads the same value in
both.  What remains is the config path itself - the ordering of the config's env relative to engine startup, or a key
that only exists on that path.

A `[engine env]` dump was added to `child_env()` in serve/server.py (sorted MAYA_/STRATA_ pairs, to stderr) but did not
land in maya-v3q-q4.log, so it must be written to a file before it can be diffed.  That is the next step: one file per
boot, diff the two.

#### The crash explained: a default honoured by ONE of four readers

`STRATA_GLM_LAYER_PIPE` had four read sites and `eb41f73`'s default-to-2 reached only one of them:

| site | read | unset meant |
|---|---|---|
| `glm_memory.hpp` `layer_pipe_mode()` | mode, default 2 | **2** - rotor engages |
| `glm_fast_path.cu:669` local lambda | mode, default 0 | 0 - **no PipeRuntime** |
| `glm_fast_path.cu:1067` `pipe_scratch` | presence (`!= nullptr`) | false - **no rows_scr** |
| `glm_rows.cu:221` `pipe_flag_on` | presence + >= 2 | false - ladder assertion never runs |

So with the variable UNSET the rotor ran with no per-row scratch and no runtime: the route kernel's
`ptr = a.d.scratch[i]` read address `8*i`, i.e. garbage slots for every row but row 0 - the illegal access in
mv / mla_q_abs / dsa_select_all / kv_q4, and it needs a non-row-0 row, which is why n=1 never crashed.

Shell runs set the variable, so all four agreed. Config runs relied on the new default, so one said yes and three said
no. **Delivery was never the variable; presence was.** The test matrix had no cell for "unset, relying on the default".

All four now read `glmfast::layer_pipe_mode()`. Measured after: config-only delivery, ZERO shell keys, two concurrent
requests - **2/2 boots clean, 64 tokens each, 0 errors, rotor on 31 steps, no declines** (before: 3/3 crashes).

**The shape, for the record:** this is the same class as the de-fused root cause. There a per-row structure was built
by only one of two paths; here a default was honoured by only one of four readers. A default is a contract, and
introducing one means auditing every reader of the thing it defaults.

#### Soak with the rotation engaged (same boot, config-only delivery)

Six sequential 256-token turns first (each is n=1, so the rotation declines by design): 256 tokens each, 10.4-12.5 s,
0 errors.

Then three rounds of TWO CONCURRENT 256-token turns, which is what actually engages the rotor:

| round | tokens | wall | aggregate |
|---|---|---|---|
| pair 1 | 512 | 22.0 s | 23.3 tok/s |
| pair 2 | 512 | 22.7 s | 22.6 tok/s |
| pair 3 | 512 | 20.8 s | 24.6 tok/s |

796 rotor entries across the session, **0 stalls, 0 drain errors, 0 non-finite, 0 illegal accesses.**  Long turns under
concurrent load are the shape that would surface the stall class, and none appeared.

Gate status: the rotation is ON by default - `layer_pipe_mode()` defaults to 2 and the live config carries
`MAYA_GLM_ROWS_ATTN=1` - and a boot with zero shell keys engages it and stays clean.  The per-row-DSA-everywhere change
stays parked (22 MB of the rows' 31 MB, changes the attention shape, needs its own measurement).
