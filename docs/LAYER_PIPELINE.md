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
