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

