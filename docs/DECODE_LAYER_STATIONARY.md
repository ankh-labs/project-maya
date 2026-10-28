# Assessment: layer-stationary out-of-core MoE (user proposal, 2026-10-28)

Your proposal in brief: decode is transfer-bound, so throughput is bytes moved; two aligned slots need the *union*
of their experts per layer, not the sum, so overlap ≈ 2x tokens for the same transfer; synchronising the slots costs
almost nothing because one layer's compute stall is trivial against the transfer it removes. Then: three tiers with
layer-aware (not global LRU) eviction, a layer-stationary schedule with pipelined advancement or a barrier, a dynamic
switch to whole-layer sequential loads, progressive routing discovery from layer L's router for L+1, and finally the
prerequisite: measure per-layer frequency/union to know whether skew makes any of this pay.

## Where our measurements agree with it

- **Transfer-bound decode.** GPU 98% util at **154 W of a 300 W cap**; expert copies run at 22-25 GB/s; prefill's lane
  reports `busy` as time spent copying. For a 181-token prompt, **48 GB** crossed PCIe. Throughput tracked bytes, not
  FLOPs, in every measurement this session.
- **Sync cost is noise.** `ring wait` was 1-232 ms against 2-19 s walls; the lane is idle 71% during prefill windows.
  Waiting one slot behind another for a layer is indeed cheaper than the transfer it avoids.
- **Layer-stationary is what prefill already does.** Its MoE runs over the *whole chunk* per layer ("every expert's
  weights are read once per chunk"), which is why bigger chunks amortise: 53,507 tokens as one chunk cost 21.4 GB /
  24.4 s, against 2 chunks at ~36 GB / ~35 s.

## Where the engine is already doing it

- **Three tiers exist**: VRAM pool **36.26 GB / 4,104 slots**, pinned RAM **86-98 GB / 9,254-9,842 slots**, disk as
  backstop. That is 12,096 experts placed across hot and warm, with the disk tail.
- **Layer-aware residency exists**: `--expert-profile` (46 layers x 288 experts, ranked by real usage, with per-layer
  floors - `glm_fast_path.cu:1272-1349`), and a live LFU on top. So "rank by per-layer execution frequency, not global
  LRU" is already the design, not a proposal.
- **Progressive routing discovery exists**: prefill reports `prestaged 6560` a chunk - L+1's experts are fetched from
  L's routing while L computes, and `STRATA_GLM_PREFETCH_AT` chooses when to start that copy.

## Where it would be new work, and the honest costs

1. **Decode is only partially stationary.** `multi_rows` steps one token per slot per layer (batched, so one launch),
   but slots are *not* held at a layer; they advance together within the step. Making it stationary across many slots
   needs per-slot activations held simultaneously - the rows arena is sized `rows_max()` = **8 rows**, and the arena
   scales with rows (x, mixer, ffn, q8 buffers each). 8 slots is the cap today.
2. **The union-vs-sum win is a function of routing skew, which nobody has measured here.** With 288 experts a layer
   and 7 picks a token, two tokens' union is at most 14 and at least 7. If picks are diffuse, union ≈ sum - the
   proposal's own §5 says it: "under uniform routing, four aligned slots reduce fetches only marginally".
3. **Whole-layer sequential loads** are a real option but the price is fixed: a layer is 288 experts x ~9 MB =
   **~2.6 GB**, regardless of how few are needed. That only wins if the union is large - again, skew.
4. **Pipelined advancement vs barrier** needs room for 2-3 layers resident (~5-8 GB). VRAM has **~1.2 GB free** with
   the current pool, so this competes directly with pool size: every layer prefetched is ~150 experts displaced.

## The measurement to do first (tooling already in the tree)

- `--dump-routing <file>` writes the routing trace (`STRATA_GLM_TRACE`): one "layer expert" line per routed expert.
- `STRATA_GLM_USAGE=<file>` writes per-layer, per-expert **counts** since start.
- `tools/glm_expert_prior.py <pack> <trace|usage>...` already consumes both and emits per-layer frequency order.

Plan: one realistic session (a couple of agents, their normal turns) with `STRATA_GLM_USAGE` pointed at a fresh file,
then compute per layer:

1. **concentration** - what fraction of picks the top 8 / 16 / 32 / 64 experts carry;
2. **union growth** - for k = 1..8 token sets drawn from the trace, mean |union| against k x 7 (the no-skew case);
3. **residency fit** - how many of the top experts per layer would fit a 2-3 layer window (~4-8 GB).

Decision rule, agreed in advance so the answer is not negotiable after the fact:

- If mean |union| for 2 slots is **< ~10 of 14** (strong skew), layer-stationary decode with union fetches is worth
  building - the expected win is up to ~1.8x on decode, matching what alignment measured on prefill.
- If it is **~13-14 of 14** (diffuse), the union trick is worth little; the effort belongs in the slot-cache bug
  found in 9z (conversations not being saved, 81k-208k tokens re-prefilled per turn) and in chunk/tier sizing.
