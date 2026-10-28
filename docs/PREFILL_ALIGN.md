# Prefill alignment — design (STRATA_GLM_PREFILL_ALIGN)

Status: **design + batch-plan core implemented; the model-path work is not.** Flag off by default; nothing in the
engine's behaviour changes when it is off.

## Why (measured, report 9y in `G:\AI\llama-server\docs\reports\MAYA_V3Q_SESSION2_2026-10-27.md`)

- The expert sweep is a **fixed cost per prefill chunk**, not per token: the same text at 1x / 2x / 3x tokens cost
  12.22 / 11.52 / 12.09 GB over PCIe, and wall grew only 355 ms a copy.
- Bytes stay flat to ~5k tokens and climb by ~17k (12.1 GB at 4.8k, 17.2 GB at 19.4k), so a batch target of
  4-8k sits in the flat region.
- Two distinct prompts prefilled **separately** cost 23.01 GB / 5,512 ms; the same two in **one chunk** cost
  10.94 GB / 3,057 ms — **1.80x**, because one chunk means one sweep *and* one mixer pass.
- There is **no retention across chunks**: three near-identical prompts as three requests cost 11.10 / 11.55 /
  11.44 GB. The lend (`prefill_carve`, borrowing ~90% of the pool's slots for the chunk's work buffers) **evicts the
  VRAM expert tier** to get the space (`PrefillState::dropped` counts it), so every prefill starts cold again.

So the cheapest schedule is the one with the fewest chunks, and the fairest is the one where no slot is left with a
private, ragged tail batch. Proportional shares give both.

## The batch plan (implemented here)

For slots with prompt lengths `p_s` and a batch target `b`:

    share_s = round(b * p_s / sum(p))      // at least 1 token while that slot has work left

Every batch carries each slot's proportional share, so all slots finish on the last batch and no tail batch exists.
Number of chunks is `ceil(sum(p) / b)` — the theoretical minimum for a given `b`.

`prefill_align_plan()` in `src/core/glm_prefill.cu` implements this and is logged at startup when the flag is on, so
the schedule can be inspected before any model-path change lands.

## What remains (the model-path work)

`Glm5Model::prefill` is single-sequence: everything is addressed by **chunk position** (`pos_ + i`) into the current
sequence's KV/pool buffers. To align slots, a chunk must carry **per-row `(seq, kv base, position)`**, which touches:

1. **KV / latent write** — currently one base pointer + `pos_`; needs a per-row target. Simplest correct shape: an
   array of per-row `(base, pos)` in device memory (or a per-row sequence id plus a table of per-sequence bases).
2. **Attention window** — the DSA/indexer window must start at the row's own sequence start, not the chunk start,
   so slot B never attends into slot A's tokens. `rows_dsa` already does per-row windows for decode; that is the
   pattern to reuse.
3. **Pool commit / rollback** — the commit is per sequence at the end of the chunk, and the rollback on a failed
   chunk must rewind every sequence in the chunk.
4. **The caller** (`generate.cpp`, the prefill issue site) — collect the requests waiting at the prompt stage, order
   them by arrival, compute the plan, and submit one aligned chunk while the total fits `Tmax`.

The MoE is already row-based (`MoeBufs.row_tok`, `MoeBufs.pos`), so it needs no structural change — it simply gets
rows from more than one sequence, which is what its row arrays already express.

## Acceptance test (do not ship without it)

Two distinct ~5k prompts, engine idle, fresh text each arm:

| arm | expected pcie | expected wall |
|---|---|---|
| separate | ~23 GB | ~5.5 s |
| aligned (flag on) | ~11 GB | ~3.1 s |

Plus: single-request prefill must be **byte-identical** to the flag-off path (same chunk line bytes and wall within
noise), and the battery must stay 7/7. Any deviation on the single-request path means the flag is not gated correctly.

## The decode side of the same idea (not implemented)

Decode is already aligned by construction: `multi_rows` takes every slot that has a token pending and steps them in
one launch, one token each, so slots share the expert fetch and the mixer pass without any scheduling work.

What decode *is* missing is protection from prefill: while one slot streams a 5k prompt, every other slot's decode
stalls for seconds, because prefill is issued as one long call. The same batch target `b` fixes this: issue prefill in
slices of `b` tokens and call `multi_rows` between slices, so the other slots keep generating at ~21 tok/s while a
long prompt streams. The slice size that minimises expert sweeps (4-8k, the flat part of the bytes curve) is also
small enough that a decode step between slices is a cheap interruption - the two payoffs come from the same knob.

Order of work: the alignment itself needs the per-row (seq, base, pos) work above; prefill slicing needs only a call
site change (a slice boundary is a place to call `multi_rows`), so it is the easier of the two and independently
useful.
