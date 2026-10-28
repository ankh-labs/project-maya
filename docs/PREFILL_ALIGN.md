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

## The decode side: interleaving the two phases is NOT viable here (corrected)

Decode is already aligned within its own phase: `multi_rows` takes every slot with a token pending and steps them in
one launch, one token each, so slots share the expert fetch and the mixer pass with no scheduling work.

An earlier revision of this file proposed slicing prefill and running `multi_rows` between slices so a long prompt
would not starve other slots' decode. **That is wrong for this engine** (user, 2026-10-28): the two phases have
incompatible VRAM layouts by design.

- **Prefill** borrows ~90% of the pool for the chunk's work buffers, and the borrow **evicts the expert tier** to get
  the space - that is what `PrefillState::dropped` counts, and it is why no residency survives a prefill (measured:
  three near-identical prompts cost 11.10 / 11.55 / 11.44 GB, never less).
- **Decode** needs the opposite layout: the expert tier resident in VRAM so a step reads weights from VRAM rather
  than PCIe.

Interleaving them would evict on every slice boundary and reload for every decode step - the eviction/reload cycle
would cost far more than the stall it tries to hide. So the phases stay separated, and alignment belongs **inside** a
phase: prefill here (`prefill_align_plan`), decode via `multi_rows` as it already does.

Consequence: a long prefill still stalls other slots' decode for its duration. The mitigations are (a) fewer and
shorter prefills - the slot cache, which removes prefill entirely on a hit - and (b) this alignment, which stops two
slots from paying two sweeps. Not slicing.

## Measured under real two-agent load (2026-10-28): chunks are already full, so alignment cannot help long prompts

Sampled `/metrics` every ~0.9 s for 120 s while two agents ran:

- **`active = 2` in 94% of samples** (125/133), `active = 1` in 6%; **`queued` was 0 throughout** - with
  `--parallel 2` and two clients, the queue never forms and both slots are always busy.
- **`state = reading` in 133/133 samples**: the engine spent the whole window in prefill, not decode.
- Only 6 prefill chunks in 120 s, so ~20 s each. One prompt was **7 chunks of 32,768 tokens** (6 x 32,768 +
  11,847 ~ 208k tokens), each chunk **17-19 s and ~19 GB of expert traffic**.

That last number is the decision-relevant one. The chunk limit is **32,768 tokens** (`--prefill auto, up to 32768`),
and both agents' chunks are **already at it**. So the proportional-share alignment cannot merge them: two 32k prompts
would need a 64k chunk. Alignment only pays when chunks are **under-filled** - which is the small-prompt case
(measured 1.80x at 2 x 1.6k). For the long-context agent workload actually running, the gain would be just the
rounding waste: 416k tokens of prompt is 13 chunks aligned versus 14 separate (~7%), not 1.8x.

### The lever the data actually points at: chunk size, not alignment

Expert bytes per chunk grow only slowly with tokens (17.16 GB at 19.4k, ~19 GB at 32.8k), so **a larger chunk amortises
the sweep over more tokens**. Halving the number of chunks for a 208k prompt (14 -> 7) would nearly halve its total
expert traffic (~270 GB -> ~140 GB for the pair). Whether that is available depends on whether the 32,768 cap is a
constant or a memory-derived layout limit: `prefill_carve` refuses a chunk whose layout does not fit the lendable
region, and `--prefill N` / `STRATA_GLM_PREFILL_CHUNK` set the request. **Test `--prefill 65536` before building
anything else** - it is a config change, and the engine says so itself when a layout does not fit.
