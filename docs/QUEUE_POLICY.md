# Admission and queue policy (design)

Companion to `docs/PREFILL_ALIGN.md`. Status: **design only, nothing implemented.** Every number below is measured
(report `G:\AI\llama-server\docs\reports\MAYA_V3Q_SESSION2_2026-10-27.md`, sections 9v–9y).

## What happens today

- `serve/server.py` accepts every request and puts it behind a **FIFO `FairLock`** with a queue watchdog; the module
  doc says "one sequence at a time behind a ...". So a second client **waits**, it is not rejected.
- The engine has `STRATA_GLM_SEQS = --parallel N` sequence states (our config: **2**). Several requests decode at once
  only when `model.seq_count() > 1` (`generate.cpp:2558`).
- **Prefill happens inside `gen_admit`, one request at a time.** This is why two simultaneous prompts cost
  6.08 s ≈ 2 × 3.03 s: their sweeps are serialised even though their decode steps batch.

## The measured inputs a policy can use

| input | where it comes from | measured value |
|---|---|---|
| prefill cost of a request | chunk line, fitted (9w) | `2.16 s + 0.33 ms/token` |
| expert sweep per prefill | chunk line `pcie` | **fixed per chunk**: 12.22 / 11.52 / 12.09 GB at 1x/2x/3x tokens |
| how much of the prompt is a hit | slot line `took back a N-token conversation; K new tokens to read`, and `live.prompt_reused` | per request, live |
| the nearly-free case | `prefill_min_n` (`glm_prefill.cu:4231`) | **< 8 new tokens → prefill declines and the token path runs**, paying no sweep |
| cost of promoting a second request now | derived | one sweep (~11-12 GB, ~2.2 s) **and** the lend evicts the VRAM expert tier, slowing the request already running |

## Proposed policy (for discussion)

The queue is drained by a decision, not a FIFO tug:

1. **Cheap promotion first.** A queued request whose prompt is almost entirely a cache hit (`new tokens` below the
   token-path threshold, and cheaply below ~32) is admitted immediately. It pays no sweep, so it cannot hurt the
   running request. This is likely the common case for an agent turn: the slot cache turns a 100k conversation into a
   few hundred new tokens, and the token path handles a handful of them.
2. **Pair instead of serialise.** Two queued requests whose combined prompt fits one chunk are admitted together and
   prefilled as one sweep (measured **1.80x** vs separate: 10.94 GB / 3,057 ms against 23.01 GB / 5,512 ms).
3. **Otherwise wait behind.** A single expensive prefill admits only when the engine would otherwise idle, because it
   evicts the expert tier the running request is using.
4. **Never preempt.** Swapping phases evicts and reloads (prefill borrows ~90% of the pool and evicts the tier to get
   it); a preemption costs more than letting the current request finish.

Ranking of queued work then becomes: *cache-hit promotions*, then *pair-able prompts*, then *everything else in
arrival order*.

## What is still unknown, and how to settle it

- **Queue depth in practice.** If two requests are rarely waiting together, branch 2 never fires and the whole
  prefill-alignment effort is worth less than the slot cache. The `glm align:` probe (`generate.cpp`, flag
  `STRATA_GLM_PREFILL_ALIGN`) counts exactly this - waiting count, free contexts, and each waiting prompt's length.
- **How accurate the cache-hit hint is before the prefill actually runs.** The slot lookup happens before the read, so
  in principle the count is known; it needs confirming that it is available early enough to decide with.
- **Whether promotion ever beats waiting.** Measure both under the same arrival pattern: promote-on-arrival vs
  wait-behind, with total turn time as the metric, not tok/s.

## Streaming a short prompt through the decode batch (user idea, 2026-10-28)

For a request whose prompt is a cache hit plus a short tail, there is a third option besides "prefill it now" and
"wait behind": **feed the tail through the decode batch**, one token a step, so the new request catches up to the
phase the running ones are already in. No prefill phase is entered at all, so nothing is evicted - which is the whole
problem with promoting a prefill.

The engine already has the seed of this: `prefill` declines below `prefill_min_n` (`glm_prefill.cu:4231`, default 8)
and the token path steps the tokens instead. The change would be to raise that threshold and to do it **inside the
shared step loop** (`multi_step` / `multi_rows`) rather than as a blocking call, so the new request's tokens ride
along in batches that were going to run anyway.

Crossover, from the measured constants:

- prefill: `2.16 s + 0.33 ms/token`, and it evicts the expert tier for the duration (a cost to everyone else);
- streamed through decode: ~`45 ms` a token (21.7 tok/s) at the *sole* user's rate, but as *extra rows in a batch
  that is already running* the marginal cost is far lower - the batch pays for the experts once either way.

So even at the pessimistic 45 ms/token, 8 tokens cost 0.36 s against a 2.2 s prefill; the two are equal around
**~48 tokens** (2.16 s / 0.045 s), and the real crossover is higher because streamed tokens share an existing batch.
A threshold in the **50-200 token** range therefore looks right, and it should be treated as a measured constant like
`prefill_min_n`, not a guess.

Second-order benefit: a streamed prompt never borrows the pool, so it cannot evict the tier that the decoding
requests are reading from - the failure mode that made promotion expensive in the first place.

Question to settle by measurement: whether `multi_rows` can take a request whose pending token is a *prompt* token
(the rows path is built for one new token a conversation; a prompt tail is the same shape, just longer) - if so this
is a scheduling change, not a kernel change.

Note the CPU variant the user also raised: a full CPU forward for a short prompt would need a CPU path for the mixer
and attention, which this engine does not have (its CPU lane covers a *subset of MoE experts*, not a whole step).
The GPU-batch version above gets the same "catch up without a phase change" benefit using machinery that exists.

## Measured: where the queue actually is (2026-10-28)

Sampling `/metrics` while two and four clients ran concurrently:

| clients | engine `live.active` | engine `queued` | requests that reached the engine |
|---|---|---|---|
| 2 | **2** | 1 (momentarily) | 2 |
| 4 | 1-2 | 0 | **2** (the other two never arrived) |

So:

- **Engine-level parallel does engage**: `active=2` with `--parallel 2`. The batched decode path is live.
- **The engine's `pending` deque is not the queue in practice.** `admit_pending` only runs when `pending` is
  non-empty, and my probe inside it logged nothing across dozens of requests even with four clients - because the
  fourth/second requests are held in the **serve layer**, not in the engine. That matches `serve/server.py`'s own
  module doc ("one sequence at a time behind a FairLock").
- **A third and fourth client is where the queue forms**, and it forms in Python.

Consequence for the policy: the admission decision described above has to be made in `serve/server.py` (or the
engine's `pending` has to become the real queue). Deciding it in the engine would mean reworking how GEN lines are
admitted, because today an admitted request goes straight to `gen_admit` and only a request that does not fit is
deferred.

Also measured, and consistent with the prefill findings: two concurrent requests still cost about 2x one prompt of
prefill (6.08 s against 2 x 3.03 s), because `gen_admit` prefills each request as it admits it. That is the seam where
prefill alignment has to go in.

## Measured: token path vs prefill for a short prompt (2026-10-28, `STRATA_GLM_PREFILL_MIN` set high to force it)

| prompt tokens | token path wall | prefill wall (measured elsewhere) |
|---|---|---|
| 39 | 2.27 s | - |
| 67 | 2.69 s | - |
| 147 | - | 1.52 s |
| 311 | 10.88 s | - |
| 1,594 | 46.48 s | ~2.7 s (extrapolated at 0.33 ms/token + 2.16 s) |

Least squares on the token-path points: **~1.7 s fixed + ~15 ms/token** (about 66 tok/s marginal). Prefill is
2.16 s + 0.33 ms/token. They cross at **~32 tokens** (0.47 s of difference over 0.0147 s/token); the earlier ~65
estimate assumed 30 tok/s for the token path and was too pessimistic.

Consequences, and they are not what I expected:

1. **Streaming a short prompt through the decode path is not a latency win.** The token path carries a ~1.7 s fixed
   cost of its own, so a 39-token prompt costs 2.27 s streamed against ~1.5-2.2 s prefilled. The existing default
   `prefill_min_n = 8` is already close to the right threshold; ~32 would be marginally better and no more.
2. **Its real value is non-blocking**, which is a different argument: a streamed prompt never borrows the pool, so it
   cannot evict the expert tier the parallel requests are reading from. That matters for *queue* behaviour (a cheap
   request can be promoted without hurting the ones running), not for that request's own speed.
3. **The unexplained piece is the ~1.7 s fixed cost in the token path.** A 39-token request should be near-instant at
   21-30 tok/s of decode. Something per request costs ~1.7 s before any useful work: slot restore/save, the
   system-prompt handling ("saved a 7985-token conversation ... for other chats that start with it"), or engine
   setup. That is on the critical path of every small agent turn and is worth chasing before any scheduling work -
   if it is avoidable, it is worth more than every scheduling idea in this document.
