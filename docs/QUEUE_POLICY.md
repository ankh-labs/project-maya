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
