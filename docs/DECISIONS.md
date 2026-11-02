# Decisions

## CLOSED 2026-10-30 — an explicit copy lane for the de-fused scheduler

**Decision:** the de-fused row scheduler does not get a dedicated copy lane. `STRATA_GLM_COPY_LANE` stays off by
default; the code is kept for the untested case below, not as a shipping path.

**What the design actually bought, measured:** two rows overlapped produce correct full-length output, and the schedule
is faster than running the same rows on one stream by **~+4 % paired (median +1.2 tok/s, overlap faster in 6 of 6
adjacent pairs), n=6 per arm, one boot**, against a baseline of serial median 31.36 tok/s and overlap median 33.37.
Marginals overlap (serial p95 33.31 sits above overlap's minimum 31.54), so the paired figure is the one to quote.

**Why the copy lane adds nothing — one line:** the two rows already hide each other's PCIe latency, so the de-fused
schedule *is* the copy overlap; an explicit lane only adds two event round trips a layer against latency that is already
covered.

**The measurements that settled it:**
- counted misses (per layer, per step, both rows): 88-140 experts, 884-1417 MB, i.e. 1.96-3.11 experts and
  19.65-31.50 MB a layer — about 1-1.5 experts and 10-16 MB per row. Deriving the same figure from the reported hit
  rate gave ~2x that, and the hit rate itself moved 71 % -> 83.5 % between requests.
- one shared copy stream: both arms ~8 % slower (serial 28.86, overlap 29.76), paired +2.6 (6/6) — the margin grew but
  the level fell: one stream serialised the rows' fetches.
- one copy stream per row: paired +0.66 +0.96 -0.07 -0.19 -0.59 +0.51 (3/6, median ~ +0.24), overlap median 31.62
  against its own 33.37 baseline. The serialisation was removed and the advantage still vanished.

**Not tested, and the only case where it could still pay:** a model that spills beyond RAM — the disk-tier rung this
design keeps for exactly that — or a layer whose fetch dominates its compute. At 1-1.5 experts a row, neither holds on
GLM-5.3-Flash on this box.

**Method note that generalised:** every cross-boot comparison on this branch was wrong (13.7 vs 25.8 tok/s, and twice a
result credited to the rotor that the rotor was not carrying). The intra-boot ramp is 2.3x (first run 14.56 tok/s,
steady state 31-35), so arms must be interleaved inside one process, a warm-up pair discarded, and the paired deltas
quoted with their spread rather than a single percentage.

## NEXT (not started) — the disk rung, Maya-L IQ3_S

Downloaded and verified: 156,319,213,529 bytes across four shards, all sha256-clean.  No pack, no config yet.

**Why it matters:** the rung is about experts against SLOT capacity, not against RAM.  Current slots are 4079 VRAM
(36.0 GB) + 9857 RAM (86.5 GB) = 122.5 GB, which is why v3q's 114 GB of experts reports disk 0.0 and always has.
Maya-L carries ~145-150 GB of experts against the same 122.5 GB, so ~25 GB (~17 %) lands on the disk rung - the first
time that code runs at all.

**Prerequisite, and it is not optional:** stop the production engine first.  `maya.exe` holds ~99.5 GB RSS and the box is
127 GB; a 156 GB model will not load beside it, and the repack needs headroom too.

**Expectation:** correctness before any throughput number.  This exercises untested code, so the first question is
whether it loads and produces coherent output, not how fast.

## 2026-10-31 - PRESTAGING: the z-after-y ordering is the design; PRED_PF is a different, weaker mechanism

Two mechanisms are easy to conflate, and the repo only has the weaker one. This is a decision record, not an
implementation task.

**What exists, gated off** (`STRATA_GLM_ROWS_PRED_PF`, `glm_rows.cu:908-919`): layer L+1's router weight is applied to the
**current** activation `x` - i.e. *before* L has computed. That is a genuine **prediction**: it can be wrong, and if it is,
the staged experts are the wrong ones and the miss path runs anyway. Its value therefore depends on a hit rate nobody has
measured.

**What the schedule in this document specifies** (`LAYER_PIPELINE.md`, the RxTy staging): stage z routes L+1's experts
**after stage y has executed L**, so the router runs on L's **real output**. The expert set is then **exact** - no
predictor, no hit rate, and "100% prestaged" holds by construction.

**Decision: the z-after-y ordering is the design. `PRED_PF` is an approximation that the ordering makes unnecessary.**
It is not a step toward the schedule and should not be read as one; enabling it would answer a different question
(how good is the predictor) rather than the one the schedule asks (can the next layer be staged exactly).

Consequence for any future work: the ordering is a scheduling change - stage z's router reads L's output - and its
prerequisite is that L and L+1 can both be in flight for one token, which is the `pipe_lead` / spare-capacity question
recorded in `LAYER_PIPELINE.md` and measured below. Do not implement it by turning on PRED_PF.

Measured context for the capacity it would need (2026-10-31, the running `--parallel 4` config): VRAM free **1258 MiB of
48935**; expert pool **~33.5 GB** of the 47.76 GB device; `4104` of `12096` pool slots resident; RAM tier 87.94 GB pinned
in 10035 slots. Slot stride is **10.15 MB, measured** (795 MISS lines: 126,183 experts / 1250.8 GB — min 9.17,
median 10.15, p95 11.00, max 11.97), and **42** MoE layers (45 layers less `dense_lead = 3`). So raising
`kSpares` 3 -> 7 reserves `4 x 42 = 168` more slots, **~1.70 GB** — which does **NOT** fit inside the 1.23 GB
free, and it competes with the 13.2 GB the prompt path borrows. (The "~2.8 MB / ~0.47 GB, fits" figure this
paragraph used to carry was wrong: it divided pool GB by **12096 keys** — `288 x 42` — not slots. The "+1.6 GB"
figure it replaced was right about *affordability*; only its derivation was off.)

## 2026-10-31 - THE FETCH BOUND: bytes OFF the link, not bytes made cheaper

**Measured** (one warm boot, `--parallel 2`, greedy 96-token requests, 6 interleaved pairs, arm
verified from the log): moving every RAM-tier expert from PCIe to the CPU lane
(`strata_tune.pcie_frac=0.00`) gives **32.52 -> 38.33 tok/s, paired delta +5.81 tok/s, sd 1.93,
faster in 6/6 = +17.9%**, 0 faults, 559 rotor entries. Verification: `moe_fetch` 5.16 -> 0.34-1.38 ms
and `moe_cpu_wait` 4.01 -> 5.6-14.1 ms at 0.00; the opposite arm (`1.00`) gives `moe_fetch`
10.8-23.9 ms and `moe_cpu_wait` ~0. The ordering is monotone in PCIe share:
0.00 (38.3) > ~0.30 calibrated (32.5) > 0.50 (25.5) > 1.00 (12.8-21.7).

**Why the engine's own split loses - the objective, not the arithmetic.** `lane_plan` minimises
`max(PCIe, CPU)` *within one route*. That is right for one route and wrong for the system: routes
are pipelined, the CPU lane has spare capacity, and the link is the shared serial resource. A
per-route min-max therefore over-prices the CPU lane and over-uses the link.

**The link is NOT saturated - retracting an earlier reading in this project.** `nvidia-smi` reports
`pcie.link.gen.max = 4` (x16, ~25 GB/s); the measured per-expert 17.9 GB/s (`0.567 ms` for
10.15 MB) is **latency/setup-bound, not bandwidth-bound**. That is why a bursty `moe_fetch`
(0.3-24 ms across arms) coexists with flat tok/s when the arm is a no-op. "PCIe 4.0 saturation,
so bytes are the exchange rate" appeared in an earlier report and is **retracted**; the useful
yardstick is *bytes off the critical path*.

**Two contaminated attempts, kept because they are the lesson:**
1. `pcie_frac` at the request body's TOP LEVEL is silently ignored - the server reads
   `body.strata_tune.pcie_frac` (`serve/server.py:1358-1367`). Four identical default runs came
   back flat to within noise, and would have "proved" that bytes do not matter. **A knob that is
   not verified from the log is not an experiment.**
2. With the field nested, the first sweep still had 5/12 requests return nothing and `faults: 9`;
   its ratios were directionally right and numerically untrustworthy. Only the run with 0 faults
   and 6/6 paired is quoted above.

**Status: NOT ADOPTED - it is a quality trade until measured.** The CPU lane and the GPU expert
path are different code, so this is a different computation, not a formatting difference. The
specified gate (`tokcmp/rank.py`) ranks GGUF builds under `llama-perplexity --kl-divergence`
against a full-precision reference; it **cannot rank an engine split**, and the GLM engine has no
KLD dump path. Next: CPU-lane capacity (`cpu_threads` / `STRATA_GLM_CPU_LANE_MAX`) once the lane,
not the link, is the exposure.

## 2026-11-01 - QUALITY GATE PASSED (as a power bound) AND THE PLANNER FIX IS CLEARED

**The claim, stated so it cannot be upgraded later:** moving every RAM-tier expert to the CPU lane
shows **no detectable difference in KLD(good) at a 95% CI of about +-0.005 on a value of 0.152
(~+-3% relative), and top-1 agreement was NOT tested because the dump holds no argmax**
(`rank.py:126`). A tie is a failure to detect a difference at this power, not a proof of equality.
Three tests, all ties: token-block -0.0029 [-0.0086,+0.0013]; document-cluster -0.0029
[-0.0059,+0.0049]; per-document cal better 11/17, sign p=0.332, -0.0006 [-0.0077,+0.0082].

**Instrument (no engine change):** `maya generate --tokens-file` +
`STRATA_GLM_SCORE=2048 STRATA_GLM_SCORE_PREFILL=1 STRATA_GLM_SCORE_DUMP` -> one log-softmax row per
scored position -> the tokcmp `.bin` layout `[f32 kld x N] ++ [f32 p_diff x N]` -> `rank.py`
unchanged. Determinism: two separate invocations, byte-identical, max|delta| 0.000e+00.
**Cost: ~85 min** per two-arm gate (~150 s/arm/chunk; ~108 s of that is model load). The batched
emitter (`--dump-logits`) is 4096 tokens in 3.7 s but unreachable through the CLI (the dump lives in
the generation loop; `--tokens-file` takes `run_request`) - making it reachable is the way to turn
an hour into seconds.

**Cross-engine caveat (load-bearing):** absolute KLD 1.07 is Maya vs the llama.cpp full-precision
base and is **not** comparable with `unsloth`/`fullours`/`v3`/`v3q`; only Maya-arm-to-Maya-arm
paired differences are valid.

**Consequence: the `lane_plan` objective fix is cleared to implement** - `s* = argmin_s
max(L_link((1-s)W), L_cpu(sW))` over measured load curves, reducing to today's plan at n=1
unloaded, concurrency as an input, today's `lane_plan` verbatim before any measurement exists, and
the back-off path (knee detection -> retreat -> recovery floor) exercised with
`STRATA_GLM_CPU_LANE_MAX=32`, which reproduces oversubscription at -34%. The planner changes
numerics, so it does **not** inherit this verdict: it is re-ranked with the same instrument.

## 2026-11-01 - STATE OF THE BOX: build\maya.exe is a CANDIDATE, not the deployed build

`build\maya.exe` was rebuilt on branch `feat/lane-plan-objective` (base `cf2f050`) with the
`lane_plan` wave-count change and has **not** passed its quality gate. A marker beside it,
`build\CANDIDATE_UNGATED.txt`, names the commit, the gate status and the exact revert. The release
tag `deployed/v1.1.42-20261030-2106` (`f383c28`) and the deployed config are untouched; only this
binary path carries the candidate. Delete the marker when the change is committed with its gates
passed, or after the revert is rebuilt.

**The change, and why the units were wrong.** `lane_plan` priced `kk` CPU experts as `c_ms * kk`
- serial accumulation - while pricing the link as `p_ms * (f - kk)`, an aggregate throughput. The
CPU lane runs `T_eff` experts at once, so the honest term is a wave count
`kLaneOverMs + c_ms * ceil(kk / T_eff)`. Independent proof from this box: the `pcie_frac=0` arm
moves ~95 lane experts a token inside a 20.9 ms token - 33 ms of serial work, impossible; 2.1 ms at
sixteen waves. `T_eff` is the lane's own thread count at setup and `cpu_lane_threads()/seq_count()`
at retime, so concurrency stays an input; the knee stays measured (`lane_retime`, 1.15x rule, the
32-thread datum as the known penalty slope).

**Consequence for the design record:** the property "reduces to today's plan at n=1 unloaded" is
**false** and was withdrawn - the old plan was mis-priced at every concurrency, which is why n=1
measured +7.6% rather than nothing. A rule that preserves a dimensional error is wrong by default.

**Same-binary A/B:** the old plan is reproduced exactly with `STRATA_GLM_CPU_PLAN=011233445` (the
table the old planner printed). An env pin leaves `F->cpu_retime` false (`:2155`, `:3476`), and
`lane_retime` returns early on `!F->cpu_retime` (`:4227`), so **the pinned arm cannot be re-planned
mid-run** - the contamination risk is structurally excluded, not assumed away.

## 2026-11-01 - THE PLANNER'S QUALITY RESULT: two statements, never conflated

**1. The corrected planner vs the `pcie_frac=0` arm - LITERALLY the same computation.** The wave
plan on this box is `0 1 2 3 4 5 6 7 8` (all-CPU for f >= 1), which is precisely what
`pcie_frac=0.0` did. The dumps are md5-identical (`48614862c4b958ece8bc183a5b013700`) and every
chunk's lane-expert count matches (chunk 0: 194,394 = 194,394); the scorer is deterministic
(`max|delta| 0.000e+00`), so the identity is by construction rather than by luck. **Quality here is
not a tie - it is UNCHANGED, and the md5 identity is the evidence.**

**2. All-CPU vs the calibrated split - a MEASURED TIE, not equality.** No detectable difference in
KLD(good) at a 95% CI of about +-0.005 on 0.152 (~+-3% relative); **top-1 agreement was NOT
tested**, because the dump holds no argmax (`rank.py:126`). A tie is a failure to detect a
difference at this power.

**Statement 1 does not imply statement 2**, and neither may be compressed into "no quality cost".
The pinned old arm's dump is `28d0870d21cf528df8773d376a107831` - byte-identical to the calibrated
arm's - which is what makes it the old behaviour exactly rather than approximately.

**Consequence: the planner needs no second re-rank.** It is one of the arms already measured.
The perf arms are a *consistency check on the harness*, not a new finding: if the planner is
bit-identical to `pcie_frac=0`, throughput must reproduce +17.9% (n=2), +14.1% (n=4) and +7.6%
(n=1) against the frozen old plan; **any materially different number, better or worse, voids the
run as a harness fault.**

## 2026-11-01 - INSTRUMENT FACTS (not notes on one run): the CPU lane's "threads" has two meanings

**1. `STRATA_GLM_CPU_LANE_MAX` sizes the POOL, it does not set the lane's count.** The lane starts
with its own count (`threads` from `STRATA_GLM_CPU_LANE`, cores, NUMA and part adjustments - stored
as `F->cpu_threads0`), and only a request's `cpu_threads` (or `set_cpu_lane_threads`) raises
`pool.active()`. **Any arm that uses `_LANE_MAX` to mean "run with N threads" measures the old
count under a new label** - a 32-thread back-off arm set that way runs 16 threads and will look
like the back-off works. Use the request field, and set `_LANE_MAX` only to make N *permitted*.

**2. `pool.active()` is NOT observable in the log or in `/metrics`.** `fast_cpu_lane_setup` prints
the lane's own count once (`:3351`), before any request can change it; `set_cpu_lane_threads`
(`:4296`) prints nothing. So a request's `cpu_threads` cannot be confirmed from outside the
process. The back-off gate therefore needs a one-line log on the change (only when the count
actually changes), or the arm cannot be verified and its timing must not be read.

**3. Two non-timing facts salvaged from void runs** (recorded because they are real, and labelled so
they are not mistaken for the gates): all three boots printed the corrected plan
`0 1 2 3 4 5 6 7 8` at both parallel 2 and parallel 4, stably across boots - that is
**restart persistence in plan terms**, not the restart-persistence gate, which still needs
throughput across two boots. And `_LANE_MAX=32` printed `16 threads`, which is how fact 1 was found.

## 2026-11-01 - FOUR VOID PERF ATTEMPTS: how the harness, not the planner, kept failing

**No throughput number from tonight is valid.** The four attempts and their exact failure modes:

1. `Group` collided with the built-in alias for `Group-Object` -> every timed run threw
   `A positional parameter cannot be found that accepts argument '96'` and reported **0 tok/s**.
   A flat table that looked like a result.
2. The renamed rerun wrote **0 KB** and appeared dead; it was not - see 4.
3. `perf2` invented its own log name (`perf-<cfg>-<n>.log`) while the engine writes to the CONFIG's
   `log` path, so the plan-line read threw and the run stopped at boot 1. (Same error as the earlier
   `maya-par4.log`.)
4. **The decisive one: two harnesses ran at once.** My "kill" matched `python.exe` with `perfarms` in
   its command line, but the harness is a **PowerShell** process running the .ps1 - so the kill took
   the *engine* and left the harness alive, which restarted the engine and fought the run I had just
   launched. Every symptom followed: 0-token runs with `An error occurred while sending the request`,
   n=4 at ~22 tok/s against an expected 34-39, rotor entries 25 instead of 583, boot 3 at 10-12 tok/s.

**The lesson, and it is the project's own:** *a killed engine is not a dead job.* The harness must be
identified by its own process (pwsh + script name), and no run may start until the box is provably
free. Concretely, the next attempt needs all three:
- a **lock file** (pid + start time) that the harness writes and the preflight refuses to cross;
- a **post-hoc exclusivity check**: count engine boot markers in the log inside the run window and
  require exactly the number of boots the harness intended - more than that means contaminated;
- kill by **pwsh/script name**, never by a substring that only matches the engine.

**What did survive, all non-timing:** every boot in every attempt printed the corrected plan
`0 1 2 3 4 5 6 7 8`, at parallel 2 and parallel 4, across at least six boots. That is restart
persistence **in plan terms** - not the restart-persistence gate, which still needs throughput
across two boots.

## 2026-11-01 - CORRECTION: the token-equality gate was measuring the wrong thing

**The live setting is stochastic by construction.** Both production configs sample at
`temperature 1.0, top_p 0.95, top_k 40, min_p 0.05`. Temperature 0 is neither recommended nor run.
So "bit-identical greedy output" was **never** a valid gate on this box: at the settings actually
in use, token divergence between two runs is *expected*, and a token-equality test measures the
RNG rather than the change under test.

**Corrected framing of the 4/6 self-divergence.** At temperature 0 the engine disagreed with itself
on 2 of 6 prompts (Fisher p ~ 0.24 against the cross-arm rate). That is a real observation about an
**unused mode** - a curiosity, not a defect, and not the headline. The headline is the rule:
**gate on distributions, never on tokens.**

**The gate specification was wrong, and it was the reviewer's, not the implementer's.** It was
stated twice, the second time in `HANDOVER_spare_ring_2026-10-31.md`, whose "bit-identical greedy
output" requirement is **replaced** by a `rank.py` gate (KLD on ref-good tokens, paired bootstrap).
Anyone meeting the old wording should treat it as superseded, not as a requirement to satisfy.

**The instrument is right for the right reason, and independent of sampling.** `rank.py` scores KLD
on the **pre-sampling distributions**, so it is temperature-independent and valid at the live
settings. The determinism PASS (`max|delta| 0.000e+00`) is a property of the **harness** -
teacher-forced, n=1, no sampling - not a claim about the server. That is the division to preserve:
**a deterministic scorer measuring a stochastic server.**

**MTP drafter: re-open candidate, with the reason recorded.** It was parked as "NOT LOSSLESS -
2/4 prompts diverge" while worth **+15.8% decode warm at accept 0.945**. At temperature 1.0 prompt
divergence is expected from sampling alone, so 2/4 is uninformative; and speculative decoding's
correctness claim is that it is **distribution-preserving**, so the only meaningful lossless test is
distributional - KLD plus accept rate, i.e. exactly the instrument that now exists. Not to be
started without a decision, and not assumed to transfer (that finding is from the llama.cpp/GLM lab,
not Maya) - but it may be the largest unclaimed win on the box, and the blocker has been removed.

## 2026-11-01 - TWO CLAIMS, TWO MEASUREMENTS: identity is not the win

**Claim A - the IDENTITY.** The corrected planner is bit-identical to the `pcie_frac=0` computation
(md5 of the dumps, lane-expert counts per chunk). The throughput check for it is a *consistency
test*: n=2 ~+17.9%, n=4 ~+14.1%, n=1 ~+7.6% against the reference arm. Measured on the new binary:
n=4 **+1.2%** and n=1 **-0.7%** vs that reference - these are **passes because they are near zero**,
i.e. the new planner reproduces the reference arm. They are NOT the planner's benefit.

**Claim B - the WIN.** What the planner buys over the OLD plan must be measured against the old plan
itself, same binary, same session: **ABBA order with the first boot discarded** (a linear drift
cancels exactly; OLD-new-OLD-new would make NEW the warmer arm in every pair and can turn a true
+5% into an apparent +17%). Reported per boot, in order, so any drift is visible rather than averaged
away.

A reader who takes Claim A's "+1.2%" as the planner's benefit is reading identity evidence as a
benefit figure; the two must never be merged in a summary.

## 2026-11-01 - THE ADOPTABLE FIGURE AT n=2 IS +11.8%; THE +17.9% IS SUPERSEDED

**Measured properly** (same binary, same session, paired **ABBA** across boots, first boot discarded):

```
discard   35.68   (dropped - page cache warms monotonically)
OLD-a     31.80      NEW-a  35.37
NEW-b     35.75      OLD-b  31.81
OLD mean 31.81 (boot-to-boot spread 0.03%)   NEW mean 35.56
NEW - OLD = +3.75 tok/s = +11.8%    per-pair +3.56, +3.94
```

**The earlier +17.9% is SUPERSEDED - kept, with its mechanism, so it cannot be re-quoted.** That arm
switched `strata_tune.pcie_frac` **per request inside one boot**, which also changes which experts
stay resident: those two arms differed in **cache state** as well as lane assignment, and on this box
throughput is a function of cache state. It also ran with `STRATA_GLM_PROF=1` and
`STRATA_GLM_MISS_LOG=1`; the ABBA run has neither. **A mid-session switch and a boot-time decision are
different experiments**; adoption is the boot-time decision, so +11.8% is the number to use.

**Reviewer's error, recorded:** the perf arms were described as "a consistency check, not a
discovery". They were a discovery - they discovered that the reference was measured under different
conditions. **Same computation does not imply same throughput when the two were measured under
different cache and instrumentation conditions.**

**Claim A (identity) stands, and is about computation only** - bit-identical dumps, identical
lane-expert counts per chunk. It does not carry throughput, and must not be summarised as if it did.

## 2026-11-01 - INSTRUMENT FACT: `maya:engine_vram_hit` is CUMULATIVE-since-idle in the multi path

The engine builds `vram_hit = h / (h + m)` from `h = sn.hits - X.hits`, `m = sn.misses - X.misses`,
where **X differs by path**:
- single-request path (`generate.cpp:2490`): `st1`, a per-request snapshot -> a genuine per-request rate;
- **multi path (`:2852`), which is what `--parallel` runs: `multi_st1`, captured when the engine was
  last IDLE** ("over everything decoded since the engine was idle").

With `--parallel`, back-to-back requests never let the engine go idle, so the exported ratio is a
**running average over the whole session and climbs monotonically by construction**. It must never be
used as a per-run residency measure, and it cannot be differenced from `/metrics` alone because hits
and misses are not exported separately.

**Use `maya:engine_vram_used` + `maya:engine_vram_slots`** ("expert slots in use in VRAM", a LEVEL).
A level can fall, so it cannot confirm a ramp by construction - which is the whole reason to prefer it.

This is the measurement form of `feedback_verify_the_instrument_reads_distinct_objects`: **prove the
number can go down before trusting it going up.**

## 2026-11-01 - PRE-REGISTERED (before the residency answer is known): the warming test

Written while the level-based residency measurement (`maya:engine_vram_used`) is still running, so it
cannot be shaped by its result.

**If `vram_used` climbs across runs 1->8 under the all-CPU plan** (the pool filling over the first
requests), the follow-up is: **warm the pool at boot, then re-measure RUN 1 ONLY.** Success is
**run 1 reaching steady state** - not merely run 1 improving. A warm-up that halves the cold cost has
not fixed it.

**If `vram_used` is flat** while throughput ramps, the residency explanation is dead: there is no
warming change to make, the cold cost needs a different explanation, and **the correct action is to
say so and stop** - not to try a warm-up anyway on the grounds that it cannot hurt.

**Scope note:** the n=1 decomposition (run 1 -21.5%, steady state +8.0%) pre-registers the SHAPE
only. Its magnitudes are not pre-registered and must not be used as the expected result of the ABBA
run - that would be choosing an expectation from the data.

## 2026-11-01 - HARNESS FAULT: a gauge read that returns nothing, and one that returns an artefact

Two failures of the same measurement, which fail differently and are worth keeping apart:

1. **Silent, and it would have "confirmed" the hypothesis - the cumulative ratio.** `vram_hit` is
   `h/(h+m)` against `multi_st1` (last idle), so sampling it per run gives a running average that
   climbs by construction. Only an external check of the definition caught it; the number looked
   exactly like the quantity wanted.
2. **Loud, and it cost nothing - the wrong endpoint.** The gauges were read from `/metrics`; the
   Prometheus text is at **`/metrics/prometheus`**. Every field came back `-1`, which is
   self-announcing: a read that returns nothing aborts, a read that returns a plausible number does
   not.

**Rule:** before trusting a gauge, prove (a) it is defined on the endpoint being read, and (b) the
quantity can move DOWN. A level can fall; a ratio referenced to "since idle" cannot, quickly.

## 2026-11-01 - CORRECTION to the `vram_hit` instrument fact: it is per-request when requests are SERIAL

The entry above said the multi path's `vram_hit` is "a running average over the whole session". That is
too broad and is corrected here.

`multi_st1` is reset **only when a request is admitted with `gens` empty**
(`generate.cpp:3298-3301`), i.e. when nothing else is in flight:
- **serial requests (n=1)**: the reset happens at every admission, so `vram_hit` **and** `promo` are
  genuine **per-request** values;
- **overlapping requests (n=2, n=4)**: `gens` is non-empty at the second admission, so the ratio spans
  the whole **batch** and does not reset between requests.

Consequence: the residency test at n=1 - which is where the cold cost lives - can use both
observables. The cumulative-ratio warning stands for concurrent arms only, and neither figure is ever
a session-wide average when requests are serial.

Note also the denominator: `promo` is promotions **per token**, so per-run promotions = `promo` x
tokens generated.

## 2026-11-01 - THE ALL-CPU PLAN PERFORMS NO PROMOTIONS AT ALL (its own fact, not a null result)

Measured on the instrumented build (8 serial runs, `tier` line): under the winning plan
`promo_total` stayed at **0 for every run**, while `vram_used` sat at **3937-3953 of 4079 slots**
(96-97%). The pool is nearly full of content the plan **never uses or refreshes**, because with no
PCIe fetches there is no promotion path into VRAM at all.

This is why the residency close is an exclusion by **mechanism** rather than by a flat trace: a plan
that promotes nothing cannot be converging its pool composition either. Both flat branches fire.

**QUEUED, NOT STARTED - is there a free win from giving the hottest experts back to VRAM?** If VRAM is
effectively idle under the winning plan, the planner's "all-CPU" verdict is optimal *given the costs
it prices* - and those costs were corrected once already tonight (a latency compared with a
throughput). Re-ask with the corrected arithmetic, and with the observation that the pool it
abandons is 96% occupied. Not tonight.

## 2026-11-01 - QUEUED CANDIDATE for the unexplained cold cost: host-side warming

With promotions at zero, **nothing about VRAM changes across the 8 runs**, so the ramp cannot be a
GPU-residency effect. The remaining candidate named tonight is **host-side warming of the expert
weights** (working set / page cache). This lab already has evidence for that shape on this box
(`feedback_page_cache_warms_monotonically`), documented for *cross-boot* rather than within-boot -
so it is a hypothesis with local support, not a proven mechanism.

Recorded and not chased: the stop rule holds, and the planner should land rather than a third
mechanism be hunted at 02:00. The next session starts from this hypothesis instead of a mystery.

**The detour's cost and return, stated plainly:** the residency investigation consumed **six
instrument faults and produced no timing data**, and it still returned a mechanism fact (zero
promotions) that materially strengthens the exclusion. That is a fair trade, not a failure - said
here rather than apologised for.

## 2026-11-01 - COMMITTED ON `feat/lane-plan-objective` IS NOT ADOPTED

**A commit records the work; it does not ship it.** Adoption - deploy, re-tag, config repoint -
requires the **back-off** gate and the **restart-persistence** gate to pass, and both are
**outstanding** at the time of the commit. The release tag `deployed/v1.1.42-20261030-2106`
(`f383c28`) and the deployed config are untouched, and that is what actually protects against a
premature adoption.

Written out because the failure mode is one of us (or a later session) reading a commit plus a table
of PASS rows as a shipped change. The gate table records what was measured, not what was adopted.

Order changed deliberately: commit before back-off and restart persistence, because 25 lines of
uncommitted work is a worse risk than an ungated branch - **not** because those gates stopped
mattering.

**Back-off gate reminders, both already bitten once tonight:**
- the arm is a **request-level `cpu_threads=32`**; `STRATA_GLM_CPU_LANE_MAX` only sizes the pool, so an
  arm using it measures the 16-thread plan under a 32-thread label;
- **verify `active()` from the log before timing anything**, and exercise **recovery as well as
  retreat** - a back-off that engages and never releases passes a one-way test.

## 2026-11-01 - GATE TABLE for `d098fa7` (lane_plan wave count) - COMMITTED, NOT ADOPTED

| gate | result | statistic | binary |
|---|---|---|---|
| quality (planner vs frozen old plan) | **PASS** - tie, KLD(good) 0.152 +-0.005, top-1 **untested** (no argmax in the dump); dumps **bit-identical** to the `pcie_frac=0` arm | paired block/doc-cluster/sign tests | uninstrumented |
| n=2 | **PASS** +11.8% (per-pair +3.56/+3.94; OLD boots 31.80/31.81, 0.03% apart) | steady, paired **ABBA**, 5 runs/boot | uninstrumented |
| n=1 | **MEASURED** steady **+6.8%**, run 1 **-21.9%** | steady = last 3 of 8; unpaired with a **drift bound**: boot-mean spread 0.35-0.61 vs 2xSE 1.74-3.13 | uninstrumented |
| n=4 | **PASS** steady **+23.9%**, full mean **+17.7%**; run 1 -3.6% | steady = runs 6-8, paired **ABBA**; VALID: OLD spread 0.337 <= 2xSE 0.763, per-pair +7.57/+8.42 same sign | uninstrumented (clean rebuild) |
| residency / cold-cost cause | **CLOSED - UNEXPLAINED** (occupancy flat, promotions **zero** -> excluded by mechanism) | 8 serial runs, two boots | **instrumented** (tier line) |
| back-off (request-level `cpu_threads=32`, retreat AND recovery) | **OUTSTANDING** | | |
| restart persistence (throughput across two boots) | **OUTSTANDING** | | |

**Statistics are not interchangeable and must be quoted with their names.** n=2's +11.8% is a
5-run-mean ABBA; n=1 and n=4 are steady-state (last 3 of 8). n=4's **full 8-run mean is +17.7%**,
which reproduces the earlier voided run's +17.9% - the two agree once the same statistic is used, and
its OLD boot (33.37) matches this run's (33.41) to 0.04 tok/s.

**Plan shape at n=4:** the win grows with concurrency (+6.8% at n=1, +11.8% at n=2, +23.9% steady at
n=4) and the run-1 cold cost shrinks (-21.9%, then -3.6%).

## 2026-11-01 - PRE-REGISTERED (before the back-off data): three-way reading, and a silent detector BLOCKS adoption

The planner now sends **every** non-resident expert to the CPU lane. We already measured that the lane
degrades under contention (`cpu_threads` 32 vs 16 = **-12.84 tok/s**). So if the lane degrades and the
detector cannot see it, there is **no safety net at all** - which makes a silent detector a deployment
blocker, not a failed checkbox.

**The three-way reading, fixed now:**
- **retreat and recovery both observed** -> gate **PASS**, proceed to restart persistence.
- **retreat without recovery** -> **FAIL**: one-way back-off, a blocker.
- **retime never fires** -> **FAIL**: the detector is too coarse (its trigger is a >15% drift sustained
  over two 2,000-expert windows, capped at 3 changes), a blocker. **Do not deploy, do not re-tag, do not
  repoint the config - report and stop.** The fix would be a detector that sees **per-expert cost**
  rather than a windowed average: a design change, not a tuning one.

Arm verification is part of the gate, not a preliminary: a **request-level `cpu_threads=32`** (never
`STRATA_GLM_CPU_LANE_MAX`, which only sizes the pool), and `active()` read from the engine log
(`CPU lane active threads N`) **before any timing is read**.

## 2026-11-01 - BACK-OFF GATE: **FAIL (retreat without recovery) - ADOPTION BLOCKED**

Measured (instrumented build, one boot, `active()` read from the log before any timing):

```
arm verified: request-level cpu_threads=32 -> "CPU lane active threads 32"   OK
setup plan:   0 1 2 3 4 5 6 7 8   (all-CPU)
phase A 32 threads: 20.14 24.60 21.46 23.17 21.84 22.15   mean 22.23 tok/s
phase B 16 threads: 27.03 27.80 26.26 25.30 26.45 27.13   mean 26.66 tok/s
retimes: 1
  "an expert 0.561 ms (planned with 0.427) -> of 0..8: 0 0 1 2 3 4 5 6 7  (PCIe share 0.34)"
active() at end: 16
```

**Observed: retreat, and no recovery.** The plan moved off all-CPU once the lane measured
0.561 ms/expert against a planned 0.427, and never moved back after `active()` returned to 16.

**The cost is visible, not inferred:** phase B ran at **26.66 tok/s** on the retreated plan, against
~38 tok/s for all-CPU at n=2 on this box. The one-way retreat is a **~30% permanent downgrade**.

**Mechanism (leading, from the code plus the window arithmetic):** `lane_retime` stops for good once
`rt_windows >= 8`, and a 96-token request consumes roughly three 2,000-expert windows (~63 lane
experts/token), so the budget is spent within about **three requests**. It also sets
`cpu_plan_start = plan`, so the retreat becomes the new baseline rather than a provisional state. A
later recovery therefore cannot fire even when the conditions that caused the retreat have gone.

**Consequence, exactly as pre-registered: do NOT deploy, do NOT re-tag, do NOT repoint the config.**
All-CPU is the plan for every non-resident expert, so with a one-way detector there is no safety net
under contention - and contention is measured, not hypothetical (-12.84 tok/s at 32 threads).

**The fix is a design change, not a tuning one:** a detector on **per-expert cost** rather than a
windowed average, a retreat that does not overwrite the start plan, and a recovery path that is not
capped by a global window budget. That is item 3 territory (the residency-value term), and it is now
also the prerequisite for adoption.

## 2026-11-01 - THE LANE'S ADAPTATION IS BOOT-SCOPED, BY CONSTRUCTION (standalone finding)

`lane_retime` is **not** an adaptive controller. Its budget is `rt_windows >= 8` with `rt_changes >= 3`,
and a 96-token request consumes **~3 windows** (~63 lane experts/token x 96 / 2,000). So the window
budget is exhausted **within roughly three requests of a boot**, after which retime is permanently
inert for the life of the process.

Combined with `cpu_plan_start = plan`, the failure mode is worse than one-way: **the first contention
event during a boot's first three requests sets a ~30% downgrade as the permanent baseline, and nothing
can undo it.** A transient at the wrong moment is a permanent penalty.

**This invalidates any claim - past or future - that the lane "adapts" at runtime.** It applies beyond
the back-off gate: any design that assumes a self-correcting lane/split is assuming something the code
does not do.

**Harness defect found while checking this, recorded deliberately:** the measurement scripts delete the
config log at every boot, so **cross-boot evidence is destroyed** - the four n=1 residency boots cannot
be interrogated afterwards. Keep per-boot logs (an offset or a unique name) in any harness that will be
asked a question later.

**The check that could not be run, and the inference that replaces it:** whether a retime fired during
the n=1 residency boots (which would make plan drift a candidate for the cold-start ramp). Those logs
are gone. Inference from throughput: a retimed plan measures ~26.66 tok/s against ~38 for all-CPU, and
the n=1 boots sustained **44.2-44.6 tok/s steady** - the best figures observed on this plan. A drift
worth ~30% is incompatible with that, so **plan drift is very unlikely to explain the ramp**, and
host-side warming stands as the sole candidate. Inferred, not grepped.

## 2026-11-01 - ITEM 2: STAGE ANATOMY RE-MEASURED - the ranking is RE-DERIVED, not re-sequenced

**PROFILE ONLY.** `STRATA_GLM_PROF=1` perturbs, so **no throughput number from these six boots may be
quoted beside the +6.8 / +11.8 / +23.9 figures**. Both plans instrumented identically, one boot per
(plan, concurrency), plan verified from each boot's own log line
(new `0 1 2 3 4 5 6 7 8`, old `0 1 1 2 3 3 4 4 5`).

Stage medians, ms/token:

| stage | new n=1 | new n=2 | new n=4 | old n=1 | old n=2 | old n=4 |
|---|---|---|---|---|---|---|
| moe_fetch | 0.401 | 0.361 | 0.478 | 1.319 | 1.089 | 1.370 |
| **moe_cpu_wait** | **8.275** | **7.202** | **8.330** | 6.229 | 5.346 | 6.979 |
| moe_gate_up | 2.243 | 2.214 | 2.180 | 2.519 | 2.160 | 2.152 |
| moe_down | 1.393 | 1.324 | 1.331 | 1.369 | 1.287 | 1.265 |
| moe_route | 0.776 | 0.793 | 0.789 | 0.818 | 0.789 | 0.792 |
| router_shexp_mv | 0.863 | 0.847 | 0.846 | 0.862 | 0.850 | 0.846 |
| kda_proj_mv | 2.708 | 2.702 | 2.719 | 2.709 | 2.709 | 2.721 |
| kda_out_mv | 1.122 | 1.119 | 1.119 | 1.121 | 1.120 | 1.122 |
| dsa_out_mv / hc_attn / hc_ffn / mla_attend | 0.594 / 0.531 / 0.551 / 0.543 | same | same | same | same | same |

GPU stream ms/token: new 23.45 / 21.86 / 24.81, old 22.12 / 22.20 / 25.93.

**The trade is exact:** fetch falls ~3x (1.32 -> 0.40 at n=1) and cpu_wait rises by about the same
(6.23 -> 8.28). Every attention/KDA/compute stage is bit-comparable across plans - the control that
the instrument measures the plan, not the boot. **`moe_cpu_wait` is now 33-34% of GPU stream time**,
almost exactly the spec-era 34.6% we disproved at the old plan (12.9%) - it is back by a different
route, because the planner moved the work there.

**Verdicts:** `pcie_frac->0` CONSUMED (now the default). **CPU-lane capacity now #1** (7.2-8.3 ms, and
32 threads already measured -12.84 tok/s). `k` 7->6 SURVIVES (cuts compute and lane work). **The
attention/KDA block (~7 ms) is PROMOTED** - comparable to the lane now, untouched by the planner, and
invisible while fetch dominated. Residency/ring/eviction, row dedupe, wire compression and
overlap-hiding are **MOOT for throughput** (fetch is 2% of the token). **§9 NEEDS RE-DERIVING:** its
premise is true again (cpu_wait 7.2-8.3 vs x+y+z ~= 5.5) but every coefficient is new.

**Open, not chased:** the all-CPU plan still shows 0.36-0.48 ms of `moe_fetch` (p95 6.86, max 8.55 at
n=4). An all-CPU plan should fetch nothing from the RAM tier - so that residual is disk-tier reads,
the scratch path, or prefetch. 2% of the token, but the plan is not as link-free as "all-CPU" implies.

## 2026-11-01 - THE 33% / 34.6% REVERSAL, AND HOW IT MUST BE READ (verbatim framing)

**The spec-era 34.6% is back at ~33% not because the old figure was right, but because the planner moved
the work onto the lane.** Those are different claims about different engines. At the old plan the same
instrument measured 12.9%, and that measurement stands. Anyone reading 34.6% and 33% side by side
without this sentence will conclude we flip-flopped; we did not - the plan changed under a fixed
instrument.

## 2026-11-01 - THE ATTENTION/KDA BLOCK IS A FIRST-CLASS item-3 CANDIDATE, not a footnote

~7 ms per token (`kda_proj_mv` 2.70 + `kda_out_mv` 1.12 + `dsa_out_mv` 0.59 + `hc_attn` 0.53 +
`hc_ffn` 0.55 + `mla_attend` 0.54), **identical across both plans to three decimals**, and never
touched by any work in this project.

It was invisible for as long as fetch dominated - which means it has **never been attacked, not that it
was ruled out**. Everything else on the board has been tried, closed, or re-derived; this is the only
genuinely **unexplored** territory, and it is now roughly the same size as the lane.

## 2026-11-01 - RETIME CHECK: no retime fired in any n=1..n=4 boot (scheduler exonerated)

Asked three times; answered now with direct evidence. Across six preserved profile boots (both plans,
n=1/2/4, 7-28 requests each): **0 `CPU lane timed again` lines after each boot's own setup line.**

**How the first attempt got it wrong, because the trap generalises:** the naive grep reported "1 retime
in every boot" - one **stale line from the back-off run** at line 282 in all six logs, carried in
because `stage.ps1` does not truncate the log. *Identical text AND identical line number across six
independent boots is the tell*; a real retime would differ in value and position. Always count after
the boot's own setup line.

**Consequence:** plan drift does **not** explain the cold-start ramp. The scheduler is exonerated and
**host-side warming stands as the sole candidate**. Caveat: the original residency boots' logs were
destroyed by per-boot truncation, so this is an equivalent-regime substitute (same plan, same 16
threads, 7 serial requests), not literally the four originals.

## 2026-11-01 - residual `moe_fetch` under the all-CPU plan: cannot be localised with existing data

The profile reports `moe_fetch` **aggregated per stage**, not per layer. The engine holds a per-layer
miss array, but it is not emitted, so **whether the 0.36-0.48 ms residual is uniform or concentrated in
specific layers is not answerable from the data on disk**. Concentration would point at disk-tier or a
prefetch path; uniformity at something structural in the plan. Left open, not chased - but the question
now has a precise form: *emit the per-layer miss counts and look.*

## 2026-11-01 - PINNED: `STRATA_GLM_RETIME_V2` defaulting OFF is BUILD HYGIENE, not a shipping choice

`STRATA_GLM_RETIME_V2` (the rewritten back-off controller: per-expert EWMA detector, minimum interval
between changes instead of a lifetime budget, retreat on the LIVE plan with `cpu_plan_start` immutable)
defaults **off** so the committed planner is bit-behaviour-identical unless the flag is set. That is a
build-hygiene choice.

**It does not fix the default.** With the flag off, the OLD controller remains the default path - the
first-three-requests one-shot that turns one contention transient into a permanent ~30% downgrade.
Configuring it off is not a safety improvement; it is a refusal to change behaviour under a flag that
was meant to protect the default.

**ADOPTION = the corrected planner WITH `RETIME_V2=on`, and the back-off gate must pass with the flag
set.** Shipping all-CPU under the old controller is exactly the no-safety-net condition the back-off
gate failed on. This sits beside the "committed != adopted" heading: neither a commit nor a green
gate table ships anything on its own.

**CORRECTION (cadence):** `lane_retime` is called from `next_line` (`generate.cpp:1285`), i.e. **once per
request**, so `rt_windows >= 8` is **~8 requests**, not the ~3 an earlier note derived from an expert-rate
argument (63 lane experts/token x 96 / 2,000). The conclusion is unchanged - the adaptation is boot-scoped
and its first event is permanent - only the number was wrong.

## 2026-11-01 - THE RESIDENCY TERM SAYS "STAGE EVERYTHING"; THE MEASUREMENTS SAY ALL-CPU WINS

First probe line from the `RETIME_V2` build (32 threads, residency term on):

```
rt2 probe: ewma 0.776 ratio 2.19 dir +1 margin 0.07
           (cpu 0.816 vs pcie 0.061, waves 1) explore_left 7 R_est 9.28
```

- `R_est 9.28` -> `p_use = p_ms / R = 0.061 ms/expert`, against `cpu_term 0.816`: amortised staging is
  **~13x cheaper**, so the term wants to **stage everything**.
- **The ~8x margin prediction is REFUTED** (`margin 0.07`, not 8): a cost-ratio trigger has ample
  headroom in the staging direction, so it is not futile and the saturation redesign is not needed on
  these numbers. `ratio 2.19` shows the detector does see a degraded lane.
- Candidate 1 (cadence) CONFIRMED as the blocker: **one probe line in six requests**, `explore_left`
  still 7. Candidate 3 contributing: the first window spans boot -> first request (`rt_e0` is not
  initialised at setup), inflating EWMA to 0.776 against a true ~0.36 - which also overstates the 2.19.

**THE TENSION, stated plainly: the residency term currently says stage everything, and the measured data
says all-CPU wins** (+23.9% at n=4 steady, +6.8% at n=1, both against a partial-PCIe plan). Both cannot
be right. Two candidate reasons, and either one means **the model is incomplete, not the measurement**:

1. **`R_est` is selection-biased.** It was measured while the exploration plan `0 0 0 1 2 3 4 5 6` staged
   only the DEEP routes - plausibly the hottest, most-reused experts. R over all experts will be lower,
   perhaps much lower, and an R measured on a hand-picked subset cannot be applied to the whole set.
2. **No displacement term, a real omission.** `p_ms / R` prices the transfer and nothing else. The pool
   sits at ~96% occupancy, so **every stage evicts something**; if the evicted expert would have been
   used, the true cost is `p_ms/R + displacement`. A model blind to eviction will always want to stage
   everything - which is exactly what we now see. That is not evidence the economics favour staging; it
   is evidence the cost function is missing a term.

**Until R is unbiased and eviction is priced, the term is NOT trustworthy enough to act on.** The cadence
fix must therefore not be allowed to swing the plan to all-PCIe untested: it would move into a
configuration we have never measured, whose nearest neighbour measured *worse*.

**Next session, in order:**
1. **`rt_e0` initialised at first use** (kills the inflated first-window EWMA).
2. **The call-site fix** so `lane_retime` runs at the intended cadence - **with a clamp on how far the
   residency term may move the plan from `cpu_plan_start` per decision.** `RESIDENCY_VALUE` stays OFF by
   default, so nothing moves unless asked.
3. **Re-measure `R` under a plan that stages a REPRESENTATIVE sample, not only deep routes** - the whole
   term rests on that number.
4. **Add the displacement/eviction term**, then re-run the four checks.

## 2026-11-01 - CORRECTION TO THE BRIEF: online replacement was the WRONG FAMILY (do not follow it by default)

An earlier instruction in this file sent the next reader to **ARC / LIRS / W-TinyLFU / GDSF** as the
policy family. **Fiddler says the dominant lever is offline POPULARITY PLACEMENT plus a per-expert
RUNTIME rule, with no online eviction policy at all.** Online replacement is the wrong shelf for this
problem; the ARC/LIRS/W-TinyLFU list is **dropped** unless a trace later says otherwise.

Fiddler's actual design (read 2026-11-01, arXiv 2402.07033):
- placement: the most POPULAR experts, by offline profiling, pinned in GPU up to capacity; the rest in
  CPU memory - its two environments hold 56/256 and 125/256 experts on GPU (22-49% resident);
- runtime, per expert: if resident -> GPU; else if `cpu_lat(s) > gpu_lat(s) + transfer_lat()` ->
  transfer and run on GPU; else -> copy the ACTIVATION to the CPU and run there;
- measured shape: GPU exec ~constant in s, CPU exec ~LINEAR in s, activation copy <1%, weight transfer
  2-5x the GPU compute time.

**THE CONSEQUENCE THAT MATTERS HERE:** at small s (decode) the inequality is FALSE, so the paper's own
model chooses CPU - our all-CPU decode result (+23.9% at n=4) is what the model predicts, not an
anomaly. At large s (prefill) the inequality FLIPS and staging wins - a configuration we have never
run. **Prefill owns real latency on this box (~67% of an agent turn, ~5.10 s fixed), so the staging win
lives in the phase that owns two thirds, while this session optimised decode.**

**FINDING 1 (verified in Maya's own code, not carried over):** the pool fill is ALREADY popularity
ranked. `fast_warm` (`glm_fast_path.cu:3802-3825`) builds each layer's order from the `--expert-profile`
ranked list (`profile_`), else `pack_dir/expert_prior.txt`, with an opt-in per-user usage overlay
(`STRATA_GLM_USAGE` / `save_usage`) leading each layer's order. So placement is already Fiddler-shaped;
**the gap is the phase split, not the policy.**

**REVISED SEQUENCE:** (1) DONE - placement is profile-ranked. (2) OUTSTANDING - measure `s` per expert in
prefill vs decode on our workload; that is the coefficient the phase split rests on and where the
inequality flips. (3) the phase-dependent plan (all-CPU decode, stage prefill), flag-gated. (4) the
trace replay ONLY if 1-3 leave a placement question open - its question is "does a phase-split plan beat
a uniform one", not "which eviction policy wins"; Belady remains the right bound if we get there.

## 2026-11-01 - PREFILL IS OUT OF SCOPE: the residency term is SHELVED, trace/Belady cancelled

**Direction: no prefill work.** Items 2 and 3 (measure `s` by phase, phase-dependent plan) are
**CANCELLED** - no `s` measurement, no phase signal, no kernel/args change. Do not pick it back up.

**Consequence:** the residency-value term loses its only remaining justification. Fiddler's model says
CPU for decode, so staging was only ever going to pay at prefill `s`; with prefill out of scope there is
no live question for `R_est` to answer. It stays **flag-gated and DORMANT** (`STRATA_GLM_RESIDENCY_VALUE`,
off by default), not removed - `R_est 9.28`, the selection-bias caveat and the missing displacement term
stay recorded as a closed line. **The trace replay and Belady bound are CANCELLED, not deferred.**

**Closed line, kept for the record:** the flip point falls out of measured constants -
`cpu_lat(s) = 0.350·s` vs `p_ms + g_ms = 0.567 + 0.43 = 0.997`, so **s >= 3 flips to staging**, and decode
(s=1) is CPU on the paper's own inequality. That is *why* all-CPU fell out of the units fix and why the
decode win was never in tension with the literature. It is also why the term had nothing left to decide
once prefill was cancelled.

**What remains is adoption.** The planner at `d098fa7` is worth +6.8 / +11.8 / +23.9 at n=1/2/4 steady and
is quality-gated; what is idle is two gates: (1) the two decode-side retime defects, (2) the back-off gate
with `RETIME_V2=1`, (3) restart persistence. **RETIME_V2=on is non-negotiable at adoption** - shipping
all-CPU with the old controller is the no-safety-net condition.

## 2026-11-01 - TWO GROUPS OF EVIDENCE, KEPT APART: the old controller's defects are NOT excused by the prime bug

**Group A - ESTABLISHED, and why `RETIME_V2` is required for adoption.** The FIRST back-off gate ran on
the committed binary's **old** controller, before `RETIME_V2` existed, and observed `retimes: 1` - a real
retreat, plan `0 1 2 3 4 5 6 7 8` -> `0 0 1 2 3 4 5 6 7`, then **no recovery**. The two structural
defects behind it were **read from source**, not inferred from a silence:
`cpu_plan_start = plan` (a retreat becomes the permanent baseline) and `rt_windows >= 8` (a terminal
budget - the adaptation is boot-scoped). **Retreat-without-recovery, permanent downgrade and a boot-scoped
budget all stand.**

**Group B - ESTABLISHED NOTHING.** The two silent runs (the new controller, zero transitions then zero
probes) are explained by a bug in my own priming fix: the prime branch reset the window origin on every
call, so the guard for "no baseline yet" made a baseline impossible. **Those runs say nothing about the
detector, in either direction.**

**The distinction matters:** a reader who takes "it was just a prime bug" as clearing the controller would
conclude the old path is acceptable. It is not - Group A has the retreat line to prove it.

**And the saturation question is UNOPENED, not revived and not refuted.** With zero probes, `ratio` and
`want` were never computed; hypotheses 2 (detector reference) and 3 (wave economics absorb degradation)
are **untested**. No conclusion follows about whether a cost trigger can work for the decode lane.

## 2026-11-01 - BACK-OFF: the DETECTOR works and the TRIGGER cannot fire (measured, not inferred)

One boot, `RETIME_V2=1`, request-level `cpu_threads=32`, whole log copied before any restart.

1. **`active()` = 32** - the arm ran (three `CPU lane active threads 32` lines).
2. **`ratio` crosses the threshold**: 1.10 -> 1.14 -> 1.20 -> 1.21 -> 1.29 -> 1.31, `dir` +1 from 1.20.
   **The detector sees the degradation; the boot c_ms reference is not wrong.**
3. **`want == current` at every sample** (`0 1 2 3 4 5 6 7 8`), with `margin` 1.32 -> 1.13 and
   `cpu_term` 0.428 -> 0.501 against `pcie 0.567`.

**The wave economics absorb it.** A lane degraded from 0.352 to 0.501 ms/expert - **+43% per-expert
price** for a **-34% throughput** loss - still prices BELOW `p_ms`, so the planner's own objective says
all-CPU at every sample. **A cost-based trigger cannot fire for the decode lane while that holds, however
well the detector is built.**

**What this establishes:** the signal that sees this failure is NOT price. 32 threads on 16 cores costs a
third of the throughput while moving the per-expert price 43% and leaving it under the staging cost - so
by construction the planner is blind to it. This is the first evidence that a safety net must be built on
**saturation** (wait, queue depth, throughput), not on `c_ms`.

**Consequences:** `RETIME_V2` cannot be completed as a cost-triggered controller - its detector works, its
trigger is structurally unable to fire on the failure we care about. **The back-off gate remains FAILED and
adoption remains blocked**, on a demonstrated signal/failure mismatch rather than a missing measurement.
No new controller is designed until these numbers have been reviewed.

## 2026-11-01 - ADOPTION RECOMMENDATION: boot plan all-CPU with `STRATA_GLM_CPU_RETIME=0`

**The recommendation:** ship the committed planner (`d098fa7`) with the **all-CPU boot plan** and
**`STRATA_GLM_CPU_RETIME=0`**. Quality-gated; **+6.8 / +11.8 / +23.9% at n=1/2/4 steady**. The n=4 arm is
a **deployment smoke test, not a safety gate** (see below). No retime work, no `RETIME_V2`
dependency, no saturation controller.

**Why retime is disabled - and the reasoning matters more than the conclusion.** Retime is **not** "pure
downside". If the CPU genuinely degrades, retreating toward staging **is the correct response** - if the
CPU really is slower, more PCIe staging is what the cost model should choose. **The defect is that a
retreat is UNRECOVERABLE**: it fires on a condition that is usually transient and can never come back when
the transient passes. So the controller's only upside requires recovery, which it does not have.

**The accepted risk, stated explicitly:** with retime disabled, **the box keeps the all-CPU plan even if
the CPU genuinely degrades.** Judged acceptable here, for three reasons:
1. a competing workload on this box is visible and transient, and standing policy is that no CPU work runs
   during measurement anyway;
2. an unrecoverable retreat would outlast the transient that caused it, so the enabled controller is
   **worse than the disabled one even in its own best case**;
3. it is a lab box - a degraded-CPU window costs throughput, not correctness.

**REOPEN CONDITION: if recovery is ever built, this decision must be revisited.** The controller becomes
worth enabling exactly when a retreat can be undone.

**Reachability, which is what settled it:** the old controller's window measures `cpu_us / cpu_experts` -
one expert on one thread - while `t_eff` enters only the planner's `waves = ceil(kk / t_eff)`. So
concurrency changes the plan's arithmetic, not the measured price: `ms` stays ~0.35 at n=4. A retreat
therefore needs genuine per-expert lane degradation, which needs oversubscription (32 on 16), which needs
two deliberate overrides (`STRATA_GLM_CPU_LANE_MAX` to size the pool **and** a request's `cpu_threads`,
since `set_active` clamps to pool size and the pool's single `active_` is REPLACED, never summed).
Corroboration: n=4 showed the largest win and no retreat line, and a retreat would have shrunk it.

**Carried forward, unexplained, and it must be seen beside the wins:** the **replicated -21.9% n=1 cold-run
cost** (steady +6.8%), reproduced across four boots, with host-side warming as the named but never
confirmed candidate. That is a real cost on the first request after a boot.

## 2026-11-01 - THE STANDING DISPOSITION TEST for every feature (owner's rule; supersedes "ship it disabled")

> **"If the feature works, we should use it. If the feature is broken, then fix it. If the feature just
> doesn't provide any improvement or open something else to be improved then get rid of it."**

Three outcomes; **"ship it disabled" is not one of them.** Applied to the two features built this session:

- **`RETIME_V2` -> REMOVE** (third branch). Its detector works, but the condition it protects against
  needs two deliberate env overrides to produce (`LANE_MAX` to size the pool **and** a request
  `cpu_threads`), so fixing the trigger would buy protection against a state the box cannot enter.
- **The residency value term -> FIX, do NOT delete** (second branch - broken, not pointless). Its subject
  is Aaron's original design point (stage-vs-compute scheduling), and its two defects are named and
  fixable: `R_est = hits/promotions` is **selection-biased** (measured while staging only deep routes),
  and **eviction/displacement is unpriced** in a pool ~96% occupied. Direct evidence it is wrong rather
  than unproven: `R_est 9.28` says stage everything while measurement says all-CPU wins by up to +23.9%.
  **A term that contradicts a measured result is a bug with a known sign.** The flag may stay only while
  it is actively being worked - it is not a resting place.

**Correction recorded:** an earlier instruction in this file lumped the residency term in with the
removal. That was wrong; the rule distinguishes broken (fix) from useless (remove), and the residency
term is the former.

**The same rule selects two follow-on work items:** fix the residency term's two defects (eviction cost,
unbiased `R_est`), and **re-test the MTP drafter** on the ranked-KLD gate - +15.8% decode warm at accept
0.945, parked on a token-equality gate since proven invalid on an engine that samples at temp 1.0. By the
rule that is a feature that may well WORK and is going unused; spec-decoding's correctness claim is
distributional, so KLD is the right and only test we have. Largest unclaimed win on the box.
