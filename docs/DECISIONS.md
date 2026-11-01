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
