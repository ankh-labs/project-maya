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
in 10035 slots. Slot stride works out at **~2.8 MB**, and **42** MoE layers (45 layers less `dense_lead = 3`). So raising
`kSpares` 3 -> 7 would reserve `4 x 42 = 168` more slots, ~**0.47 GB** - which fits inside the 1.23 GB free, but only
just, and it competes with the 13.2 GB the prompt path borrows. (The earlier "+1.6 GB" figure was wrong twice: it used 45
layers and an 8.8 MB stride that is the *pinned prefetch buffer*, not a VRAM slot.)
