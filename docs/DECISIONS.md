# Decisions

> **Scope:** Maya decisions only. llama.cpp-era closures live in the lab copy at
> `G:\AI\llama-server\docs\DECISIONS.md` — check there before assuming an old decision was never made.

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

## RE-DECIDED 2026-10-30 — `6ae276e`'s deletion of the BF16 KDA-output fallback is reversed

`6ae276e` (28 Oct) removed a 21-line fallback in `multi_rows` that decoded a BF16-output batch one row at a time,
replacing it with an `mv_nt` pre-pass that (a) referenced three symbols it never defined, so nothing from `6ae276e`
onward compiled, and (b) did not cover the single-job `mv()` calls. It also deleted the comment recording why:
`rows_layers`' KDA output projection is BF16 in this pack, the rows path only has `kda_rec`'s q8_1 activations, and
`n == 1` never showed it because it stays on `mv()`.

Decision: **the fallback is restored** (verbatim, comment included) and `mv()` also gained a q8_1→float pre-pass so a
single BF16 job works. The fallback is the correct behaviour until the KDA output job is given float activations; the
cost is the shared weight reads, and its arithmetic equals the `n == 1` step's per row.

**Consequence, recorded so it is not discovered late:** with this pack the fallback fires for every batch, so the
rotation does not run at `n > 1`. See `docs/STATE.md` for the re-ranked options (repack first).

## CLOSED 2026-10-30 — a textual union is only valid for pure insertions

Resolving a rebase conflict by concatenating both sides is valid only when both sides are pure insertions into
unchanged surrounding text. It is invalid the moment either side **restructures or deletes** inside the hunk — that is
a three-way merge, which no textual union can express. Brace balance is not the test; purity of insertion is.

This cost a build: on the v1.1.40→41 replay one hunk had upstream restructuring a block our side had edited inside, and
concatenation duplicated a block, orphaned an atomic outside its loop, and left an unclosed brace that swallowed the
file. Any helper that auto-resolves must **refuse when either side contains a deletion**.

Second rule from the same episode: a conflict that compiles is not evidence. A union that duplicates a statement with
balanced braces compiles perfectly and is silently wrong. Audit by **count** against the last verified build — every
marker line must appear as often as it did there; up is a union artefact, down is a lost edit.
