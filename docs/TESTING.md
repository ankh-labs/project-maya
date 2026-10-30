# Maya battery log (append-only)

## 2026-10-30 12:10 — v1.1.41 rebase, label `v1.1.41-rebase`

Engine: v1.1.41 build (`56ced70`), started FROM THE CONFIG with no shell env. Pack `pack-repack-20261027`.
kv=fp16, slots=4064, kv_gb=1.26, vram_hit=0.672.

| case | result |
|---|---|
| needle 2k | PASS — 2489 tok prompt, 92.46 s, 0.17 tok/s, answer 7741 |
| needle 6k + decoys | PASS — 3954 tok, 175.63 s, 0.07 tok/s, 7741 and no decoy |
| verbatim copy | PASS — 48 tok, 4.26 s, 8.68 tok/s, exact |
| word position 9 | PASS — 36 tok, 2.37 s, 4.64 tok/s, iota |
| arithmetic 3 steps | PASS — 70 tok, 11.95 s, 16.65 tok/s, 6345 |
| code + output | PASS — 47 tok, 7.16 s, 14.1 tok/s, def + 140 |
| long summary facts | PASS — 3463 tok, 133.98 s, 0.16 tok/s, towers/pressure/Okonjo all true |

SUMMARY: **7/7 passes**, mean 6.35 tok/s (range 0.07–16.65). JSON: `battery-v1141.json`.

**This is a gate result, not a throughput measurement.** No paired v1.1.40 run of the same battery in the same boot
exists, and a mean over cases that are prefill-dominated is not comparable to a decode figure (Oct 27 production:
23.8 tok/s). v1.1.41 throughput: **unmeasured**.

## 2026-10-30 — the de-fused rotation: correct, but short-lived

**The de-fused rotation computes correct concurrent output; it does not yet survive more than a few steps.**

What is established (same-boot comparisons, one boot each):
- Row 0 at n=2 reproduces the n=1 baseline token for token at the same positions (57/8484/77009 at pos 25..27), BF16 layer 0 included.
- Row 1 draws faithfully from its own logits: hashes change every step (2a53b92c, 52c721a1, da74abf0) and each token IS that step's top1 (320 gap 3.9, 18698 gap 0.24, 220 gap 2.06). The earlier "constant 320" was two steps of boots that died at step 2.
- Per-row strides are exact: head_x 16384 (E*4), head_xq 4608 ((E/32)*36), logits 619520 (n_vocab*4) at E=4096, n_vocab=154880.
- Row 1's CPU-lane path works: 112 ch=1 requests answered with an advancing seq (layers 3..8, 26..31).
- Residuals chain correctly across steps for both rows (each step's post-hash is the next step's pre-hash).

What fails: one kernel never retires. Localised by the ev_layer sweep — lowest incomplete layer 33 of 0..44, on cs2 (row 1's stream), with both cursors already at 45. The service's last logged ch=1 answer is layer 31, two layers before the stuck layer, so the answer stream stops rather than never starting.

Falsified today (each by a one-line instrument, recorded so they are not re-walked): parity aliasing of pf_buf (spread never exceeded 1; occ/reo deferrals 0); the round-cap fail-open (gate does not fire, all_done true); stream-fetch raw channel reads (flag off, unchanged); "row 1's residual is never written" (all three hashes differ); the head's i vs R(i) vs rb asymmetry (latent only; strides exact); "no ch=1 evidence" (112 answered).

## 2026-10-30 — the route serialization, three boots (repeated, per the standing rule)

`ev_route` (route launches ordered so publication order equals seq order) is in the tree and builds clean.

| boot | tokens | rotor entries | drain timeouts | non-finite | floor |
|---|---|---|---|---|---|
| 1 | -1 + -1 | 5 | 2 | 0 | 0 |
| 2 | -1 + -1 | 2 | 2 | 0 | 0 |
| 3 | -1 + -1 | 14 | 2 | 0 | 0 |

**3/3 still fail, and the ranges overlap the pre-fix range (2..6), so the serialization is NOT established as an
improvement** - it is within this bug's normal variance. It is also not established as useless: it removes one real
mechanism and costs one event per route. What it cannot be is evidence.

The shape is identical in all three: one kernel on `cs2`, **both cursors already at `il 45`**, and the service's last
`ch=1` answer lands about **two layers before the stuck layer** (last answer layer 11, stuck layer 13 in boot 3; last
answer 31, stuck 33 in the earlier boot). The answer stream stops; it does not fail to start.

### The two failure modes are one fault seen at two timeouts

The 90 s drain boot did not wedge at all - it failed `NON-FINITE SCORE at layer 5` instead. Read with the shape above,
that is not a different bug: the wait kernel is bounded (~60 s), so a request that is never answered has two visible
outcomes. Give the waiter 20 s and it is a "drain timeout"; give it 90 s and it returns an answer-shaped garbage that the
router rejects as non-finite. The variable is how long the observer waits, not which fault occurred.

So the target is the **answer path** - a request for which no answer arrives - and not the rotor, the slicing, or the
strides: those are measured correct. Of the two options LAYER_PIPELINE.md:307 names, the route serialization is now
tested and does not remove it. The remaining documented option is the **per-channel ring**: the service answers each
channel on its own, with no global seq order to violate.

Before building that, the cheap instrument is a **published-but-unanswered counter**: increment when a route publishes
seq N on channel c, and when the service answers N on c. The difference at the wedge names the unanswered request
directly, which is the fact the last three boots could only infer from silence.

## 2026-10-30 — the lap: kRingSize was a single-row constant

The ring quote settled it in all three boots, as a uniform negative offset rather than a lost request:

| boot | service waiting for | its slot holds | difference |
|---|---|---|---|
| 1 | seq 2773 | 2709 | **-64** |
| 2 | seq 2620 | 2556 | **-64** |
| 3 | seq 2424 | 2360 | **-64** |

`-64` is exactly `kRingSize`. The device outran the service and wrapped the ring; the resync guard at `glm_fast_path.cu:4441`
only recognises being *ahead* (`(int)(sq - next) > 0`), so a lapped service polls a slot that will never hold its seq
again - forever, which surfaced as a 20 s "drain timeout" or as NON-FINITE depending on the observer's window.

Root cause is arithmetic, not timing: **the ring was sized for one producer.** The token path publishes at most one
route a layer (45 <= 64, cannot lap); the rotation publishes one a row a layer (45 x 2 = 90 > 64, laps as soon as enough
layers are resident and a row does not wait for its answer - `if (empty) continue` at `glm_rows.cu:971` skips the wait
entirely). That is why it never appeared at n=1, and why it is indifferent to streams, serialisation and BF16.

Changes:
- **`kRingSize` 64 -> 256.** The bound is `n_rows x layers`, so 90 for the rotation; 256 is 2.8x that and still covers
  the window paths. `sizeof(gf::MoeRequest)` is ~16.8 KB (dominated by its `float x[4096]`), so 256 slots is 4.3 MB of
  pinned host memory; 512 (8.6 MB) would cover the 8-row maximum if the rotation is ever extended past n=2.
- **The lap fails closed.** `(int)(sq - next) <= 0` on the parked slot, with `seq != 0` (an empty slot is not a lap),
  prints `LAPPED: the service waits for seq N and its slot holds M` and sets `svc_lapped`, which every step-failure
  message now quotes. Today that would have found this in the first boot instead of the fifteenth.
- **Backpressure demoted to a documented option, not built.** Throttling the producer to fit a 64-slot ring would trade
  away the latency the rotor exists for, to work around a constant.

## 2026-10-30 — raising kRingSize moved the failure, it did not remove it

Three boots at 512 tokens with `kRingSize` 64 -> 256 and the lap guard in:

| boot | tokens | rotor | face | lap line |
|---|---|---|---|---|
| 1 | -1 + -1 | 3 | non-finite | `waits for seq 257, slot holds 1 (256 slots)` |
| 2 | -1 + -1 | 10 | drain | same |
| 3 | -1 + -1 | 14 | drain | same |

`257 - 1 = 256 = kRingSize`, exactly as `2773 - 2709 = 64` was with the old value. **The lag is always one full
ring**, whatever the ring's size. So `kRingSize` was not the cause and the arithmetic argument that justified raising it
("at most 90 in flight") was wrong: the host enqueues a step's launches asynchronously and can run several steps ahead
of the service, so in-flight publications are **not** bounded by `n_rows x layers`.

What the constant did buy: the failure point moved from -64 to -256 and the shape stayed identical, which is what proved
the producer is unbounded. That is a clean falsification of "raise the constant", by measurement, in three boots.

What is now established:

- the producer's lead over the service is unbounded from the ring's side (it grows until it laps);
- a lap is permanent because a lapped service polls a slot that cannot hold its seq again;
- the **guard is the win**: boot 1 failed in 3.1 s instead of 50.9 s, with one line naming the seq and the slot.

**Therefore backpressure is required, not optional** - and the earlier ranking (demote it, it costs latency) was based on
the same wrong assumption about in-flight size. The form that does not cost pipeline latency: before the rotor hands out
a seq, require the service to be within a window - `F->seq_h - F->processed < kRingSize / 2` - yielding only while that
window is closed. When the service keeps up it never binds; when it falls behind it stops the producer instead of lapping.
