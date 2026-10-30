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

> **CORRECTION (same day, later) — THERE WAS NO LAP, AND THE LAP CLAIM IS WITHDRAWN.**
>
> `next - kRingSize` in the parked slot is its **previous occupant**: that is what an entry that has *not yet been
> published* looks like, by construction. A genuine lap is the opposite — the device would have written something
> **newer** into that slot, so it would hold `N`, or `N + R`, or anything `> N`. Both readings here (`-64` at 64 slots,
> `-256` at 256) therefore say the same thing: **the producer stopped publishing.** The offset tracked `kRingSize`
> because arithmetic forces it to, not because the ring size mattered — which is also why the larger ring stalled at
> seq 257 instead of 2773: a bigger ring changes which seq is in flight when the producer dies, not whether it laps.
>
> Consequences, recorded so they are not re-derived:
> - **No lap has ever been observed on this path.** A true lap is `(int)(sq - next) > 0`, which the pre-existing resync
>   branch at `glm_fast_path.cu` already handles.
> - **Backpressure is NOT the fix and is deliberately unwritten.** It bounds a producer that runs ahead; ours stops.
> - `kRingSize = 256` stays on its own merits (`n_rows x layers` headroom, 4.3 MB pinned) but it is a sizing change,
>   not a fix, and the failure point moving 64 -> 256 is the evidence that it is not one.
> - The `LAPPED` guard written here was a **false positive generator** (`sq < next` is the ordinary unpublished case)
>   and has been replaced by a stall report that quotes the two numbers which actually separate the cases: the host's
>   hand-out count and the **device's own publish mark** (`MoeDev::dev_pub`). Their difference is the number of launches
>   that never published — the answer, with no inference left in it.

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

## 2026-10-30 — the producer/consumer framing is dead; the waiter's release condition is the question

Boot 1's arithmetic killed it. `pub_ch` counts **host hand-outs** (`md_h.hseq = ++F->seq_h`), not device publishes:

```
ch0 published 2268 + ch1 84 = 2352 = H (host hand-outs)
ch0 answered  2255 + ch1 70 = 2325
service waiting for seq 2326   (2325 + 1)
```

So the `BEHIND 13/14` read all day was **the host's enqueue lead over device execution**, not a service backlog. The
service has consumed every request that reached the ring and is parked on the next one, which no kernel has published.

`H - T` (host hand-outs vs highest seq in the ring) therefore cannot discriminate: a stream is in-order, so once a
kernel wedges, every launch the host enqueued behind it has not run either. `H - T` is forced non-zero in any stall and
zero in none. It was built, measured once (`H = 2352`, ring top `2352`, i.e. zero — a boot whose step died before the
next hand-out) and removed rather than carried.

A publish-mark word written by the kernel was also tried and **removed**: it read 0 in a boot that had answered 2255
requests, and an illegal memory access appeared in the same build. A diagnostic that can fault is not worth its reading.

**What is left, and it is narrow:** a device-side waiter (row 1, layer 31/32 on `cs2`) blocked although everything it
could be waiting for has already been answered. The release is not "the answer arrived" — it is `*moe_wseq(d)`
**changing**: `moe_wait` reads `s_need = *moe_wseq(d)`, block 0 watches for the response and clears the word to 0, and
every other block spins on `while (*moe_wseq(d) == s_need)`. A waiter therefore sits forever if the word it polls is not
the word that gets cleared - a per-channel indexing question between the publish (`*moe_wseq(a.d) = mask ? seq : 0u`)
and the clear.

Instrument: at a stall, dump the eight `wait_seq` words against the eight `resp[ch].seq` values, beside the parked
`next`. `wait_seq[1]` holding a seq whose `resp[1].seq` already matches it means the answer landed and the clear never
happened. The `wait_seq` read is issued on its own stream and polled for at most 2 s, because a blocking host sync is
the deadlock the diagnostics note warns about; if it cannot complete the line says so.

Unit correction on the counters above: `pub_ch` is HOST HAND-OUTS and `ans_ch` is ANSWER-WRITES. They are different
quantities - a request that needs no answer never increments `ans_ch` - and their difference is **not** a backlog. The
service's position is `next`, which the stall line prints separately, and in every boot measured it reads `H + 1`: the
service consumed every hand-out and is parked on a seq no step ever handed out. The output now says "hand-outs" and
"answer-writes" instead of "published"/"answered", because the word BEHIND sent two people looking for a queue that does
not exist.

## 2026-10-30 — the sixteen words: the wedge is NOT the answer handshake

Boot 3 (drain, lowest incomplete layer 38, both cursors already at il 45):

```
service waiting for seq 2760   (slots hold 2504.. = previous occupants; no lap)
wait_seq   0=2758  1=0  2=0  3=0  4=0  5=0  6=0  7=0
resp.seq   0=2758  1=2683  2=0  3=0  4=0  5=1032389371  6=0  7=0
```

Three facts, in order of weight:

1. **`wait_seq[1] = 0`** - row 1's channel has NO pending wait word. `*moe_wseq(a.d) = mask ? seq : 0u`, so this says the
   stuck row's last route had **nothing to wait for from the host** (mask 0): no CPU-lane and no disk expert requested an
   answer. The wedged stream therefore cannot be parked in `moe_wait`, and the answer handshake is exonerated for the
   stuck layer. By the criterion set earlier today, the suspect moves to the **arrival path** - an `ev_fill` / `ev_pf`
   event that never fires (a `cudaStreamWaitEvent` on the row's stream is a wait, not a kernel, and reports exactly as
   "a kernel from this step is still running").
2. **`resp.seq[5] = 1032389371`** - garbage in a slot that was never written. No waiter polls channel 5 today, so it is
   not the fault, but it is an uninitialised-memory tell worth removing on principle.
3. **`resp.seq[0] = wait_seq[0] = 2758`** - the answer landed on channel 0 and the word was never cleared back to 0.
   Harmless at a step's end only if nothing waits on it afterwards; that assumption is now written down rather than
   left implicit.

Next instrument: **stage events on the row's stream** (route -> fetch -> mixer -> FFN -> head) so the first event that
never retires names the stage, not just the layer. The drain's layer sweep already proves the granularity is needed:
layer 38 alone cannot distinguish a parked wait from a stuck kernel.

## 2026-10-30 — the response array was a 64-byte buffer (pre-existing; found by reading the allocator)

`FastState::resp_h` is the 8-slot array the service writes each answer into. `MoeResponse` is ~17.3 KB (its
`float cpu_part[4096]`), so the array is ~138 KB. The allocation block did this:

```cpp
cudaHostAlloc(&hp, sizeof(MoeResponse) * 8, Mapped);   // ~138 KB, zeroed
cudaHostAlloc(&hp, 64, Mapped);                        // hp REPLACED by the 64-byte channel-seq words
F->need_seq_h = (unsigned int*) hp;
F->resp_h     = (MoeResponse*) hp;                     // <- resp_h now points at 64 bytes
```

So every answer the service wrote (`Rp = F->resp_h + (rq->ch & 7)`, then `Rp->part[...]`) landed **past the end of a
64-byte pinned allocation**, and the device reads that same region through `F->md.resp`. Observed consequences:

- `resp.seq[5]` read `1032389371`, `998340161`, `3178233609` in three consecutive boots - not uninitialised memory (the
  array is memset) but whatever the write ran into;
- an **illegal memory access** in one boot;
- and, unavoidably, corruption of whatever pinned allocation follows, which is shared with the device.

**It is pre-existing**: `15ff4f1` has the same two lines at 976 and 984, so it is in every build we have called
verified tonight, including the one whose 796 rotor entries were treated as the working baseline. It is not a replay
artefact.

Fixed by giving the response array its own handle (`gf::MoeResponse* resp8`) and its own device pointer, with
`need_seq_h` keeping the 64 bytes:

```cpp
cudaHostAlloc(&resp8, sizeof(MoeResponse) * 8, Mapped);  memset;  F->resp_h = resp8;  get dev ptr;  F->md.resp = ...
cudaHostAlloc(&hp, 64, Mapped);                          memset;  F->need_seq_h = hp;  ...
```

**What this does NOT yet establish:** that it is the wedge. It is a real memory-corruption bug on the answer path, of the
kind that produces timing-dependent faults, and it is now fixed and arm-able. Three boots decide whether the wedge was a
symptom of it.

Also in this build: the drain polls **both** streams (`cs` and `cs2`) and the failure line names which is `NotReady` -
"row 1's stream" has been in every error line today because that was the only stream the loop looked at.
