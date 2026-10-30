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
exists, and a mean over cases that are prefill-dominated is not comparable to a decode figure (the Oct 27 deployed build:
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

### The aliasing, stated fully (2026-10-30, verified independently)

At `15ff4f1` lines 970-986 the 138 KB response allocation is made, memset, and **its handle is then discarded**: `hp` is
overwritten by the 64-byte allocation and *both* pointers are taken from it.

```cpp
F->need_seq_h = (unsigned int*) hp;          // the 64-byte buffer
cudaHostGetDevicePointer(&dp, hp, 0);
F->md.need_seq = (unsigned int*) dp;
F->resp_h     = (gf::MoeResponse*) hp;       // THE SAME 64-byte buffer
cudaHostGetDevicePointer(&dp, hp, 0);
F->md.resp    = dp;                          // THE SAME device pointer
```

So `resp` and `need_seq` are one object on the host and on the device, `MoeResponse::seq` (its first member) *is*
`need_seq_h[0]`, and the 138 KB block is leaked and used by nothing.

Two readings from tonight are retired by that, both of them mine as much as anyone's:

- `resp.seq[0] == wait_seq[0]` was **structural aliasing**, not a satisfied waiter, and not an uncleared word;
- `resp.seq[5] = 1032389371` is byte 20 of a 64-byte buffer read as the sixth response's `seq`.

And every per-channel conclusion needs re-reading: `Rp = resp_h + (rq->ch & 7)` steps ~17.3 KB per channel while the
channel's *seq word* sits 4 bytes in, so the "per-channel answer slots" were never per-channel - they were one 64-byte
region with every write but channel 0's landing outside it. The 112 `ch = 1` answers prove the code path ran, not that
the slots worked.

**Consequence for the record: no build before `bf3b16f` had a working answer path - including every build tonight called
verified, and including the `cd52717` baseline whose 796 rotor entries were treated as the known-good state.** If the
allocations separated at `bf3b16f` make `n = 2` survive, that baseline was lucky rather than correct.

## 2026-10-30 — STREAM_FETCH=0 re-tested three times: the stream-fetch path is dead

| boot | result | face |
|---|---|---|
| 1 | -1 + -1, 2.7 s | `NON-FINITE SCORE` at layer 24 (x2) |
| 2 | -1 + -1, 3.2 s | **`row lane (cs2): an illegal memory access was encountered`** |
| 3 | -1 + -1, 22.9 s | drain: `NotReady: cs(row0) cs2(row1)`, layer 40, `ev_fill` generation stale, `wait_seq 1=0` |

With the flag off the stream mask is zero and `moe_stream_copy` is never entered, so the mask leak cannot be reached -
and 0/3 clean means the stream-fetch path is **not** the cause. The earlier one-boot "falsification" is now a proper
three-boot one, reached from the other direction: not "it still fails, so it is innocent" but "it fails with the
path disabled, so disabling it proves nothing".

The per-channel + single-read fix (mask and `s_sm` through `moe_wseq(d)` / `moe_resp_ch(d)`, one read per field)
**stays** - it removes a real cross-channel leak and a host-write race that were genuine - but it is a correctness fix,
not this fault's fix, and it will be built and run with the flag ON so the instrument sits on the path the live box uses.

**New signal worth holding:** the illegal memory access has now appeared twice today, both on `cs2` and both after a
rotor run of several steps. It is reported through `cudaStreamQuery` on that stream, so it is a genuine device fault, not
a diagnostic artefact.

Next: **stage events** per row per layer (route -> fetch -> mixer -> FFN -> head), so the first event that never retires
names the stage. The layer sweep has already shown that a layer number alone cannot distinguish a parked wait from a
stuck kernel, and both compute streams being `NotReady` with no host wait outstanding puts the answer in that
distinction.

## 2026-10-30 — THE MISSING HUNK: row 1 has been publishing on channel 0 all evening

The differential that should have been run this morning — **`cd52717` (the tree that ran 796 clean rotor entries) against HEAD** — found it in one line:

```cpp
cd52717 :  m.wait_ch = R(i) & 7;     // row_base_ + i -> the row's OWN channel
now     :  m.wait_ch = i & 7;        // i within THIS call
```

The rotor calls `rows_layers(rw + r, 1, ...)`, so `i` is **always 0** in that lambda: row 1 has been asserting channel 0
for every route. Every neighbouring line in the same lambda (`m.plan_ptr + 8 * R(i)`, `m.fetch_src`, `m.cpu_part`,
`m.scratch`, `m.cpu_seq`) still uses `R(i)`, and the comment above it says *"this row's OWN handshake channel"* - so this
is a single reverted line, exactly the shape the v1.1.40 replay produced elsewhere (it also dropped the whole
`q8_1_to_float` block, which `55192e5` correctly described as "port ... the merge dropped").

Why it explains every symptom we chased tonight:

- **`wait_seq[1]` always read 0.** Nothing ever wrote channel 1, because no route ever claimed it. We read that as "row 1
  has nothing to wait for" and built three theories on it.
- **Both rows answered into `resp[0]`.** The service writes slot 0 in seq order, so the later answer overwrites the one
  an earlier waiter is still spinning for. A waiter that misses its value spins to `moe_wait`'s ~60 s ceiling.
- **The two-window shape**: through a 20 s drain that is a "drain timeout"; through a 90 s window the waiter gives up
  and the router sees `NON-FINITE`. One fault, two observers - as concluded, but the mechanism is this line, not the
  protocol.
- **Both compute streams `NotReady`, no host wait, prefetch lanes idle**: both rows were parked in `moe_wait` on the same
  channel, one of them chasing a value that had already been overwritten.
- **Indifferent to serial/overlap, to BF16, and absent at n = 1**: with one row, `i` and `R(i)` are both 0.

The `resp_h`/`need_seq` aliasing is **also** present at `cd52717`, so that find stands, and the 796-entry soak really did
run with a live heap overflow - fix both.

### Method correction

The count audit compared `15ff4f1` against the rebased tree: **both post-rebase**, so it could not see this. The audit
that finds dropped code is against `cd52717` - the tree with evidence of working. `pub_ch`-style counts, three boots and
log-verified arms all still stand; the *reference* was wrong.

### Correction on the restored `wait_ch` line: real regression, not this wedge

Three boots with `m.wait_ch = R(i) & 7` restored and the response array separated: **0/3 clean** (one drain at layer 29,
two NON-FINITE at layers 34 and 27, 3-4 s).

Why the restored line did not move the fault, stated plainly:

- **`ch 1` hand-outs were already non-zero before the restore** (84..756 across earlier boots), and `resp.seq[1]` already
  held real sequences (2683, 3623, 3493). So the per-channel path was already live for row 1: the route kernel's `MoeDev`
  comes from `fast_moe_experts`, which sets `md.wait_ch = r & 7` itself. The `row_dev` line is a *second* assignment on a
  path the rotation does not take.
- So the reverted line is a **genuine regression restored** - every neighbouring line in that lambda indexes by `R(i)` and
  the comment says "this row's OWN handshake channel" - but it is not the wedge.

`wait_seq[1] = 0` also reads differently now: `*moe_wseq(a.d) = mask ? seq : 0u`, so a zero means the last route on that
channel had **nothing to wait for** (mask 0), which is ordinary, not an unexercised channel.

### What remains, unchanged by any fix so far

Every drain boot tonight, across every build, reports the same five facts together: both compute streams `NotReady`,
prefetch/copy lanes idle, `ev_fill` for the stuck layer **recorded by nobody this step**, `ev_layer` recorded by the row
that was waiting, and no wait word outstanding. The stuck layer wanders (17, 29, 34, 40, 41, 43). That is a stream
waiting on an event whose arrival never happens, on a path where the host has nothing left to answer.

Next instrument is the one already designed and not yet built: **stage events per row per layer** on the row's own
stream, so the first event that never retires names the stage rather than the layer.

## 2026-10-30 — the reference build works: the regression is inside cd52717..HEAD

Built the pre-rebase tree (`cd52717`, tag `backup-pre-20261030-0835`) through the working pipeline and ran the same arm
the soak used: `run-maya-cfgonly.bat`, the same config, **zero shell keys** (verified empty), two concurrent requests,
512 tokens, three boots.

| boot | tokens | finish | time | rotor entries | drain | non-finite |
|---|---|---|---|---|---|---|
| 1 | 44 + 176 | stop / stop | 9.4 s | **43** | 0 | 0 |
| 2 | 43 + 177 | stop / stop | 9.2 s | **42** | 0 | 0 |
| 3 | 48 + 129 | stop / stop | 8.0 s | **47** | 0 | 0 |

Coherent answers in all six replies, ~24 tok/s aggregate - the morning's 23-25 tok/s. **Clean 3/3, rotor engaged
throughout.**

So: the rotation worked before the rebase, the regression is inside `cd52717..HEAD`, and the search is now a bounded
bisect rather than an open hunt. Two bounds on what that means:

- the morning tree carries the `resp_h`/`need_seq` aliasing and **works** - both sides of the handshake use the same
  undersized pointer, so answers land while adjacent pinned memory is corrupted. The bug is real; it was not the
  functional failure. Separating them (bf3b16f) is still correct but is a bisect variable, not a proven neutral.
- the rotor was genuinely running here (42-47 entries), unlike `15ff4f1`, where the BF16 fallback pre-empted the
  rotation and a "coherent n=2" measured the fallback.

Note: rebuilding `build/` on `cd52717` discarded the instrumented binary; `788b1da` is committed and rebuilds in one pass.

### Two build facts worth keeping

- A fresh build directory on this tree never had the toolchain environment (`nvcc fatal: Cannot find compiler 'cl.exe'
  in PATH`), so `fatbinary fatal: Could not open input file '*.ptx'` there was a *build* failure, not a code fault.
  Building the reference commit through the existing pipeline was the cheap correct move.
- `sccache` layered under `nvcc` is a known hazard for nvcc's intermediate `.ptx`/`.cubin` temporaries; for a one-off
  reference build, empty the `CMAKE_*_COMPILER_LAUNCHER` variables rather than tuning `-j`.

## 2026-10-30 — single-variable test: the resp_h separation is NEUTRAL

`cd52717` (the tree that passes) plus **only** the `resp_h` / `need_seq` separation - `1 file changed, 9 insertions(+),
5 deletions(-)` - same config-only arm, three boots:

| boot | tokens | finish | time | rotor | drain | non-finite |
|---|---|---|---|---|---|---|
| 1 | 44 + 172 | stop / stop | 9.5 s | **43** | 0 | 0 |
| 2 | 42 + 134 | stop / stop | 8.1 s | **41** | 0 | 0 |
| 3 | 47 + 136 | stop / stop | 8.5 s | **46** | 0 | 0 |

**Clean 3/3.** Separating `resp` from `need_seq` does not break the working rotor: the aliasing was self-consistent and
its removal is genuinely neutral. `bf3b16f` **stays**.

That closes the strongest prior we had - the only edit either of us made to a contract that was live on a passing tree -
and it puts the regression in the replay itself (`cd52717..HEAD`). Next: the **full-width deleted-lines audit** of
`cd52717` against `HEAD` over the scheduler files (not marker-filtered: the marker audit found `wait_ch` and could miss
others), then a bisect over the replay only if that reports nothing but upstream's own changes.

Withdrawn, recorded so it is not re-derived: "no build before `bf3b16f` had a working answer path" is false - the answer
path worked; the *adjacent* pinned memory was what the aliasing trampled.

## 2026-10-30 — the per-row revert family: four found, all restored, fault still present

The full-width audit and its follow-up sweep of `cd52717` against `HEAD` found **four reverts of one family** - per-row
code resolved back to its single-row form:

| # | reverted | what it did |
|---|---|---|
| 1 | `m.wait_ch = R(i) & 7` -> `i & 7` | row 1 published on channel 0 (inert: `fast_moe_experts` sets it again) |
| 2 | `if (R(i) > 0 && md.cpu_seq ...)` -> `i > 0` | row 1 never got its own CPU-lane `cpu_seq` / `ans` slot |
| 3 | `h.part`/`h.counter` per-row -> shared `F->part`/`F->counter`; `h.xq` by `i` | **both rows mixed into one mHC scratch** - the non-finite residual the old comment describes |
| 4 | mHC scratch allocated only under `rows_batch_on()` | null-plus-offset per-row pointer -> **illegal access** |

All four are restored (`57c5ebe`, `e7e058a`, `fdcfdc2`). A mechanical sweep of every `R(i)`-indexed line and every
`f32`/`take` allocation line in `cd52717` against `HEAD` now comes back **clean** - no remaining revert of this family and
no missing allocation gate.

**Three boots after the fourth restore: still 0/3** (non-finite at layer 22; two drains at layers 36 and 15; rotor entries
12 / 4 / 34 - the highest yet on HEAD, and no more illegal accesses, so #4 was real). The fault therefore is **not** this
family, or not only it: what remains is upstream's own work (50 commits of v1.1.40/41) interacting with the rotation.

Known-good end of the bisect, established tonight: `cd52717` clean 3/3 (rotor 41-47) and `cd52717 + the resp_h
separation` clean 3/3 - so the bisect over the replay has a verified low end, and `bf3b16f` is proven neutral.

Next: bisect the replay (`cd52717..HEAD`), three boots a step. The rotor-adjacent upstream changes to look at first are
the ones the rotor consumes: the six raw `d.resp` reads in `moe_stream_mask`/`moe_fetch_kernel`, `route_err_d`, and the
`ev_fill`/`ev_layer` recording path - all of them reached only when the rotation runs, and all of them invisible to a
single-row test.

## 2026-10-30 (late) — both v1.1.41 rotor-adjacent features excluded by measurement

After the four per-row restores (`57c5ebe`, `e7e058a`, `fdcfdc2`, `73133ca`), two flags were tested three boots each on
this tree. Both are **new in v1.1.41** - neither exists at `cd52717`:

| flag | what it controls | boots | result |
|---|---|---|---|
| `STRATA_GLM_STREAM_FETCH=0` | `moe_stream_mask` / `moe_stream_copy`, the `s_sm` branch in `moe_fetch_kernel` (all absent at cd52717) | 3 | **0/3** (rotor 3 / 4 / 7; one drain, one non-finite, one drain) |
| `STRATA_GLM_ROUTE_FAST=0` | the `a.fast` pre-write that indexes the ring by `*a.d.seq + 1`, plus the reordered reads | 3 | **0/3** (rotor 2 / 21 / 52; non-finite x2 and an illegal access) |

So neither upstream feature is the fault on its own. The `*a.d.seq + 1` pre-write remains suspicious *in principle* (it
is a shared device counter where a per-row one is required - row 1 can write its 16 KB FFN input into a slot row 0
already published), but with `ROUTE_FAST` off entirely the failure persists, so it is not sufficient either.

State of the bisect: low end verified (`cd52717` clean 3/3; `cd52717 + bf3b16f` clean 3/3), high end failing, four of
our own reverts restored, two upstream feature flags excluded by three boots each. What is left is the **replay itself**:
the interaction of v1.1.40/41 with the rotation, three boots per step.

## 2026-10-30 (late) — git bisect returned a false result, and the shared-state filter came back clean

### The bisect that answered nothing

`git bisect start HEAD cd52717` finished with exit=0 and "bisect found first bad commit" = **the endpoint itself**, with
`0` step-log lines: the test never ran.

```
git merge-base --is-ancestor cd52717 HEAD   -> false
cd52717 reachable only via: backup/feat/rows-de-fused-pf-20261030-0835, tag backup-pre-20261030-0835
```

The rebase **rewrote the lineage**, so the tree with evidence of working is off-lineage and the range was empty. **Before
any bisect on a rebased branch, check `git merge-base --is-ancestor <good> <bad>`** - otherwise the tool reports the
endpoint as the answer and looks successful. The harness is otherwise sound and stays: skip (125) on unbuildable, skip
(125) when the rotor never engaged, good (0) only on 3/3 clean, with the range, first-bad line and restored tip written to
`docs/bisect-log.txt`.

Also not usable: `c18d67f` (v1.1.40 base) is **not** an ancestor of `8d9979f` (v1.1.41 base) either - they are siblings.
A *tree diff* between them is still valid and is what the shared-state scan used.

### Step 1: the shared-state filter across the upstream delta

Searched both `c18d67f..8d9979f` (v1.1.40 -> v1.1.41, 2,892 added lines) and `cd52717..HEAD` (18,143 added lines) for
added lines touching state the rotor shares - ring indices, device counters, `__shared__`, single events, answer slots,
scratch bases, parity buffers.

**38 matches in the v1.1.41 delta, 83 in the full delta, and exactly one is on the rotor's path:**

```
MoeRequest* rq0 = (MoeRequest*) a.d.ring + ((*a.d.seq + 1) % kRingSize);   // ROUTE_FAST pre-write
```

Everything else is unrelated kernels (`exl3`, `ds4`, `prefill`), per-block `__shared__` declarations, or our own
instrumentation. `ROUTE_FAST=0` was already 0/3, so this line is suspicious in principle and insufficient on its own.

**Conclusion:** the fault is not a single upstream line of that shape either. With four of our reverts restored, the
family swept clean, and two upstream feature flags excluded by three boots each, what remains is the *interaction* of
v1.1.40/41 with the rotation - which needs the branch rebased onto the older base to split v1.1.41 off:

```
git rebase --onto c18d67f 8d9979f        # replay our commits onto v1.1.40's base, then: build, 3 boots
```

One build and three boots. Clean there means the fault arrived with v1.1.41 and that range **is** in lineage and
bisectable properly; failing means it arrived with v1.1.40 or earlier and the same move repeats onto that base.

Option 3 stands and is not a failure state: `cd52717` is a verified working `n=2` rotor (41-47 entries, ~24 tok/s,
coherent output), so Aaron's disk-spill rung starts there whenever he wants, independent of the replay.

## 2026-10-30 (late) — the reverse rebase: attempted, stopped at 6/42, aborted, parked

Setup done as directed before moving the tree:

- **rerere disabled entirely** - `rerere.enabled=false`, `rerere.autoUpdate=false`. Resolutions were recorded during the
  *forward* v1.1.40 -> v1.1.41 replay; going backwards they are not merely unverified, they are **backwards**, and would
  reinstate the newer side while we try to return to the older one.
- **backup tag**: `backup-pre-onto-20261030-1826` -> `ebd2a36`. Restore: `git reset --hard backup-pre-onto-20261030-1826`.

The operation itself: `git rebase --onto c18d67f 8d9979f`, which replays exactly **42 commits** (8d9979f is an ancestor of
HEAD; the range is our replayed work plus tonight's fixes).

**It stopped at 6/42**, on `56ced70` - *"rebase onto v1.1.41: repair the three hunks the union helper mangled"* - with one
conflicted file (`src/kernels/cuda/glm_fast.cu`). That commit exists **only because of v1.1.41**: its hunks describe
repairs to code the newer base introduced. Replaying it onto the older base is not a merge problem, it is a semantic
one - the commit has nothing to say there. So the backwards replay needs a decision per commit about which hunks are
version-specific, not the mechanical conflict resolution a bisect split wants.

`git rebase --abort` -> HEAD back at `ebd2a36`, tree clean, no `rebase-merge` left behind.

Also established: **there is no merge commit** in `c18d67f..HEAD`, so `git revert -m 1 <merge>` - the clean way to undo
the v1.1.41 delta in place - is not available: the v1.1.41 work arrived as a linear replay.

**Decision: parked.** The split costs a per-commit semantic replay over 42 commits and buys a halving we can live without
for now. Option 3 is not a failure state: **`cd52717` is a verified working `n=2` rotor** (41-47 entries, ~24 tok/s,
coherent output, 3/3 boots), so Aaron's disk-spill rung starts there whenever he wants it.

If someone does want the split later: re-enable rerere for the forward direction, do the replay as a deliberate task with
a decision recorded per commit, and verify each with the arm (build, three boots, rotor count) - not as a bisect step.

## Handover: the one category never examined (for whenever this resumes)

Every other shape has now been excluded by measurement: dropped lines (the audit found the four reverts and the family is
swept clean), upstream's new code (the shared-state filter is clean, and both rotor-adjacent features are 0/3 under their
own flags), memory errors (memcheck clean with the rotor engaged), and protocol faults (the wait-word dump, the both-stream
poll and the channel counters all cleared it).

What remains is **a resolution that compiles and is semantically wrong**. Six sites are where our code met upstream's and
something chose: the five hand-resolved conflicts, plus the union-mangled hunk repaired afterwards. Those resolutions were
checked against the compiler, not against what our side actually did at `cd52717` - and a wrong one adds or removes
nothing, so no count audit could see it.

The three repair commits name the files and the two poles of each:

```
935a22b  rebase: restore the BF16 fallbacks the v1.1.40 merge dropped, and make the q() error name its cause
73f08ab  rebase onto v1.1.40: port the q8_1->float scratch helpers the merge dropped
56ced70  rebase onto v1.1.41: repair the three hunks the union helper mangled (by text, not concatenation)
```

`56ced70` is the least verified of the six: its hunks were fixed until the compiler stopped complaining, which is exactly
the "does it build" test this category defeats. Start there.

For each site, read three versions side by side - `git show cd52717:<file>`, `git show c18d67f:<file>` (or `8d9979f`),
and what the replay produced - and ask only one question: **does our side still do what it did at `cd52717`?** The pattern
that has paid out four times tonight is per-row behaviour quietly reduced to per-call (`R(i)` -> `i`, per-row buffer ->
shared, gated allocation). Bounded at six sites, no boots, and it needs no rebuild to read.
