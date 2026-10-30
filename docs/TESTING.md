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
