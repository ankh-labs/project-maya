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
