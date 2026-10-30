# Maya engine — live state

> **Scope:** this file is the live state of the **Maya** engine, and the source of truth for it. The **llama.cpp** stack
> at `G:\AI\llama-server\` is the *other* engine and is **not current**: its docs there hold the llama.cpp history,
> the older measurements and the pointer back here.

## 2026-10-30 — v1.1.41 deployed; the rotation is serialised off on this pack

**Deployed:** v1.1.41, tag `deployed/v1.1.41-20261030-1210` @ `56ced70`, branch and tag pushed to Sia. Battery 7/7 PASS.
Fused/token path verified; `docs/TESTING.md` has the run.

**THE CURRENT LIMITATION, in one paragraph.** The de-fused rotation does not run at `n > 1` on this pack.
`blk.0.attn_output` lives in the pack's `dense.bin` (BF16), so the per-row fallback — restored from `6ae276e`'s
deletion — decodes **every batch one row at a time**; the verified build measured **0 rotor entries**. Layer 0 is
always in the stack, so this is not a corner case: it fires for any batch. The feature this branch exists to deliver is
therefore not running in its concurrent form on this pack, and the +4 % paired win measured earlier is not what is
deployed now.

**Options, re-ranked (this is the key correction):**
1. **REPACK so the nine BF16 tensors go native** — the *only* option that removes the fallback's trigger and lets the
   rotor run concurrently. Previously ranked third; it is now first. `iq_pack.py`'s fail-closed check tests the NAME,
   not the stored type, so a `GLM5NEXT_NATIVE_OK` tensor whose source GGUF row is BF16 is packed to the dense arena
   (blk.0 / blk.3 / blk.10 are exactly that).
2. Teach `kda_rec` to emit float activations (the real fix: a new activation buffer per layer, not local).
3. Leave the fallback and accept no rotation at `n > 1`.

**Loader dependency introduced by v1.1.41.** Upstream `50a3313` moved BF16-kept tensors out of `ws_map_` into `w16_`,
so `q()` cannot resolve them. The fast path needs `q_or_b16` at six sites (`attn_q/k/v`, `attn_q_b`, `attn_output`,
`ffn_down_shexp`) to read the BF16 copies. This is a deliberate response to that loader change, not a restoration.

**Open hazard.** `mv_xf_scratch()` is one process-wide allocation while `mv()` is on the concurrent path; unreachable
today only because the fallback serialises the rows. Fix is per-stream or per-slot scratch, not a bigger buffer.
See `docs/HYPOTHESIS_shared_mv_scratch.md` (it also carries the named hypothesis for the one unexplained 182 s
`state=reading` stall).

**Throughput of v1.1.41 is UNMEASURED.** The battery passing 7/7 is a gate result, not a measurement: there is no
paired v1.1.40 run of the same battery in the same boot, and its mean (6.35 tok/s) mixes prefill-dominated cases, so it
does not compare against the Oct 27 deployed-build figure of 23.8 tok/s. Do not quote 6.35 as a throughput number.

## Standing posture (2026-10-30, corrected)

**This is a homelab lab workstation, not production.** The box serves Aaron's own agents. Consequences for how findings are
written here:

- Say **the deployed build** or **the live box**, never "production".
- There is **no release register**: no "not ready", no "ready for testing", no "safe to cut over". There is nothing to be
  ready *for*. A wedged rotor is an experiment in progress, not an outage.
- **A crash is a data point** - reported flatly, then the next arm runs.
- The rotation stays default-on while it is being worked on. Restoring the fused build is a thing to *do* if the box is
  wanted for something else, not a risk to escalate.

Unchanged, and worth keeping: three boots per arm, verify from the log that the arm actually exercised what it claims
(the rotor check), and no claim from a single run.
