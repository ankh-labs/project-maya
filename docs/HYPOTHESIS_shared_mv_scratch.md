# THE SHARED-SCRATCH HYPOTHESIS (unexplained 182 s "reading" hang, 30 Oct)
# Not filed as unexplained any more: it has a mechanism and a prediction.
#
# Mechanism: mv_xf_scratch() is ONE process-wide cudaMalloc.  mv() is now on the concurrent path (SUBMIT/TRANSFER/RETIRE
# on separate streams, --parallel 2), so two concurrent mv() calls receive the same base pointer and the second launch
# overwrites activations the first may still be reading.  mv_nt got away with it because it was called from one place
# under one step; mv() is not.
#
# Why the verified boot did not show it: blk.0.attn_output is BF16, so the restored per-row fallback decoded the pair
# one row at a time and the measured boot had 0 rotor entries - the race was serialised away, not absent.
# Why the hanging boot could have: active=2 with no rotor-entry measurement, i.e. the rotation on by default since
# cd52717 and mv() possibly concurrent with it.
#
# PREDICTION: the stall returns as soon as the rotor and a BF16 output job overlap (e.g. a pack whose attn_output is
# quantised, so the fallback does not trigger, or the fallback relaxed).  If a future stall looks like the 182 s
# "state=reading" hang, check first whether mv() ran concurrently; the fix is per-stream or per-slot scratch, NOT a
# bigger buffer (a bigger buffer makes the race rarer, which is worse than a hang).
