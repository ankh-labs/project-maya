# Integration ledger — what is in the reference tree and what is still owed

Built 2026-10-31 by **patch-id** (`git cherry -v cd52717 <branch>`), not by SHA, because
most of this work was cherry-picked across rebases and so carries different SHAs with
identical content. **A SHA-based "not merged" list over-reports badly here**: of the 11
commits on `pick/560-iq-mmvq` that are not reachable from `cd52717`, **8 are already in it
by content.**

Re-run after any rebase or merge:

```
for b in $(git branch --format='%(refname:short)'); do echo "--- $b"; git cherry -v cd52717 $b; done
```

`-` = equivalent patch already present. `+` = genuinely missing.

## Already integrated into `cd52717` (nothing owed)

`feat/prefill-align`, `fix/mvnt-bf16-xf`, `fix/rows-bf16-serial`, `homelab/v3q-support`,
`integrate/picks`, `pick/487-iq-mmq` — all `IN` by ancestry, and `fix/rows-bf16-serial`
confirmed `-` by patch-id.

Also `IN` by content from the `pick/560` series: the v3q pack fast-path fixes
(`a2b6d47`), **`slot_load`'s 8-stream read (`cb3ddc5`)**, prompt-cache stage 2A
(`e698f41`), q4_0 stages and their four fixes (`9993899`, `fb54d78`, `d89bfc9`,
`025e60a`), and the `STRATA_GLM_EXPERT_USED` dual-name acceptance (`d361aec`).

> Note `cb3ddc5`: `slot_load`'s parallel read **is** in `cd52717`. What thinned it was the
> v1.1.41 replay (4 markers -> 2), which is why it is a replay loss and not an
> unintegrated branch. Do not re-pick it; restore it.

## Owed — re-integrate onto the v1.1.42 tree

| # | Commit | What | Note |
|---|---|---|---|
| 1 | `e30cac5` | `iq_mmvq`: read each weight block once per batch (block index outermost) — **our PR 560** | a measured decode win; highest value of the four |
| 2 | `bdda0b6` | GLM prompt GEMMs: tuning preset for the shapes it asks for + a warning on a table miss — **our PR 582** | prefill is ~67% of a real agent turn, so this is not cosmetic |
| 3 | `ef534ab` | prompt cache: identity, content-hashed pack, state signature, async save, q4_0 stage A | **see caution** |
| 4 | `18bc211` | prompt cache: slot identity by meaning, checksummed state, marker-free slot at the last turn boundary | **see caution** |

**Caution on 3 and 4.** Their neighbours in the same series read `-` (already present),
so the content most likely landed in `cd52717` in reworked or squashed form and only these
two hunks differ. **Diff them against the tree before cherry-picking** — a blind pick will
either conflict or double-apply. Treat them as a content question, not a pick.

## Deliberately excluded — do NOT re-integrate

| Commit | Why |
|---|---|
| `1f71d1d` `pick/118-fetch-overlap` | **feature DEFECTIVE**, never enabled |
| `add5e86` `fix/step-mutex` | WIP; **does not fix** the KDA output failure |
| `34eebde` `fix/rows-mvnt-fallback` | **NOT SHIPPABLE** — computes one column, wrong output for n>1 |

These three stay as branches for the record. The reason each is excluded is in its own
subject line; keep that true of anything added here.

## Replay losses — restore, do not re-pick

Distinct from the list above: these were **in** `cd52717` and were dropped or thinned by
the v1.1.41 replay. Five restored 2026-10-30/31; two outstanding.

| Item | State |
|---|---|
| four per-row hunks, and the mover's `std::min<size_t>(8, F->stage.size())` clamp | **restored** |
| `slot_load` parallel read (markers 4 -> 2) | **outstanding** |
| `kv_norm` H512 butterfly (replaced by upstream's own kernel) | **outstanding — acknowledged divergence**, arithmetic only, no per-row state |

`tools/patch_invariants.txt` carries all of these; the checker is the gate.
