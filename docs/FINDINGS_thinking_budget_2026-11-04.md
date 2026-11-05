# FINDINGS — "garbage output" is reasoning delivered as the answer (2026-11-04)

## The symptom we chased

Agent traffic (Sphinx on CT 915, Whiteout on CT 912) produced outputs that read as garbage:
counting loops ("5 5 5"), meta-commentary ("I apologize for the confusion"), fragmented tags.
The loop guard sometimes saw it (`detected 13 / stopped 4`) and sometimes did not.

## What it actually is

**The model's reasoning text is delivered to the client in the `content` channel.**
`reasoning_content` is present and coherent; `content` continues the same deliberation.

Cause: the thinking budget reaches its limit while the model is still reasoning. The engine
injects the end-of-thinking token, sets `in_think = false`, and from that point every further
token is classified as the answer — including the rest of the model's thinking.

The model does not reliably stop thinking when handed the forced close. Usually it recovers
within a sentence and writes the answer anyway; sometimes it never recovers and the whole
answer is deliberation.

## The budget policy (upstream — `serve/server.py:3860-3884`)

- `thinking_budget` comes from the config (`thinking_budget`, 0 = none), else the model's
  default (GLM: 8192), else 32768 for other engines.
- **`max_tokens <= THINK_TINY_LIMIT` (1024)**: the request's own limit lowers the budget:
  `reserve = max(max_tokens//4, min(1024, max_tokens//2))`, `budget = max_tokens - reserve - 1`.
  Measured exactly: 500 -> 249, 12 -> 5, 200 -> 99, 700 -> 349.
- **This "tiny" path sets no `soft` close.** The `soft` (end the reasoning at the next
  sentence or line end, `#613`) is applied only for `max_tokens > 1024`. So small requests get
  a HARD cut mid-sentence, and that is where the leak is worst.

## Evidence (all on the v3q pack, temp 1.0, DIAG on)

| max_tokens | budget | shrink | result |
|---|---|---|---|
| 4000 | 8192 (never reached) | - | answers EMPTY: all tokens spent thinking |
| 700 | 349 | no | clean |
| 500 | 249 | no | mostly clean, brief reasoning leak at the start of some rows, recovered |
| 500 | 249 | yes | one row per round never recovered (round 1 row 1, round 2 row 0) |
| 200 | 99 | no | clean 3/3 |
| 12 | 5 | yes | clean (answer started immediately) |

Engine-side, across every run: **zero non-finite scores**, guard `detected 0`, every row ran
every layer, 1176 rotor steps in one run. Arithmetic, KV and the layer stack are exonerated.

## Corrections made during the hunt (do not repeat these)

- **"Position 1 is corrupted" was wrong.** It leaked on row 0 in one round. The earlier
  "10/10 position 1" was small-sample coincidence.
- **"The rotor causes it" was wrong.** The rotor was on in most leaky runs, but the leak is
  proportional to how often the budget cuts mid-thought, not to the row pipeline. The clean
  "rotor off" run was luck, not evidence.
- **"No shrink = clean" was wrong.** Those rows leaked too; they recovered within a sentence,
  and a tail-only check scored them clean.

## Instrumentation added (keep — cheap and diagnostic)

- `src/program/generate.cpp`: one stderr line per request when its reasoning closes, naming the
  arm that closed it (`the model emitted think_end` / `the budget` / `WRAP`) with `produced` and
  `budget`. Without it a premature close is indistinguishable from a model that chose to answer.
- `src/core/glm_rows.cu`: fail-closed invariant after the rotor's round loop - every row must
  have run every layer, else the step is refused instead of sampled short. It never fired (so
  the layer stack was never holing), but the absence of that check was a real gap.
- `WRAP` is shutdown-only (`serve/server.py:6678`, single caller) - ruled out.

## Mitigations

1. **Config**: set `thinking_budget` explicitly. `0` disables the budget (but a small
   `max_tokens` then spends everything on thinking -> empty answers). A generous value keeps
   the budget out of the way for normal requests.
2. **Client**: do not pair a high `reasoning_effort` with a small `max_tokens`. The reserve
   steals half the output for the answer, so the reasoning gets half of a small number.
3. **Code (the real fix)**: apply the `soft` sentence-end close to the `<= THINK_TINY_LIMIT`
   path as well, so reasoning ends at a boundary instead of mid-sentence; and consider
   detecting non-compliance (the model continuing to reason after the forced close) rather than
   silently labelling it the answer.

## Two routes to the same symptom (second found after the first fix)

1. **The engine forces the close at the thinking budget.** Everything the model thinks afterwards is
   labelled the answer. Now closes at a sentence/line end for short requests too (the grace is the
   same half of what `max_tokens` leaves after the budget, capped at `THINK_GRACE`) and logs a
   `WARNING ... hit its thinking budget`.
2. **The model writes the end-of-thinking tag inside its own reasoning.** GLM's template spells the
   end as TEXT, and `serve/frontend.py` (1128-1133) closes the reasoning on that text - the code
   itself warns that "a model that quotes the tag while it talks about it ... would have its
   reasoning cut there and spill into the answer". Observed at `max`: leaked `content` contained a
   literal `</think>` followed by a clean essay. **NOT fixed.** The engine logs no close in this
   case, so the budget fix cannot help: the cut is in the parser, not the budget.

## Verified after the fix (v3q, arm = overlap, DIAG off)

| cell | result |
|---|---|
| `low`, 500 tokens, 3 rows | clean 3/3 - closes are "the model emitted think_end" at produced=10 |
| `max`, 200 tokens, 3 rows | clean 3/3 (budget 99 + grace) |
| `max`, 500 tokens, 3 rows | row 1 still leaks in both rounds, via route 2 (no engine close logged) |

**Operational conclusion: `reasoning_effort: low` is safe (verified twice); `high`/`max` on this
model is not reliable for production traffic until route 2 is addressed.**

Options for route 2 (not implemented): close only on the end-of-thinking TOKEN where the vocabulary
has one (`OutputParser.token_only_end`), or hold a text end-tag until what follows looks like an
answer. Either is a design decision, not a bug fix.

## This is NOT unique to our engine or our pack — it is a known GLM-5.3-family defect

Found by searching after the fact (2026-11-04). Independent projects document both routes:

- **vLLM #54744 / PR #54825** — GLM-5.3's template has no thinking on/off switch; when extraction is
  disabled "the raw scratchpad plus a dangling `</think>` lands in `message.content`".
- **ollama #18082** — "GLM chat template emits orphaned closing think tag": reasoning inline in
  content with ONLY the closing tag, opening tag never emitted. Exactly our S1 specimen.
- **hermes-agent #96735** — the GLM shape is `reasoning + </think> + answer`, and the reasoning can
  be on BOTH channels (a separate `reasoning_content` AND an inline copy in content) — our O run.
- **pi #9554** — `zai/glm-5.3-flash` chain-of-thought in `content` with no `reasoning_content`,
  31/31 turns on an affected session, clustering on 216K-376K-token cross-model histories. That is
  Sphinx's shape (376k) and it explains why Sphinx leaked at `low` while short-context benches did not.
- **NEAR AI docs** — `enable_thinking: false` does not speed GLM-5.3 Flash up; the reasoning simply
  moves into `content`. Their guidance: small `thinking_token_budget` or `reasoning_effort: low`.

So the practical rule others have arrived at matches ours: **keep effort low, or budget thinking
explicitly** — and any consumer that needs strict channel separation must handle the orphan tag.

## Fix attempted for route 2 (gated: `MAYA_GLM_ORPHAN_THINK=1`)

`serve/frontend.py`: `OutputParser(orphan_think=...)`. In content state, when NOTHING has been emitted
as content yet and a close tag appears with a prefix, that prefix is reasoning: it is reported as a
reasoning event, or DROPPED when the reasoning already went out on its own channel (the "both
channels" shape). `_feed` is a thin wrapper that records what actually went out, so the rule keys on
emissions rather than on parser intent.

Gated rather than default because a legitimate answer that quotes a close tag early would be split.
Read from the config's env, not the shell: the server does NOT inherit the config's `env` the way
the engine child does, and the config may carry either spelling (`STRATA_*` is rewritten to `MAYA_*`).

A SHORT preamble (<= 2048 chars) is HELD while the gate is on, because a tag can arrive split
across deltas and text already emitted as content cannot be taken back - the same limit vLLM
states for its streaming path.  `OutputParser.finish()` flushes anything still held, so the cost
is streaming latency, not lost output.  Pinned by `serve/test_orphan_think.py` (5 cases: orphan
split, duplicate dropped, gate off unchanged, normal answer untouched, tag split across deltas).

## Final verification of the orphan-tag fix (v3q, gate on, arm = overlap)

| cell | before (gate off) | after (gate on) |
|---|---|---|
| `max` 500, 3 rows, no shrink (2 rounds) | row 1 leaked both rounds | **5 of 6 clean**; one row still delivered planning notes as content (no close tag at all - nothing for a tag rule to find) |
| `max` 500, 3 rows, with shrink, both channels | row 1 leaked both rounds | **clean 4 of 4**: `reasoning_content` carries the scratchpad (1083-1210 chars), `content` opens with the title |
| `low` 500, 3 rows (control) | clean | clean, unchanged |

Correction to the expectation: the fix reclassifies the TAG shape, which is the documented and
dominant one, and V2 shows it doing exactly that - reasoning preserved, not deleted. It cannot
help the no-tag variant (the model's notes arrive with no marker at all), which is the same limit
vLLM states for its streaming path. Closing that one needs the ENGINE to tag the channel, not the
parser to infer it.

## Open

- Whether the same leak explains Whiteout's repeated-token runaways, or whether that is a
  separate degeneration. The guard only fires on repetition, and it stayed silent on every
  leak reproduced here.
- The correct long-term policy for `thinking_budget` vs `max_tokens`; today's rule (half the
  output, hard cut) is what produces the symptom.
- **The no-tag variant**: the model's planning notes delivered as content with NO close tag
  anywhere (seen once in 6 rows). No parser rule can catch that; the ENGINE knows `in_think` per
  request and should tag the channel itself when reasoning was requested.

## Route 2: close on the TOKEN, not a text search (2026-11-05) - and what it does NOT fix

The end of thinking is a TOKEN in our stack and always was: the template opens the block in the
generation prompt, the engine is handed `think_end` as an ID and force_next()s it, and the server
already compares ids (`end_think`).  `token_only_end` was enabled for DeepSeek only, on the belief
that GLM "spells the end as text" - and that belief is what let a `</think>` the model writes INSIDE
its reasoning close the channel early.  It is now the DEFAULT for GLM-5.x (`template.glm_effort`,
verified True on the pack's own `tokenizer/chat_template.jinja`), with `MAYA_GLM_THINK_TOKEN_ONLY=0`
to restore the text search.

**What it fixes:** the tag cannot be mistaken for the end of thinking any more, because the text
search never runs.  Measured: `low`/500 splits at the token (produced=7/13), content 2114-2448 chars,
clean every time.

**What it does NOT fix - CORRECTION to an earlier 4/4 claim in this file.**  A single 4-row run
looked clean and I recorded it as solved; a second run at the same settings leaked on 2 of 4 rows.
The mechanism is not parsing: the model emits the close token, the channel splits, and then it
CONTINUES PLANNING in the answer channel ("Actually, let me write about some famous historic
lighthouse keepers... hmm, I meant:").  That is the model, after a close it did not want, and no
parser can classify it.  The lever there is the budget/effort policy, not the parser.

Honest state: token-only strictly removes one failure class (text false-close) and leaves the other
(model continues after a legitimate close).  `low` effort is clean; `high`/`max` under a cut budget
is still unreliable.

