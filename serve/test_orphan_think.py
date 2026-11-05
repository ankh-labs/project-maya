"""The orphaned-close-tag shape (GLM-5.3 family).

The model can emit its scratchpad INLINE with only the closing tag and no opening tag.  A parser
that starts in content then hands the reasoning to the client as the ANSWER.  These tests pin the
three shapes we measured, plus the guard that a normal answer is untouched.

Run: python serve/test_orphan_think.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.frontend import OutputParser, THINK_END   # noqa: E402


def run(deltas, **kw):
    p = OutputParser(**kw)
    evs = []
    for d in deltas:
        evs += p.feed(d)
    evs += p.close_reasoning()
    evs += p.finish()          # end of generation: flushes whatever the orphan hold is holding
    return [(e.kind, e.text) for e in evs if e.text]


def joined(evs, kind):
    return "".join(t for k, t in evs if k == kind)


def test_orphan_tag_splits_when_reasoning_never_went_out():
    evs = run(["The user wants an essay. Let me draft it.", THINK_END, "# A Title\n\nBody."],
              thinking=False, orphan_think=True)
    assert joined(evs, "reasoning").startswith("The user wants an essay"), evs
    assert joined(evs, "content").startswith("# A Title"), evs


def test_orphan_tag_is_dropped_when_reasoning_already_went_out():
    # the "both channels" shape: the same scratchpad went out as reasoning AND inline in content
    p = OutputParser(thinking=True, orphan_think=True)
    evs = list(p.feed("The user wants an essay."))
    evs += list(p.close_reasoning())          # reasoning delivered on its own channel
    evs += list(p.feed(THINK_END + "# A Title\n\nBody."))
    evs += list(p.close_reasoning())
    evs += list(p.finish())
    evs = [(e.kind, e.text) for e in evs if e.text]
    assert joined(evs, "reasoning") == "The user wants an essay.", evs   # not duplicated
    assert joined(evs, "content").startswith("# A Title"), evs


def test_gate_off_is_unchanged():
    evs = run(["scratchpad", THINK_END, "answer"], thinking=False, orphan_think=False)
    assert joined(evs, "reasoning") == "", evs
    assert joined(evs, "content") == "scratchpad" + THINK_END + "answer", evs


def test_normal_answer_is_untouched():
    evs = run(["# The Flame\n\nFor centuries the keeper..."], thinking=False, orphan_think=True)
    assert joined(evs, "content").startswith("# The Flame"), evs


def test_orphan_split_across_deltas():
    # the tag may arrive split, which is why the parser buffers
    evs = run(["thinking text" + THINK_END[:3], THINK_END[3:], "# Title\nBody"],
              thinking=False, orphan_think=True)
    assert joined(evs, "reasoning").startswith("thinking text"), evs
    assert joined(evs, "content").startswith("# Title"), evs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} passed")
