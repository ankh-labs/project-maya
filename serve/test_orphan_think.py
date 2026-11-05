"""The orphaned-close-tag shape (GLM-5.3 family).

The model can emit its scratchpad INLINE with only the closing tag and no opening tag.  A parser
that starts in content then hands the reasoning to the client as the ANSWER.  These pin the shapes
we measured, plus the guard that a normal answer is untouched.

Run: python -m unittest serve.test_orphan_think   (CI: unittest discover -s serve)
"""
import unittest

from serve.frontend import OutputParser, THINK_END


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


class OrphanThink(unittest.TestCase):
    def test_orphan_tag_splits_when_reasoning_never_went_out(self):
        evs = run(["The user wants an essay. Let me draft it.", THINK_END, "# A Title\n\nBody."],
                  thinking=False, orphan_think=True)
        self.assertTrue(joined(evs, "reasoning").startswith("The user wants an essay"), evs)
        self.assertTrue(joined(evs, "content").startswith("# A Title"), evs)

    def test_orphan_tag_is_dropped_when_reasoning_already_went_out(self):
        # the "both channels" shape: the same scratchpad went out as reasoning AND inline in content
        p = OutputParser(thinking=True, orphan_think=True)
        evs = list(p.feed("The user wants an essay."))
        evs += list(p.close_reasoning())          # reasoning delivered on its own channel
        evs += list(p.feed(THINK_END + "# A Title\n\nBody."))
        evs += list(p.close_reasoning())
        evs += list(p.finish())
        evs = [(e.kind, e.text) for e in evs if e.text]
        self.assertEqual(joined(evs, "reasoning"), "The user wants an essay.", evs)   # not duplicated
        self.assertTrue(joined(evs, "content").startswith("# A Title"), evs)

    def test_gate_off_is_unchanged(self):
        evs = run(["scratchpad", THINK_END, "answer"], thinking=False, orphan_think=False)
        self.assertEqual(joined(evs, "reasoning"), "", evs)
        self.assertEqual(joined(evs, "content"), "scratchpad" + THINK_END + "answer", evs)

    def test_normal_answer_is_untouched(self):
        evs = run(["# The Flame\n\nFor centuries the keeper..."], thinking=False, orphan_think=True)
        self.assertTrue(joined(evs, "content").startswith("# The Flame"), evs)

    def test_orphan_split_across_deltas(self):
        # the tag may arrive split, which is why the parser holds a short preamble
        evs = run(["thinking text" + THINK_END[:3], THINK_END[3:], "# Title\nBody"],
                  thinking=False, orphan_think=True)
        self.assertTrue(joined(evs, "reasoning").startswith("thinking text"), evs)
        self.assertTrue(joined(evs, "content").startswith("# Title"), evs)

    def test_token_only_ignores_a_tag_inside_the_reasoning(self):
        # with token_only_end the text search never runs: a tag written mid-reasoning is inert
        p = OutputParser(thinking=True, token_only_end=True)
        evs = list(p.feed("scratchpad mentioning " + THINK_END + " while thinking"))
        evs += list(p.close_reasoning())
        evs += list(p.feed("# A Title\nBody"))
        evs += list(p.finish())
        evs = [(e.kind, e.text) for e in evs if e.text]
        self.assertIn("scratchpad mentioning", joined(evs, "reasoning"), evs)
        self.assertTrue(joined(evs, "content").startswith("# A Title"), evs)

    def test_a_tool_call_is_never_held_by_the_orphan_rule(self):
        # REGRESSION: the hold waited for a close tag that never came, and finish() flushed the raw
        # <tool_call> markup as the ANSWER - a real session ran no tool at all (tools=[] every turn).
        tools = [{"type": "function",
                  "function": {"name": "execute",
                               "parameters": {"type": "object",
                                              "properties": {"code": {"type": "string"}}}}}]
        body = "<tool_call>execute<arg_key>code</arg_key><arg_value>x</arg_value></tool_call>"
        p = OutputParser(thinking=False, tools=tools, orphan_think=True)
        evs = list(p.feed(body[:20])) + list(p.feed(body[20:])) + list(p.finish())
        kinds = [e.kind for e in evs]
        self.assertTrue(any(k in ("tool_start", "tool_call") for k in kinds), kinds)
        for e in evs:                       # the markup must never go out as the answer
            if e.kind == "content" and e.text:
                self.assertNotIn("<tool_call>", e.text, evs)


if __name__ == "__main__":
    unittest.main()
