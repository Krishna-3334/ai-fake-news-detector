"""Offline unit tests for the verdict rules and evidence validation in factcheck.py (no network, no API key).

Run:  .venv/bin/python -m unittest discover -s tests -v
"""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import factcheck as fc

RBI = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=1"
NEWS = "https://www.reuters.com/world/india/story"
OTHER = "https://someblog.example/post"
PAGES = {
    RBI: {"title": "RBI notifications", "text": "Master Direction on Prepaid Payment Instruments issued on 27 August 2021 by the Reserve Bank"},
    NEWS: {"title": "Reuters story", "text": "The Reserve Bank of India kept its key repo rate unchanged on Wednesday, the central bank said"},
    OTHER: {"title": "Blog", "text": "Some unrelated text about wallets"},
}


def claim(status, sources=(), central=True, **kw):
    return {"claim": "c", "status": status, "central": central, "reason": "r", "sources": list(sources), **kw}


def src(url, excerpt="", published=None):
    return {"url": url, "publisher": "p", "excerpt": excerpt, "published": published}


class Validate(unittest.TestCase):
    def test_unretrieved_source_dropped_and_claim_downgraded(self):
        out = fc.validate({"claims": [claim("supported", [src("https://invented.example/x", "anything at all here")])]}, PAGES, "")
        self.assertEqual(out[0]["sources"], [])
        self.assertEqual(out[0]["status"], "unresolved")
        self.assertIn("treated as unresolved", out[0]["note"])

    def test_verbatim_excerpt_verified_paraphrase_not(self):
        out = fc.validate({"claims": [claim("supported", [
            src(RBI, "Master Direction on Prepaid Payment Instruments issued on 27 August 2021"),
            src(NEWS, "The RBI raised rates sharply on Wednesday according to officials"),
        ])]}, PAGES, "")
        s = out[0]["sources"]
        self.assertTrue(s[0]["excerpt_verified"]); self.assertIsNotNone(s[0]["excerpt"])
        self.assertFalse(s[1]["excerpt_verified"]); self.assertIsNone(s[1]["excerpt"])
        self.assertEqual(s[0]["tier"], "primary"); self.assertEqual(s[1]["tier"], "news")

    def test_published_date_kept_only_if_on_page(self):
        out = fc.validate({"claims": [claim("supported", [src(RBI, published="27 August 2021"), src(NEWS, published="3 March 2026")])]}, PAGES, "")
        self.assertEqual(out[0]["sources"][0]["published"], "27 August 2021")
        self.assertIsNone(out[0]["sources"][1]["published"])

    def test_conflicting_needs_two_domains(self):
        out = fc.validate({"claims": [claim("conflicting", [src(NEWS)])]}, PAGES, "")
        self.assertEqual(out[0]["status"], "unresolved")

    def test_unknown_status_becomes_unresolved(self):
        out = fc.validate({"claims": [claim("true", [src(NEWS)])]}, PAGES, "")
        self.assertEqual(out[0]["status"], "unresolved")

    def test_no_central_claims_means_all_central(self):
        out = fc.validate({"claims": [claim("unresolved", central=False), claim("unresolved", central=False)]}, PAGES, "")
        self.assertTrue(all(c["central"] for c in out))

    def test_official_check_only_believed_if_official_page_retrieved(self):
        no_official = {NEWS: PAGES[NEWS]}
        self.assertFalse(fc.validate({"claims": [claim("unresolved", checked_official_source=True)]}, no_official, "")[0]["checked_official_source"])
        self.assertTrue(fc.validate({"claims": [claim("unresolved", checked_official_source=True)]}, PAGES, "")[0]["checked_official_source"])


def V(*claims, **data):
    """Validated claims with real retrieved sources, for aggregate tests."""
    return fc.validate({"claims": list(claims)}, PAGES, ""), data


class Aggregate(unittest.TestCase):
    def cat(self, *claims, **data):
        cl, d = V(*claims, **data)
        return fc.aggregate(cl, d)[0]

    def test_contradicted_central_claim_is_fake(self):
        self.assertEqual(self.cat(claim("supported", [src(NEWS)]), claim("contradicted", [src(RBI)])), "likely_fake")

    def test_real_story_with_one_fabricated_detail_is_misleading(self):
        cl, d = V(claim("supported", [src(NEWS)]), claim("contradicted", [src(RBI)], central=False))
        cat, rule = fc.aggregate(cl, d)
        self.assertEqual(cat, "likely_fake"); self.assertIn("misleading", rule)

    def test_all_central_supported_is_real(self):
        self.assertEqual(self.cat(claim("supported", [src(NEWS)]), claim("supported", [src(RBI)])), "likely_real")

    def test_partly_supported_is_unverified(self):
        self.assertEqual(self.cat(claim("supported", [src(NEWS)]), claim("unresolved")), "unverified")

    def test_absence_of_evidence_is_unverified_not_fake(self):
        # the RBI-style fabricated circular: nothing found anywhere -> unverified, never "real"
        self.assertEqual(self.cat(claim("unresolved"), claim("unresolved")), "unverified")

    def test_claim_supported_without_retrieved_source_does_not_make_it_real(self):
        self.assertEqual(self.cat(claim("supported", [src("https://invented.example/x")])), "unverified")

    def test_satire_is_fake_or_misleading(self):
        self.assertEqual(self.cat(claim("unresolved"), article_type="satire"), "likely_fake")

    def test_old_news_as_new_is_misleading(self):
        self.assertEqual(self.cat(claim("supported", [src(NEWS)]), timeliness={"old_news_presented_as_new": True}), "likely_fake")

    def test_misleading_headline_needs_evidence(self):
        self.assertEqual(self.cat(claim("supported", [src(NEWS)]), headline_check={"misleading": True}), "likely_fake")
        self.assertEqual(self.cat(claim("unresolved"), headline_check={"misleading": True}), "unverified")

    def test_conflicting_sources_is_unverified(self):
        self.assertEqual(self.cat(claim("conflicting", [src(NEWS), src(RBI)])), "unverified")


class Strength(unittest.TestCase):
    def test_levels(self):
        cl, _ = V(claim("supported", [src(RBI, "Master Direction on Prepaid Payment Instruments issued on 27 August 2021")]))
        self.assertEqual(fc.evidence_strength(cl), "strong")
        cl, _ = V(claim("supported", [src(NEWS)]))
        self.assertEqual(fc.evidence_strength(cl), "weak")
        cl, _ = V(claim("unresolved"))
        self.assertEqual(fc.evidence_strength(cl), "none")


class Parsing(unittest.TestCase):
    def test_parse_answer_skips_snippet_braces(self):
        raw = 'snippet {"x": 1} more {"claims": [{"claim": "a"}], "summary": "s"}'
        self.assertEqual(fc.parse_answer(raw)["summary"], "s")

    def test_retrieved_pages_collects_all_tool_outputs(self):
        msg = NS(executed_tools=[NS(output="opened https://a.example/p 【1】", browser_results=[NS(url="https://b.example", title="B", content="bee")],
                                    search_results=NS(results=[NS(url="https://c.example", title="C", content="sea")]))])
        pages, corpus = fc.retrieved_pages(msg)
        self.assertEqual(set(pages), {"https://a.example/p", "https://b.example", "https://c.example"})
        self.assertIn("opened", corpus)


class NoKey(unittest.TestCase):
    def test_missing_key_reports_unavailable(self):
        orig = fc.load_env
        fc.load_env = lambda: {"GROQ_API_KEY": "paste-your-groq-key-here"}
        try:
            out = fc.check("t", "some article text here for testing", {"label": "real", "score": -1.0})
            self.assertEqual(out["code"], "no_key")
        finally:
            fc.load_env = orig


if __name__ == "__main__":
    unittest.main()
