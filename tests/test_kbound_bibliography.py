"""Offline regression checks for the reviewed K-Bound bibliography records.

These tests check the maintained source against primary metadata verified on
2026-08-31. They do not reverify publication records, citation relevance, or
rendered layout, and they neither import the training stack nor touch data.
"""

from pathlib import Path
import re
import unittest


PAPER_ROOT = Path(__file__).resolve().parents[1] / "docs/research/kbound"
BIBLIOGRAPHY = PAPER_ROOT / "paper/references_kbound_expanded.tex"
CONTEXT_ARCHIVE = PAPER_ROOT / "paper/references_kbound_context_archive.tex"
ITEM = re.compile(r"^\\KBbibitem\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}", re.MULTILINE)
AUDITABLE_REFS = re.compile(r"\\ifdefined\\IncludeAuditableRefs\b.*?\\fi\b", re.DOTALL)
CITE = re.compile(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def without_comments(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def parse_entries(text):
    matches = list(ITEM.finditer(text))
    entries = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries[match.group(1)] = (
            match.group(2),
            match.group(3),
            " ".join(text[match.end():end].split()),
        )
    return [match.group(1) for match in matches], entries


class KBoundBibliographyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Strip TeX comments so provenance notes cannot satisfy content checks.
        text = without_comments(BIBLIOGRAPHY.read_text(encoding="utf-8"))
        cls.keys, cls.entries = parse_entries(text)
        _, cls.printed_entries = parse_entries(AUDITABLE_REFS.sub("", text))
        cls.archive_text = without_comments(CONTEXT_ARCHIVE.read_text(encoding="utf-8"))
        cls.archive_keys, cls.archive_entries = parse_entries(cls.archive_text)
        cls.cited_keys = set()
        # These are the citation-bearing prose inputs shared by both maintained
        # drivers. Generated numerical tables are not a bibliography authority.
        prose_inputs = [
            "kbound_submission_body.tex",
            "kbound_submission_supplement.tex",
            "kbound_abstract_core.tex",
            "paper/sections/theory_core_main.tex",
            "paper/sections/theory_certificate.tex",
        ]
        for relative_path in prose_inputs:
            prose = without_comments((PAPER_ROOT / relative_path).read_text(encoding="utf-8"))
            for match in CITE.finditer(prose):
                cls.cited_keys.update(key.strip() for key in match.group(1).split(","))

    def assert_in_order(self, text, items):
        position = -1
        for item in items:
            found = text.find(item, position + 1)
            self.assertGreater(found, position, f"Missing or out-of-order author: {item}")
            position = found

    def test_keys_are_unique_and_reviewed_records_are_preserved(self):
        self.assertTrue(self.keys)
        self.assertEqual(len(self.keys), len(set(self.keys)))
        self.assertEqual(len(self.archive_keys), len(set(self.archive_keys)))
        self.assertFalse(set(self.entries) & set(self.archive_entries))
        prior_uncited_keys = {
            "bendavid2012hardness", "corradaemmanuel2024", "croce2021robustbench",
            "deng2009imagenet", "garg2020labelshift", "gibbs2021adaptive",
            "gulrajani2021domainbed", "gupta2021toplabel", "hoang2024petta",
            "kalai2021abstain", "lee2024deyo", "liang2025ttasurvey", "lim2026reset",
            "lipton2018bbse", "madras2018learntodefer", "park2022pac",
            "paszke2019pytorch", "pedregosa2011sklearn", "sonoda2025lean", "tempora2026",
        }
        self.assertTrue(prior_uncited_keys.issubset(set(self.entries) | set(self.archive_entries)))

    def test_active_citations_resolve_without_uncited_printed_records(self):
        self.assertTrue(self.cited_keys)
        self.assertEqual(self.cited_keys, set(self.printed_entries))

    def test_context_archive_is_not_an_active_driver_input(self):
        for relative_path in [
            "kbound_submission.tex", "kbound_tmlr.tex", "kbound_submission_body.tex",
            "kbound_submission_supplement.tex",
        ]:
            text = without_comments((PAPER_ROOT / relative_path).read_text(encoding="utf-8"))
            self.assertNotIn("references_kbound_context_archive", text)
        self.assertFalse(self.cited_keys & set(self.archive_entries))

    def test_conditional_auditable_budget_records_remain_separate(self):
        self.assertEqual(set(self.entries) - set(self.printed_entries), {
            "bartl2022structure", "bartl2023uniformdkw", "boedihardjo2024sharp",
            "nietert2022sliced", "dunn2022hierarchical", "lee2023hierarchical",
            "fournier2015rate", "niles2019estimation", "weed2019sharp",
            "deng2017wasserstein", "wang2024kernelmsw",
        })

    def test_rxrx1_uses_correct_paper_and_author_year(self):
        # https://arxiv.org/abs/2301.05768 and the CVPRW 2023 CVMI record.
        author, year, body = self.entries["taylor2019rxrx1"]
        self.assertEqual((author, year), ("Sypetkowski et al.", "2023"))
        self.assert_in_order(body, [
            "M. Sypetkowski", "M. Rezanejad", "S. Saberian", "O. Kraus", "J. Urbanik",
            "J. Taylor", "B. Mabey", "M. Victors", "J. Yosinski", "A. Rezazadeh Sereshkeh",
            "I. Haque", "B. Earnshaw",
        ])
        self.assertIn("RxRx1: A dataset for evaluating experimental batch correction methods.", body)
        self.assertIn("(CVPR) Workshops", body)
        self.assertIn("4285--4294", body)
        self.assertIn("arXiv:2301.05768", body)
        self.assertNotIn("1907.04758", body)

    def test_cifar10_1_uses_2018_cifar_preprint(self):
        # https://arxiv.org/abs/1806.00451
        author, year, body = self.entries["recht2019cifar10"]
        self.assertEqual((author, year), ("Recht et al.", "2018"))
        self.assertIn("Do CIFAR-10 classifiers generalize to CIFAR-10?", body)
        self.assertIn("arXiv:1806.00451", body)
        self.assertNotIn("ICML", body)

    def test_sonoda_uses_five_authors_and_pinned_2025_title(self):
        # https://arxiv.org/abs/2503.19605v3
        author, year, body = self.entries["sonoda2025lean"]
        self.assertEqual((author, year), ("Sonoda et al.", "2025"))
        self.assert_in_order(body, ["S. Sonoda", "K. Kasaura", "Y. Mizuno", "K. Tsukamoto", "N. Onda"])
        self.assertIn("generalization error bound by Rademacher complexity.", body)
        self.assertIn("arXiv:2503.19605v3", body)
        self.assertNotIn("Dudley", body)

    def test_pacs_pages_match_ieee_doi_edition(self):
        # https://www.pure.ed.ac.uk/ws/portalfiles/portal/41072820/li2017dg.pdf
        _, year, body = self.entries["li2017pacs"]
        self.assertEqual(year, "2017")
        self.assertIn("5543--5551", body)
        self.assertIn("10.1109/ICCV.2017.591", body)
        self.assertNotIn("5542--5550", body)

    def test_cct_pages_match_springer_doi_edition(self):
        # https://authors.library.caltech.edu/records/m2211-qkc66
        _, year, body = self.entries["beery2018recognition"]
        self.assertEqual(year, "2018")
        self.assertIn("472--489", body)
        self.assertIn(r"10.1007/978-3-030-01270-0\_28", body)
        self.assertNotIn("456--473", body)

    def test_agreement_on_the_line_has_no_inserted_space(self):
        _, _, body = self.entries["baek2022aol"]
        self.assertIn("Agreement-on-the-line:", body)
        self.assertNotRegex(body, r"Agreement-on-\s+the-line")

    def test_distinct_bendavid_2010_works_have_distinct_citation_labels(self):
        # https://proceedings.mlr.press/v9/david10a.html
        theory_author, theory_year, theory_body = self.entries["bendavid2010theory"]
        author, year, body = self.entries["bendavid2010impossibility"]
        self.assertEqual((theory_author, theory_year), ("Ben-David et al.", "2010a"))
        self.assertEqual((author, year), ("Ben-David et al.", "2010b"))
        self.assertIn("151--175, 2010a", theory_body)
        self.assert_in_order(body, ["S. Ben-David", "T. Lu,", "T. Luu,", r"D. P\'al."])
        self.assertIn("Impossibility theorems for domain adaptation.", body)
        self.assertIn("PMLR 9:129--136, 2010b", body)
        self.assertIn("https://proceedings.mlr.press/v9/david10a.html", body)

    def test_drift_to_action_is_pinned_preprint_not_unverified_venue(self):
        # https://arxiv.org/abs/2603.08578v1
        author, year, body = self.entries["lamaakal2026drifttoaction"]
        self.assertEqual((author, year), ("Lamaakal et al.", "2026"))
        self.assert_in_order(body, ["I. Lamaakal", "C. Yahyati", "K. El Makkaoui", "I. Ouahbi", "Y. Maleh"])
        self.assertIn("Drift-to-Action Controllers: Budgeted interventions with online risk certificates.", body)
        self.assertIn("arXiv preprint", body)
        self.assertIn("arXiv:2603.08578v1", body)
        self.assertNotIn("ICLR", body)

    def test_tta_line_uses_published_neurips_record_and_all_five_authors(self):
        # https://proceedings.neurips.cc/paper_files/paper/2024/hash/d96fcc07d623a9eba68616629911143a-Abstract-Conference.html
        author, year, body = self.entries["kim2024ttaline"]
        self.assertEqual((author, year), ("Kim et al.", "2024"))
        self.assert_in_order(body, ["E. Kim", "M. Sun", "C. Baek", "A. Raghunathan", "J. Z. Kolter"])
        self.assertIn("Test-Time Adaptation Induces Stronger Accuracy and Agreement-on-the-Line.", body)
        self.assertIn("Advances in Neural Information Processing Systems", body)
        self.assertIn("37, 2024", body)
        self.assertIn("10.52202/079017-3820", body)
        self.assertNotIn("arXiv preprint", body)

    def test_selection_conditional_coverage_uses_2025_journal_metadata(self):
        # https://academic.oup.com/jrsssb/article/87/4/1239/8113856
        author, year, body = self.entries["jin2025selection"]
        self.assertEqual((author, year), ("Jin and Ren", "2025"))
        self.assert_in_order(body, ["Y. Jin", "Z. Ren"])
        self.assertIn("Confidence on the focal: conformal prediction with selection-conditional coverage.", body)
        self.assertIn("Journal of the Royal Statistical Society Series B: Statistical Methodology", body)
        self.assertIn("87(4):1239--1259, 2025", body)
        self.assertIn("10.1093/jrsssb/qkaf016", body)
        self.assertNotIn("1239--1257", body)

    def test_conditional_guarantees_is_distinct_from_adaptive_conformal(self):
        # https://academic.oup.com/jrsssb/article/87/4/1100/8058684
        author, year, body = self.entries["gibbs2025conditional"]
        self.assertEqual((author, year), ("Gibbs et al.", "2025"))
        self.assert_in_order(body, ["I. Gibbs", "J. J. Cherian", r"E. J. Cand\`es"])
        self.assertIn("Conformal prediction with conditional guarantees.", body)
        self.assertIn("Journal of the Royal Statistical Society Series B: Statistical Methodology", body)
        self.assertIn("87(4):1100--1126, 2025", body)
        self.assertIn("10.1093/jrsssb/qkaf008", body)
        self.assertEqual(self.entries["gibbs2021adaptive"][:2], ("Gibbs and Candes", "2021"))
        self.assertIn("Adaptive conformal inference under distribution shift.", self.entries["gibbs2021adaptive"][2])

    def test_partial_identification_decisions_uses_2026_corrected_proof(self):
        # https://academic.oup.com/restud/advance-article-abstract/doi/10.1093/restud/rdag017/8502914
        author, year, body = self.entries["christensen2026optimal"]
        self.assertEqual((author, year), ("Christensen et al.", "2026"))
        self.assert_in_order(body, ["T. Christensen", "H. R. Moon", "F. Schorfheide"])
        self.assertIn("Optimal Decision Rules When Payoffs are Partially Identified.", body)
        self.assertIn("The Review of Economic Studies", body)
        self.assertIn("rdag017, 2026", body)
        self.assertIn("Advance article, corrected proof.", body)
        self.assertIn("10.1093/restud/rdag017", body)
        self.assertNotIn("arXiv preprint", body)


if __name__ == "__main__":
    unittest.main()
