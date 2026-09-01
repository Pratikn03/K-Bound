"""Binding/mutation guards, not automatic English-to-Lean semantic equivalence."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))
import verify_track as verification


class StatementBindingTests(unittest.TestCase):
    def ledger(self):
        return json.loads((TRACK / "theorem_ledger.json").read_text())

    def expect_changed_claim_rejected(self, field, replacement):
        original = self.ledger()["claims"][0]
        changed = copy.deepcopy(original)
        changed[field] = replacement
        before = {"T1": verification.digest(verification.payload_bytes(original))}
        after = {"T1": verification.digest(verification.payload_bytes(changed))}
        with self.assertRaises(verification.VerificationFailure):
            verification.check_snapshot(after, before)

    def test_quantifier_change_detected(self):
        self.expect_changed_claim_rejected("statement", "There exists one candidate for which the identity holds.")

    def test_inequality_change_detected(self):
        self.expect_changed_claim_rejected("statement", "Positive benefit implies frozen cost <= candidate cost.")

    def test_constant_change_detected(self):
        self.expect_changed_claim_rejected("assumptions", ["All policy costs in [0,2]"])

    def test_orientation_change_detected(self):
        self.expect_changed_claim_rejected("statement", "Benefit is candidate cost minus frozen cost.")

    def test_scope_change_detected(self):
        self.expect_changed_claim_rejected("scope", "A statistical population guarantee from observed batch accuracy.")

    def test_assumption_removal_detected(self):
        self.expect_changed_claim_rejected("assumptions", [])

    def test_unsupported_promotion_rejected(self):
        ledger = self.ledger()
        ledger["claims"][0]["promotion_status"] = "PROMOTED"
        names = {name for claim in ledger["claims"] for name in claim["lean_declarations"]}
        with self.assertRaises(verification.VerificationFailure):
            verification.check_claim_ledger(ledger, names)

    def test_uncompiled_declaration_rejected(self):
        ledger = self.ledger()
        names = {name for claim in ledger["claims"] for name in claim["lean_declarations"]}
        ledger["claims"][0]["lean_declarations"].append("MulticlassVectorCapacity.NotProved")
        with self.assertRaises(verification.VerificationFailure):
            verification.check_claim_ledger(ledger, names)

    def test_verified_identity_cannot_drop_its_declarations(self):
        ledger = self.ledger()
        names = {name for claim in ledger["claims"] for name in claim["lean_declarations"]}
        ledger["claims"][0]["lean_declarations"] = []
        with self.assertRaises(verification.VerificationFailure):
            verification.check_claim_ledger(ledger, names)

    def test_verified_identity_cannot_use_an_unrelated_proof(self):
        ledger = self.ledger()
        names = {name for claim in ledger["claims"] for name in claim["lean_declarations"]}
        ledger["claims"][0]["lean_declarations"] = ["MulticlassVectorCapacity.ThreeClass.no_negative_world"]
        with self.assertRaises(verification.VerificationFailure):
            verification.check_claim_ledger(ledger, names)

    def test_unproved_rate_cannot_gain_verified_status(self):
        ledger = self.ledger()
        names = {name for claim in ledger["claims"] for name in claim["lean_declarations"]}
        next(claim for claim in ledger["claims"] if claim["id"] == "T6")["mathematical_status"] = "LEAN_VERIFIED_COMPLETE_PASSIVE_THEOREM"
        with self.assertRaises(verification.VerificationFailure):
            verification.check_claim_ledger(ledger, names)

    def expect_inventory_mutation_rejected(self, mutate):
        ledger = self.ledger()
        path = (TRACK / ledger["formal_evidence"]["inventory"]).resolve()
        original_read = verification.resident_bytes
        inventory = json.loads(original_read(path))
        mutate(inventory)
        data = verification.payload_bytes(inventory)
        with patch.object(verification, "resident_bytes", side_effect=lambda p: data if p.resolve() == path else original_read(p)):
            with self.assertRaises(verification.VerificationFailure):
                verification.check_formal_inventory(ledger)

    def test_source_file_cannot_impersonate_compiled_object(self):
        def mutate(inventory):
            for theorem in inventory["theorems"]:
                theorem["olean_path"] = "MulticlassVectorCapacity.lean"
                theorem["olean_sha256"] = verification.digest((TRACK / "formal/MulticlassVectorCapacity.lean").read_bytes())
        self.expect_inventory_mutation_rejected(mutate)

    def test_compiled_module_must_match_declaring_source(self):
        self.expect_inventory_mutation_rejected(lambda inventory: inventory["theorems"][0].__setitem__("module", "MulticlassVectorCapacity.DoesNotExist"))

    def test_required_formal_source_bindings_cannot_be_dropped(self):
        for name in ("MulticlassVectorCapacity/Audit.lean", "MulticlassVectorCapacity/Regression.lean", "MulticlassVectorCapacity.lean", "export_inventory.py", "lean-toolchain", "lake-manifest.json", "lakefile.lean", "tests/test_export_inventory.py"):
            with self.subTest(source=name):
                self.expect_inventory_mutation_rejected(lambda inventory: inventory["source_hashes"].pop(name))

    def test_theorem_logs_must_match_verified_top_level_logs(self):
        for key in ("build_log", "axiom_log"):
            with self.subTest(log=key):
                self.expect_inventory_mutation_rejected(lambda inventory: inventory["theorems"][0].__setitem__(key, {"path": "missing.log", "sha256": "0" * 64}))

    def test_dropped_claim_id_rejected(self):
        ledger = self.ledger()
        ledger["claims"].pop()
        with self.assertRaises(verification.VerificationFailure):
            verification.check_claim_ledger(ledger, set())

    def test_local_path_cannot_escape(self):
        for path in ("../kbound/kbound_tmlr.tex", "/tmp/outside", ""):
            with self.assertRaises(verification.VerificationFailure):
                verification.local_file(path)

    def test_pending_lean_audit_not_accepted(self):
        ledger = self.ledger()
        ledger["formal_evidence"]["status"] = "PENDING"
        with self.assertRaises(verification.VerificationFailure):
            verification.check_formal_inventory(ledger)

    def test_changed_lean_mutation_source_rejected(self):
        receipt = json.loads((TRACK / "formal/verification/mutation-checks.json").read_text())
        receipt["probes"][0]["source_text"] += "\n-- changed after verification\n"
        with self.assertRaises(verification.VerificationFailure):
            verification.check_mutation_receipt(receipt)

    def test_false_lean_control_cannot_be_counted_as_rejected_if_it_passed(self):
        receipt = json.loads((TRACK / "formal/verification/mutation-checks.json").read_text())
        next(probe for probe in receipt["probes"] if probe["expected_exit"] == 1)["actual_exit"] = 0
        with self.assertRaises(verification.VerificationFailure):
            verification.check_mutation_receipt(receipt)

    def test_final_recorded_local_snapshot_matches(self):
        path = TRACK / verification.SNAPSHOT
        self.assertTrue(path.is_file(), "Record the reviewed local snapshot after the final formal inventory exists")
        verification.check_snapshot(verification.current_snapshot(), json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
