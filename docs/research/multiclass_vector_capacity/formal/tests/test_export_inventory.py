"""Fail-closed behavior of the kernel-message inventory parser."""

import json
import unittest

from export_inventory import AuditFailure, parse_messages


NAME = "MulticlassVectorCapacity.example_theorem"
COMMANDS = {10: ("type", NAME), 11: ("axioms", NAME)}


def message(line, data, severity="information"):
    return json.dumps({"pos": {"line": line}, "data": data, "severity": severity})


def valid_messages():
    return [
        message(3, "MVC_NAMESPACE_AUDIT_PASS 1"),
        message(4, f"MVC_DIRECT_DEPENDENCIES {NAME} []"),
        message(10, f"{NAME} : True"),
        message(11, f"'{NAME}' depends on axioms: [propext, Classical.choice.{{u}}, Quot.sound.{{u}}]"),
    ]


class InventoryParserTests(unittest.TestCase):
    def test_complete_verified_pair_preserves_type_and_normalizes_universe_parameters(self):
        parsed, checked = parse_messages("\n".join(valid_messages()), COMMANDS)
        self.assertEqual(checked, 1)
        self.assertEqual(parsed[NAME]["printed_type"], "True")
        self.assertEqual(parsed[NAME]["transitive_axioms"], ["Classical.choice", "Quot.sound", "propext"])
        self.assertEqual(parsed[NAME]["local_direct_dependencies"], [])

    def test_missing_axiom_report_fails(self):
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(valid_messages()[:-1]), COMMANDS)

    def test_missing_namespace_check_fails(self):
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(valid_messages()[1:]), COMMANDS)

    def test_duplicate_axiom_report_fails(self):
        lines = valid_messages()
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines + [lines[-1]]), COMMANDS)

    def test_custom_axiom_fails(self):
        lines = valid_messages()
        lines[-1] = message(11, f"'{NAME}' depends on axioms: [MulticlassVectorCapacity.hiddenOracle]")
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_compiler_error_cannot_be_masked_by_complete_printed_reports(self):
        lines = valid_messages() + [message(40, "declaration uses a proof gap", "error")]
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_compiler_warning_fails(self):
        lines = valid_messages() + [message(40, "declaration warning", "warning")]
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_elided_type_fails(self):
        lines = valid_messages()
        lines[2] = message(10, f"{NAME} : ∀ x, ⋯")
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_mismatched_name_fails(self):
        lines = valid_messages()
        lines[2] = message(10, "MulticlassVectorCapacity.other : True")
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_unregistered_message_fails(self):
        lines = valid_messages() + [message(80, "another theorem report")]
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_missing_dependency_report_fails(self):
        lines = valid_messages()
        del lines[1]
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)

    def test_dependency_outside_compiled_namespace_inventory_fails(self):
        lines = valid_messages()
        lines[1] = message(4, f'MVC_DIRECT_DEPENDENCIES {NAME} ["MulticlassVectorCapacity.missing"]')
        with self.assertRaises(AuditFailure):
            parse_messages("\n".join(lines), COMMANDS)


if __name__ == "__main__":
    unittest.main()
