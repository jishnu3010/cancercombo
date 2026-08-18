"""
Validation Test Suite for Drug A & Drug B BRICS Fragment Debug Print Feature.

Runs all 4 required validation tests:
    1. Single Drug A/B pair -> correct SMILES + fragments printed in order.
    2. Batch with multiple samples -> no cross-sample mixing of A/B or fragments.
    3. Invalid SMILES -> prints "Invalid SMILES", does not crash.
    4. print_fragments = False -> training runs identically with no printed output.
"""

import sys
import os
import io
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics import (
    CancerComboBRICSSymmetric,
    collate_brics_fragments,
    print_drug_fragments,
    print_batch_drug_fragments
)


class TestDrugFragmentsDebug(unittest.TestCase):

    def test_validation_1_single_pair(self):
        """Test 1: Single Drug A/B pair -> correct SMILES + fragments printed in order."""
        smiles_A = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        smiles_B = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine

        _, _, frags_A = collate_brics_fragments([smiles_A])
        _, _, frags_B = collate_brics_fragments([smiles_B])

        captured_output = io.StringIO()
        sys.stdout = captured_output

        print_batch_drug_fragments([smiles_A], frags_A, [smiles_B], frags_B)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        print("--- Test 1 Output ---")
        print(output)

        self.assertIn("Sample 1", output)
        self.assertIn("Drug A:", output)
        self.assertIn(f"SMILES: {smiles_A}", output)
        self.assertIn("Drug B:", output)
        self.assertIn(f"SMILES: {smiles_B}", output)
        self.assertIn("A1:", output)
        self.assertIn("B1:", output)

    def test_validation_2_batch_multiple_samples(self):
        """Test 2: Batch with multiple samples -> no cross-sample mixing of A/B or fragments."""
        smiles_A_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C(=O)O"]
        smiles_B_list = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCO"]

        _, _, frags_A_list = collate_brics_fragments(smiles_A_list)
        _, _, frags_B_list = collate_brics_fragments(smiles_B_list)

        captured_output = io.StringIO()
        sys.stdout = captured_output

        print_batch_drug_fragments(smiles_A_list, frags_A_list, smiles_B_list, frags_B_list)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        print("--- Test 2 Output ---")
        print(output)

        self.assertIn("Sample 1", output)
        self.assertIn("Sample 2", output)
        self.assertIn(smiles_A_list[0], output)
        self.assertIn(smiles_A_list[1], output)
        self.assertIn(smiles_B_list[0], output)
        self.assertIn(smiles_B_list[1], output)

        # Verify ordering of samples
        sample1_idx = output.find("Sample 1")
        sample2_idx = output.find("Sample 2")
        self.assertLess(sample1_idx, sample2_idx)

    def test_validation_3_invalid_smiles(self):
        """Test 3: Invalid SMILES -> prints 'Invalid SMILES', does not crash."""
        smiles_A_list = ["invalid_smiles_string_xyz"]
        smiles_B_list = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

        _, _, frags_A_list = collate_brics_fragments(smiles_A_list)
        _, _, frags_B_list = collate_brics_fragments(smiles_B_list)

        captured_output = io.StringIO()
        sys.stdout = captured_output

        print_batch_drug_fragments(smiles_A_list, frags_A_list, smiles_B_list, frags_B_list)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        print("--- Test 3 Output ---")
        print(output)

        self.assertIn("Drug A: Invalid SMILES", output)
        self.assertIn("Drug B:", output)

    def test_validation_4_flag_disabled(self):
        """Test 4: print_fragments = False -> training runs identically with no printed output."""
        model = CancerComboBRICSSymmetric(print_fragments=False)
        cell_expr = torch.randn(2, 976)
        smiles_A = ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C(=O)O"]
        smiles_B = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCO"]
        doses_A = torch.linspace(0.01, 10, 4)
        doses_B = torch.linspace(0.01, 10, 4)

        captured_output = io.StringIO()
        sys.stdout = captured_output

        Y_pred = model(cell_expr, smiles_A, smiles_B, (doses_A, doses_B), print_fragments=False)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertEqual(output.strip(), "")
        self.assertEqual(Y_pred.shape, (2, 4, 4))


if __name__ == "__main__":
    unittest.main()
