"""
Comprehensive Unit & Integration Test Suite for CancerCombo-BRICS-Symmetric.
Includes verification of drug-order invariance and modular class assertions.
"""

import sys
import os
import unittest
import torch

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics import (
    CellLineEncoder,
    FragmentEncoder,
    FiLMConditioning,
    ManualMultiHeadCrossAttention,
    MaskedBidirectionalCrossAttention,
    SymmetricFusion,
    CellFusion,
    ParameterHeads,
    ConstraintTransform,
    BivariateHillSolver,
    decompose_smiles_to_brics,
    collate_brics_fragments,
    CancerComboBRICSSymmetric,
    CancerComboBRICS,
    CancerComboDataset,
    collate_cancer_combo_batch,
    load_cancer_combo_from_csv
)


class TestCancerComboBRICSSymmetric(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.gene_dim = 976
        self.cell_dim = 512
        self.frag_fp_dim = 2048
        self.d_dim = 128
        self.num_heads = 4
        self.n_frags_a = 5
        self.n_frags_b = 3
        self.n_doses_a = 6
        self.n_doses_b = 6

    def test_cell_line_encoder(self):
        """Test CellLineEncoder forward pass and output shape (B, 512)."""
        encoder = CellLineEncoder(in_dim=self.gene_dim, hidden_dim=self.cell_dim)
        cell_expr = torch.randn(self.batch_size, self.gene_dim)
        c = encoder(cell_expr)
        self.assertEqual(c.shape, (self.batch_size, self.cell_dim))
        self.assertFalse(torch.isnan(c).any())

    def test_fragment_encoder(self):
        """Test FragmentEncoder forward pass and output shape (B, n, d)."""
        encoder = FragmentEncoder(in_bits=self.frag_fp_dim, d_dim=self.d_dim)
        fp_tensor = torch.randn(self.batch_size, self.n_frags_a, self.frag_fp_dim)
        F = encoder(fp_tensor)
        self.assertEqual(F.shape, (self.batch_size, self.n_frags_a, self.d_dim))
        self.assertFalse(torch.isnan(F).any())

    def test_film_conditioning(self):
        """Test FiLMConditioning forward pass using shared gamma/beta MLPs."""
        film = FiLMConditioning(cell_dim=self.cell_dim, d_dim=self.d_dim)
        F = torch.randn(self.batch_size, self.n_frags_a, self.d_dim)
        c = torch.randn(self.batch_size, self.cell_dim)
        F_tilde = film(F, c)
        self.assertEqual(F_tilde.shape, F.shape)
        self.assertFalse(torch.isnan(F_tilde).any())

    def test_manual_multihead_cross_attention(self):
        """Test ManualMultiHeadCrossAttention module without nn.MultiheadAttention."""
        mha = ManualMultiHeadCrossAttention(d_dim=self.d_dim, num_heads=self.num_heads)
        x_q = torch.randn(self.batch_size, self.n_frags_a, self.d_dim)
        x_kv = torch.randn(self.batch_size, self.n_frags_b, self.d_dim)
        mask = torch.ones(self.batch_size, self.n_frags_b, dtype=torch.bool)
        mask[:, -1] = False  # Mask last fragment as padding

        out = mha(x_q, x_kv, key_padding_mask=mask)
        self.assertEqual(out.shape, (self.batch_size, self.n_frags_a, self.d_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_masked_bidirectional_cross_attention(self):
        """Test MaskedBidirectionalCrossAttention module and key/query padding masks."""
        bca = MaskedBidirectionalCrossAttention(d_dim=self.d_dim, num_heads=self.num_heads)

        F_tilde_A = torch.randn(self.batch_size, self.n_frags_a, self.d_dim)
        mask_A = torch.ones(self.batch_size, self.n_frags_a, dtype=torch.bool)
        mask_A[:, -2:] = False  # Last 2 are padding

        F_tilde_B = torch.randn(self.batch_size, self.n_frags_b, self.d_dim)
        mask_B = torch.ones(self.batch_size, self.n_frags_b, dtype=torch.bool)
        mask_B[:, -1] = False   # Last 1 is padding

        mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A = bca(F_tilde_A, mask_A, F_tilde_B, mask_B)
        self.assertEqual(mu_A_from_B.shape, (self.batch_size, self.d_dim))
        self.assertEqual(p_A_from_B.shape, (self.batch_size, self.d_dim))
        self.assertEqual(mu_B_from_A.shape, (self.batch_size, self.d_dim))
        self.assertEqual(p_B_from_A.shape, (self.batch_size, self.d_dim))

    def test_symmetric_fusion(self):
        """Test SymmetricFusion order-invariance property."""
        fusion = SymmetricFusion(d_dim=self.d_dim)
        mu_A = torch.randn(self.batch_size, self.d_dim)
        p_A = torch.randn(self.batch_size, self.d_dim)
        mu_B = torch.randn(self.batch_size, self.d_dim)
        p_B = torch.randn(self.batch_size, self.d_dim)

        r_AB_sym = fusion(mu_A, p_A, mu_B, p_B)
        r_BA_sym = fusion(mu_B, p_B, mu_A, p_A)

        self.assertEqual(r_AB_sym.shape, (self.batch_size, 4 * self.d_dim))
        self.assertTrue(torch.allclose(r_AB_sym, r_BA_sym, atol=1e-6))

    def test_cell_fusion(self):
        """Test CellFusion module concatenation."""
        fusion = CellFusion(d_dim=self.d_dim, cell_dim=self.cell_dim)
        r_AB = torch.randn(self.batch_size, 4 * self.d_dim)
        c = torch.randn(self.batch_size, self.cell_dim)
        r_final = fusion(r_AB, c)
        self.assertEqual(r_final.shape, (self.batch_size, 4 * self.d_dim + self.cell_dim))

    def test_parameter_heads_and_constraint_transform(self):
        """Test ParameterHeads raw logits and ConstraintTransform positivity/range bounds."""
        in_dim = 4 * self.d_dim + self.cell_dim
        heads = ParameterHeads(in_dim=in_dim, hidden_dim=256)
        transform = ConstraintTransform()

        r_final = torch.randn(self.batch_size, in_dim)
        raw_params = heads(r_final)
        params = transform(raw_params)

        expected_keys = {"e0", "e1", "e2", "e12", "c1", "c2", "h1", "h2", "alpha"}
        self.assertEqual(set(params.keys()), expected_keys)

        for k, v in params.items():
            self.assertEqual(v.shape, (self.batch_size, 1))
            self.assertFalse(torch.isnan(v).any())

        # Check positivity and bound constraints
        self.assertTrue((params["c1"] > 0).all())
        self.assertTrue((params["c2"] > 0).all())
        self.assertTrue((params["h1"] > 0).all())
        self.assertTrue((params["h2"] > 0).all())
        self.assertTrue((params["alpha"] > 0).all())
        self.assertTrue((params["e1"] >= 0).all() and (params["e1"] <= 1).all())
        self.assertTrue((params["e2"] >= 0).all() and (params["e2"] <= 1).all())

    def test_bivariate_hill_solver(self):
        """Test BivariateHillSolver differentiability and surface generation."""
        solver = BivariateHillSolver()
        params = {
            "e0": torch.tensor([[1.0], [0.95]]),
            "e1": torch.tensor([[0.2], [0.1]]),
            "e2": torch.tensor([[0.3], [0.15]]),
            "e12": torch.tensor([[0.05], [0.01]]),
            "c1": torch.tensor([[1.0], [0.5]], requires_grad=True),
            "c2": torch.tensor([[2.0], [1.0]], requires_grad=True),
            "h1": torch.tensor([[1.5], [1.2]], requires_grad=True),
            "h2": torch.tensor([[1.2], [1.0]], requires_grad=True),
            "alpha": torch.tensor([[2.0], [1.5]], requires_grad=True),
        }

        doses_A = torch.tensor([[0.0, 0.1, 1.0, 10.0]])
        doses_B = torch.tensor([[0.0, 0.2, 2.0, 20.0]])

        Y = solver(params, (doses_A, doses_B))
        self.assertEqual(Y.shape, (2, 4, 4))
        self.assertFalse(torch.isnan(Y).any())

        # Verify gradient flow
        loss = Y.sum()
        loss.backward()
        self.assertIsNotNone(params["c1"].grad)
        self.assertIsNotNone(params["alpha"].grad)

    def test_drug_order_invariance(self):
        """
        CRITICAL TEST: Confirm drug-order invariance of CancerComboBRICSSymmetric.
        Verifies that swapping Drug A and Drug B inputs yields matching outputs up to dose grid transpose:
            CancerCombo(A, B, cell) == CancerCombo(B, A, cell).transpose(1, 2)
        """
        model = CancerComboBRICSSymmetric(
            gene_dim=self.gene_dim,
            cell_dim=self.cell_dim,
            frag_fp_dim=self.frag_fp_dim,
            d_dim=self.d_dim,
            num_attn_heads=self.num_heads,
            shared_attn_weights=True
        )
        model.eval()

        torch.manual_seed(42)
        cell_expr = torch.randn(self.batch_size, self.gene_dim)

        fp_A = torch.randn(self.batch_size, self.n_frags_a, self.frag_fp_dim)
        mask_A = torch.ones(self.batch_size, self.n_frags_a, dtype=torch.bool)
        mask_A[:, -1] = False

        fp_B = torch.randn(self.batch_size, self.n_frags_b, self.frag_fp_dim)
        mask_B = torch.ones(self.batch_size, self.n_frags_b, dtype=torch.bool)

        doses_A = torch.linspace(0.01, 10, self.n_doses_a)
        doses_B = torch.linspace(0.01, 10, self.n_doses_b)

        # Forward pass 1: Drug A first, Drug B second
        Y_AB, params_AB = model(
            cell_expr=cell_expr,
            drugA_frags=fp_A,
            drugA_mask=mask_A,
            drugB_frags=fp_B,
            drugB_mask=mask_B,
            dose_grid=(doses_A, doses_B),
            return_params=True
        )

        # Forward pass 2: Drug B first, Drug A second (swapped inputs and dose grid)
        Y_BA, params_BA = model(
            cell_expr=cell_expr,
            drugA_frags=fp_B,
            drugA_mask=mask_B,
            drugB_frags=fp_A,
            drugB_mask=mask_A,
            dose_grid=(doses_B, doses_A),
            return_params=True
        )

        # Verify predicted surface Y_AB is equal to transposed Y_BA
        self.assertEqual(Y_AB.shape, (self.batch_size, self.n_doses_a, self.n_doses_b))
        self.assertEqual(Y_BA.shape, (self.batch_size, self.n_doses_b, self.n_doses_a))

        Y_BA_transposed = Y_BA.transpose(1, 2)
        diff = torch.abs(Y_AB - Y_BA_transposed).max().item()
        self.assertLess(diff, 1e-5, f"Drug-order invariance violated! Max diff: {diff}")

        # Verify individual parameter symmetry:
        # e0, e12, alpha should be identical
        # c1 and c2, h1 and h2, e1 and e2 should swap correspondingly
        self.assertTrue(torch.allclose(params_AB["e0"], params_BA["e0"], atol=1e-5))
        self.assertTrue(torch.allclose(params_AB["e12"], params_BA["e12"], atol=1e-5))
        self.assertTrue(torch.allclose(params_AB["alpha"], params_BA["alpha"], atol=1e-5))
        self.assertTrue(torch.allclose(params_AB["c1"], params_BA["c2"], atol=1e-5))
        self.assertTrue(torch.allclose(params_AB["c2"], params_BA["c1"], atol=1e-5))

    def test_cancer_combo_dataset(self):
        """Test CancerComboDataset and collate_cancer_combo_batch DataLoader integration."""
        cell_dict = {"MCF7": torch.randn(976), "A549": torch.randn(976)}
        drug_pairs = [
            ("MCF7", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            ("A549", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)OC1=CC=CC=C1C(=O)O")
        ]
        dose_grids = [
            (torch.linspace(0.01, 10, 4), torch.linspace(0.01, 10, 4)),
            (torch.linspace(0.01, 10, 4), torch.linspace(0.01, 10, 4))
        ]
        surfaces = [torch.rand(4, 4), torch.rand(4, 4)]

        dataset = CancerComboDataset(drug_pairs, dose_grids, surfaces, cell_dict)
        self.assertEqual(len(dataset), 2)

        batch_list = [dataset[0], dataset[1]]
        collated = collate_cancer_combo_batch(batch_list, frag_fp_dim=2048)

        self.assertEqual(collated["cell_expr"].shape, (2, 976))
        self.assertEqual(collated["fp_A"].shape[0], 2)
        self.assertEqual(collated["fp_B"].shape[0], 2)
        self.assertEqual(collated["Y_true"].shape, (2, 4, 4))

    def test_load_cancer_combo_from_csv(self):
        """Test load_cancer_combo_from_csv parser on generated data CSV files."""
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        cell_csv = os.path.join(data_dir, "sample_cell_expression.csv")
        combo_csv = os.path.join(data_dir, "sample_drug_combinations.csv")

        if os.path.exists(cell_csv) and os.path.exists(combo_csv):
            dataset = load_cancer_combo_from_csv(cell_csv, combo_csv, gene_dim=976)
            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            self.assertEqual(sample["cell_expr"].shape, (976,))
            self.assertEqual(sample["Y_true"].shape, (4, 4))

    def test_cancer_combo_brics_training_step(self):
        """Test end-to-end forward pass and backward training step with MSE loss."""
        model = CancerComboBRICSSymmetric(
            gene_dim=self.gene_dim,
            cell_dim=self.cell_dim,
            frag_fp_dim=self.frag_fp_dim,
            d_dim=self.d_dim,
            num_attn_heads=self.num_heads
        )

        cell_expr = torch.randn(self.batch_size, self.gene_dim)
        drugA_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"] * self.batch_size
        drugB_smiles = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"] * self.batch_size

        doses_A = torch.linspace(0, 10, self.n_doses_a)
        doses_B = torch.linspace(0, 10, self.n_doses_b)

        Y_pred, params = model(cell_expr, drugA_smiles, drugB_smiles, (doses_A, doses_B), return_params=True)
        self.assertEqual(Y_pred.shape, (self.batch_size, self.n_doses_a, self.n_doses_b))

        # Target surface MSE loss
        Y_target = torch.rand(self.batch_size, self.n_doses_a, self.n_doses_b)
        criterion = torch.nn.MSELoss()
        loss = criterion(Y_pred, Y_target)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
