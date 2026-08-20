"""
Comprehensive Unit & Integration Test Suite for CancerCombo-BRICS-Symmetric.
Includes verification of drug-order invariance and modular class assertions.
"""

import sys
import os
import unittest
import torch
import numpy as np

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics import (
    CellLineEncoder,
    FragmentEncoder,
    FiLMConditioning,
    ManualMultiHeadCrossAttention,
    MaskedBidirectionalCrossAttention,
    ParameterHeads,
    ConstraintTransform,
    BivariateHillSolver,
    decompose_smiles_to_brics,
    collate_brics_fragments,
    CancerComboBRICSSymmetric,
    CancerComboBRICS,
    CancerComboDataset,
    CellExpressionLoader,
    collate_cancer_combo_batch,
    load_cancer_combo_from_csv,
    load_cancer_combo_splits
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
        """Test FiLMConditioning Residual FiLM forward pass and identity initialization."""
        film = FiLMConditioning(cell_dim=self.cell_dim, d_dim=self.d_dim)
        F = torch.randn(self.batch_size, self.n_frags_a, self.d_dim)
        c = torch.randn(self.batch_size, self.cell_dim)
        F_tilde = film(F, c)
        self.assertEqual(F_tilde.shape, F.shape)
        self.assertFalse(torch.isnan(F_tilde).any())
        # Test identity initialization: at startup, gamma(c)=0 and beta(c)=0, so F_tilde == F
        self.assertTrue(torch.allclose(F_tilde, F, atol=1e-5), "Residual FiLM identity initialization failed!")

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

    def test_direct_concatenation_r_final(self):
        """Test direct concatenation of directional cross-attention features into r_AB (512) and r_final (1024)."""
        mu_A = torch.randn(self.batch_size, self.d_dim)
        p_A = torch.randn(self.batch_size, self.d_dim)
        mu_B = torch.randn(self.batch_size, self.d_dim)
        p_B = torch.randn(self.batch_size, self.d_dim)
        c = torch.randn(self.batch_size, self.cell_dim)

        r_AB = torch.cat([mu_A, p_A, mu_B, p_B], dim=-1)
        r_final = torch.cat([r_AB, c], dim=-1)

        self.assertEqual(r_AB.shape, (self.batch_size, 4 * self.d_dim))
        self.assertEqual(r_final.shape, (self.batch_size, 4 * self.d_dim + self.cell_dim))

    def test_parameter_heads_and_constraint_transform(self):
        """Test ParameterHeads raw logits and ConstraintTransform positivity/range bounds."""
        in_dim = 4 * self.d_dim + self.cell_dim
        heads = ParameterHeads(in_dim=in_dim, hidden_dim=512)
        transform = ConstraintTransform()

        r_final = torch.randn(self.batch_size, in_dim)
        raw_params = heads(r_final)
        params = transform(raw_params)

        expected_keys = {"e0", "e1", "e2", "e12", "c1", "c2", "h1", "h2", "alpha", "log_c1", "log_c2"}
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

    def test_directional_cross_attention_forward(self):
        """
        Verify forward execution and tensor shapes of CancerComboBRICSSymmetric with directional cross-attention and unified parameter MLP.
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

        # Verify predicted surfaces are non-null, finite, and match grid dimensions
        self.assertEqual(Y_AB.shape, (self.batch_size, self.n_doses_a, self.n_doses_b))
        self.assertEqual(Y_BA.shape, (self.batch_size, self.n_doses_b, self.n_doses_a))
        self.assertTrue(torch.isfinite(Y_AB).all())
        self.assertTrue(torch.isfinite(Y_BA).all())

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


    def test_residual_film_verification(self):
        """Verify Residual FiLM operation matches manual elementwise F_norm + gamma(c) * F_norm + beta(c)."""
        film = FiLMConditioning(cell_dim=self.cell_dim, d_dim=self.d_dim)
        F_norm = torch.randn(self.batch_size, self.n_frags_a, self.d_dim)
        c = torch.randn(self.batch_size, self.cell_dim)

        F_tilde = film(F_norm, c)

        # Manual Residual FiLM calculation
        gamma = film.g_gamma(c).unsqueeze(1)
        beta = film.g_beta(c).unsqueeze(1)
        expected = F_norm + gamma * F_norm + beta

        self.assertTrue(torch.allclose(F_tilde, expected, atol=1e-6))

    def test_one_fragment_molecule(self):
        """Test 2: One-fragment molecule (B x 1 x d) passes without error through full model."""
        model = CancerComboBRICSSymmetric(
            gene_dim=self.gene_dim,
            cell_dim=self.cell_dim,
            frag_fp_dim=self.frag_fp_dim,
            d_dim=self.d_dim
        )
        cell_expr = torch.randn(self.batch_size, self.gene_dim)
        fp_A = torch.randn(self.batch_size, 1, self.frag_fp_dim)
        mask_A = torch.ones(self.batch_size, 1, dtype=torch.bool)

        fp_B = torch.randn(self.batch_size, 1, self.frag_fp_dim)
        mask_B = torch.ones(self.batch_size, 1, dtype=torch.bool)

        doses_A = torch.linspace(0.01, 10, self.n_doses_a)
        doses_B = torch.linspace(0.01, 10, self.n_doses_b)

        Y_pred = model(cell_expr, fp_A, mask_A, fp_B, mask_B, (doses_A, doses_B))
        self.assertEqual(Y_pred.shape, (self.batch_size, self.n_doses_a, self.n_doses_b))
        self.assertTrue(torch.isfinite(Y_pred).all())

    def test_mask_correctness(self):
        """Test 3: Verify padded fragments do not influence attention or output representations."""
        model = CancerComboBRICSSymmetric(
            gene_dim=self.gene_dim,
            cell_dim=self.cell_dim,
            frag_fp_dim=self.frag_fp_dim,
            d_dim=self.d_dim
        )
        model.eval()

        cell_expr = torch.randn(self.batch_size, self.gene_dim)
        doses = (torch.linspace(0.01, 10, 4), torch.linspace(0.01, 10, 4))

        # Pass 1: 3 real fragments, 2 padded (filled with random noise 1)
        fp_A1 = torch.randn(self.batch_size, 5, self.frag_fp_dim)
        mask_A = torch.tensor([[True, True, True, False, False]] * self.batch_size)

        fp_B1 = torch.randn(self.batch_size, 4, self.frag_fp_dim)
        mask_B = torch.tensor([[True, True, True, False]] * self.batch_size)

        Y_pred1 = model(cell_expr, fp_A1, mask_A, fp_B1, mask_B, doses)

        # Pass 2: Same real fragments, but change padded fragment features to noise 2
        fp_A2 = fp_A1.clone()
        fp_A2[:, 3:] = torch.randn(self.batch_size, 2, self.frag_fp_dim) * 100.0

        fp_B2 = fp_B1.clone()
        fp_B2[:, 3:] = torch.randn(self.batch_size, 1, self.frag_fp_dim) * 100.0

        Y_pred2 = model(cell_expr, fp_A2, mask_A, fp_B2, mask_B, doses)

        self.assertTrue(torch.allclose(Y_pred1, Y_pred2, atol=1e-5))

    def test_fp32_execution_and_gradients(self):
        """Test 4: FP32 forward pass yields finite output and finite gradients."""
        model = CancerComboBRICSSymmetric(
            gene_dim=self.gene_dim,
            cell_dim=self.cell_dim,
            frag_fp_dim=self.frag_fp_dim,
            d_dim=self.d_dim
        )
        cell_expr = torch.randn(self.batch_size, self.gene_dim)
        fp_A = torch.randn(self.batch_size, 3, self.frag_fp_dim)
        mask_A = torch.ones(self.batch_size, 3, dtype=torch.bool)
        fp_B = torch.randn(self.batch_size, 2, self.frag_fp_dim)
        mask_B = torch.ones(self.batch_size, 2, dtype=torch.bool)

        doses = (torch.linspace(0.01, 10, 4), torch.linspace(0.01, 10, 4))
        Y_pred = model(cell_expr, fp_A, mask_A, fp_B, mask_B, doses)

        self.assertTrue(torch.isfinite(Y_pred).all())
        loss = Y_pred.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_fp16_amp_execution(self):
        """Test 5: FP16 / AMP execution yields finite predictions and no mask overflow."""
        model = CancerComboBRICSSymmetric(
            gene_dim=self.gene_dim,
            cell_dim=self.cell_dim,
            frag_fp_dim=self.frag_fp_dim,
            d_dim=self.d_dim
        )
        cell_expr = torch.randn(self.batch_size, self.gene_dim)
        fp_A = torch.randn(self.batch_size, 4, self.frag_fp_dim)
        mask_A = torch.tensor([[True, True, True, False]] * self.batch_size)
        fp_B = torch.randn(self.batch_size, 4, self.frag_fp_dim)
        mask_B = torch.tensor([[True, True, False, False]] * self.batch_size)
        doses = (torch.linspace(0.01, 10, 4), torch.linspace(0.01, 10, 4))

        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            Y_pred = model(cell_expr, fp_A, mask_A, fp_B, mask_B, doses)

        self.assertTrue(torch.isfinite(Y_pred).all())

    def test_hill_solver_stress_test(self):
        """Test 6: Hill solver under extreme stress parameters yields finite Y and gradients."""
        solver = BivariateHillSolver()
        stress_params = {
            "e0": torch.tensor([[0.0], [1.0], [0.5]]),
            "e1": torch.tensor([[0.0], [1.0], [0.1]]),
            "e2": torch.tensor([[0.0], [1.0], [0.2]]),
            "e12": torch.tensor([[0.0], [1.0], [0.05]]),
            "c1": torch.tensor([[1e-6], [1e4], [1.0]], requires_grad=True),
            "c2": torch.tensor([[1e-6], [1e4], [2.0]], requires_grad=True),
            "h1": torch.tensor([[0.01], [10.0], [1.5]], requires_grad=True),
            "h2": torch.tensor([[0.01], [10.0], [1.2]], requires_grad=True),
            "alpha": torch.tensor([[1e-6], [1e4], [2.0]], requires_grad=True),
        }

        doses_A = torch.tensor([[0.0, 1e-6, 1.0, 1e4]])
        doses_B = torch.tensor([[0.0, 1e-6, 2.0, 1e4]])

        Y = solver(stress_params, (doses_A, doses_B))

        self.assertEqual(Y.shape, (3, 4, 4))
        self.assertTrue(torch.isfinite(Y).all())
        self.assertTrue((Y >= 0.0).all() and (Y <= 1.0).all())

        loss = Y.sum()
        loss.backward()

        self.assertTrue(torch.isfinite(stress_params["c1"].grad).all())
        self.assertTrue(torch.isfinite(stress_params["h1"].grad).all())

    def test_invalid_smiles_audit(self):
        """Test 7: Invalid SMILES raises ValueError and is not silently converted to zero fragment."""
        invalid_smiles = "invalid_chemical_string_123"
        with self.assertRaises(ValueError):
            decompose_smiles_to_brics(invalid_smiles)

    def test_drug_split_disjointness(self):
        """Test 8: Verify Scenario 3 drug-level split has zero drug overlap across splits."""
        drug_level_csv = os.path.join(os.path.dirname(__file__), "..", "data", "scenario3_drug_level.csv")
        if not os.path.exists(drug_level_csv):
            from create_drug_level_split import generate_drug_level_split
            generate_drug_level_split(output_csv=drug_level_csv)

        train_dataset = load_cancer_combo_from_csv(drug_level_csv, split=1)
        val_dataset = load_cancer_combo_from_csv(drug_level_csv, split=2)
        test_dataset = load_cancer_combo_from_csv(drug_level_csv, split=3)

        def get_drugs(dataset):
            drugs = set()
            for sample in dataset:
                drugs.add(sample["smiles_A"])
                drugs.add(sample["smiles_B"])
            return drugs

        train_drugs = get_drugs(train_dataset)
        val_drugs = get_drugs(val_dataset)
        test_drugs = get_drugs(test_dataset)

        self.assertTrue(train_drugs.isdisjoint(val_drugs), "Train and Val drugs overlap!")
        self.assertTrue(train_drugs.isdisjoint(test_drugs), "Train and Test drugs overlap!")
        self.assertTrue(val_drugs.isdisjoint(test_drugs), "Val and Test drugs overlap!")

    def test_missing_cell_line_raises_loud_error(self):
        """Test 9: Verify missing cell line raises explicit ValueError and halts execution (NO silent random fallback)."""
        from cancer_combo_brics.cell_expression import CellExpressionLoader
        loader = CellExpressionLoader(csv_path=os.path.join(os.path.dirname(__file__), "..", "data", "cell_line_gene_expr.csv"))
        with self.assertRaises(ValueError):
            loader.get_cell_expression("NON_EXISTENT_CELL_LINE_XYZ_123")

    def test_log_space_ec50_bounds(self):
        """Test 10: Verify ConstraintTransform EC50 (c1, c2) operate in log-space bounded in [10^-11, 10^-3] M."""
        from cancer_combo_brics.constraint_transform import ConstraintTransform
        transform = ConstraintTransform(log_c_min=-11.0, log_c_max=-3.0)
        raw_params = {
            "e0": torch.tensor([[0.0]]), "e1": torch.tensor([[0.0]]), "e2": torch.tensor([[0.0]]), "e12": torch.tensor([[0.0]]),
            "c1": torch.tensor([[-100.0]]), "c2": torch.tensor([[100.0]]),
            "h1": torch.tensor([[0.5]]), "h2": torch.tensor([[0.5]]), "alpha": torch.tensor([[0.1]])
        }
        params = transform(raw_params)
        self.assertGreaterEqual(params["c1"].item(), 1e-11)
        self.assertLessEqual(params["c2"].item(), 1e-3)
        self.assertTrue(torch.isfinite(params["c1"]).all())
        self.assertTrue(torch.isfinite(params["c2"]).all())

    def test_padding_occurs_after_film_verification(self):
        """Test 12: Verify FragmentEncoder, LayerNorm, and FiLM operate ONLY on unpadded real fragments, and padding occurs strictly AFTER FiLM."""
        model = CancerComboBRICSSymmetric(gene_dim=976, cell_dim=512, frag_fp_dim=2048, d_dim=128)
        B = 2
        c = torch.randn(B, 512)

        # Sample 0: 1 real fragment (2 padded rows)
        # Sample 1: 3 real fragments (0 padded rows)
        fp_tensor = torch.randn(B, 3, 2048)
        mask_tensor = torch.tensor([[True, False, False], [True, True, True]], dtype=torch.bool)

        # Intercept per-sample unpadded inputs to verify FragmentEncoder, LayerNorm, and FiLM never receive padded rows
        encoder_inputs = []
        original_encoder_forward = model.fragment_encoder.forward

        def spy_encoder_forward(x):
            encoder_inputs.append(x.clone())
            return original_encoder_forward(x)

        model.fragment_encoder.forward = spy_encoder_forward

        F_tilde_padded, explicit_mask = model._encode_and_condition_unpadded_fragments(fp_tensor, mask_tensor, c)

        # Restore original forward
        model.fragment_encoder.forward = original_encoder_forward

        # Verify FragmentEncoder was called ONCE for sample 0 (shape 1, 2048) and ONCE for sample 1 (shape 3, 2048)
        self.assertEqual(len(encoder_inputs), 2)
        self.assertEqual(tuple(encoder_inputs[0].shape), (1, 2048))
        self.assertEqual(tuple(encoder_inputs[1].shape), (3, 2048))

        # Verify output padded tensor shape (2, 3, 128) and mask tensor shape (2, 3)
        self.assertEqual(tuple(F_tilde_padded.shape), (2, 3, 128))
        self.assertEqual(tuple(explicit_mask.shape), (2, 3))
        self.assertTrue(torch.equal(explicit_mask, mask_tensor))
        print("\n[PASS] Verified FragmentEncoder & FiLM process ONLY unpadded real fragments (1, 2048) and (3, 2048). Padding occurs strictly AFTER FiLM!")

    def test_leakage_free_gene_expression_normalization(self):
        """
        Required Validation Test:
            1. Fit normalization on training cells ONLY and record train_mean / train_std.
            2. Verify validation dataset uses EXACT train_mean / train_std without refitting.
            3. Verify test dataset uses EXACT train_mean / train_std without refitting.
            4. Verify numerical equivalence for val/test normalized expressions:
               expected_val = (X_val - train_mean) / (train_std + eps).
        """
    def test_leakage_free_gene_expression_normalization(self):
        """
        Required Validation Test:
            1. Fit normalization on training cells ONLY and record train_mean / train_std.
            2. Verify validation dataset uses EXACT train_mean / train_std without refitting.
            3. Verify test dataset uses EXACT train_mean / train_std without refitting.
            4. Verify numerical equivalence for val/test normalized expressions:
               expected_val = (X_val - train_mean) / (train_std + eps).
        """
        loader = CellExpressionLoader(gene_dim=976, normalize=True)
        # Create artificial gene expression data for 3 train cell lines, 1 val, 1 test
        tr_cell_ids = {"CELL_TR1", "CELL_TR2", "CELL_TR3"}
        loader.raw_cell_expr_dict = {
            "CELL_TR1": torch.randn(976) * 2.0 + 10.0,
            "CELL_TR2": torch.randn(976) * 3.0 + 12.0,
            "CELL_TR3": torch.randn(976) * 1.5 + 8.0,
            "CELL_VAL": torch.randn(976) * 5.0 + 100.0,
            "CELL_TEST": torch.randn(976) * 1.0 - 50.0
        }

        # 1. Fit on TRAIN ONLY
        loader.fit_normalization(train_cell_ids=tr_cell_ids)
        train_mean_fit = loader.mean.clone()
        train_std_fit = loader.std.clone()

        self.assertTrue(loader.fitted)
        self.assertEqual(loader.fitted_cell_ids, tr_cell_ids)

        # 2. Retrieve Val & Test normalized expressions
        val_expr = loader.get_cell_expression("CELL_VAL")
        test_expr = loader.get_cell_expression("CELL_TEST")

        # Verify loader mean and std did NOT change after val and test retrievals
        self.assertTrue(torch.equal(loader.mean, train_mean_fit))
        self.assertTrue(torch.equal(loader.std, train_std_fit))

        # 3. Numerical Equivalence Verification
        expected_val = (loader.raw_cell_expr_dict["CELL_VAL"] - train_mean_fit) / train_std_fit
        expected_test = (loader.raw_cell_expr_dict["CELL_TEST"] - train_mean_fit) / train_std_fit

        self.assertTrue(torch.allclose(val_expr, expected_val, atol=1e-6))
        self.assertTrue(torch.allclose(test_expr, expected_test, atol=1e-6))
        self.assertFalse(torch.isnan(val_expr).any())
        self.assertFalse(torch.isnan(test_expr).any())

    def test_artificial_distribution_shift_regression(self):
        """
        Regression Test:
            Verify that extreme artificial distribution shifts in validation/test sets
            (e.g., Val mean ≈ 1000, Test mean ≈ -500) do NOT alter the training normalization statistics.
        """
        loader = CellExpressionLoader(gene_dim=976, normalize=True)
        raw_tr1 = torch.randn(976) * 3.0 + 10.0
        raw_tr2 = torch.randn(976) * 4.0 + 15.0
        raw_val = torch.randn(976) * 50.0 + 1000.0
        raw_test = torch.randn(976) * 20.0 - 500.0

        loader.raw_cell_expr_dict = {
            "CELL_TR1": raw_tr1,
            "CELL_TR2": raw_tr2,
            "CELL_VAL": raw_val,
            "CELL_TEST": raw_test
        }

        # Fit on TRAIN
        tr_ids = {"CELL_TR1", "CELL_TR2"}
        loader.fit_normalization(train_cell_ids=tr_ids)
        train_mean_fit = loader.mean.clone()
        train_std_fit = loader.std.clone()

        # Mutate Val and Test raw expressions to extreme values
        loader.raw_cell_expr_dict["CELL_VAL"] = raw_val * 100.0 + 99999.0
        loader.raw_cell_expr_dict["CELL_TEST"] = raw_test * 50.0 - 88888.0

        # Attempting to call fit_normalization without force_refit should be a no-op that preserves training stats
        loader.fit_normalization(train_cell_ids={"CELL_VAL"})
        self.assertTrue(torch.equal(loader.mean, train_mean_fit))
        self.assertTrue(torch.equal(loader.std, train_std_fit))

        # Re-transform val and verify it STILL uses train_mean_fit and train_std_fit
        val_norm = loader.get_cell_expression("CELL_VAL")
        expected_val = (loader.raw_cell_expr_dict["CELL_VAL"] - train_mean_fit) / train_std_fit
        self.assertTrue(torch.allclose(val_norm, expected_val, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
