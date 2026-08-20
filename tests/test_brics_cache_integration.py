"""
Unit Tests for BRICSCache Integration into CancerCombo Active DataLoader Pipeline.

Verifies:
    1. Cache miss populates cache on first lookup.
    2. Cache hit retrieves precomputed representation without recomputing.
    3. Multi-combination reuse (Drug A in A+B, A+C, A+D processed once).
    4. Numerical equivalence between cached and uncached outputs.
    5. Active DataLoader integration (batch creation reaches cache).
    6. Multi-worker safety (num_workers=0 and num_workers=2).
"""

import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader

from cancer_combo_brics.brics_cache import BRICSCache, get_global_brics_cache, set_global_brics_cache, reset_global_brics_cache
from cancer_combo_brics.brics_utils import collate_brics_fragments
from cancer_combo_brics.dataset import CancerComboDataset, collate_cancer_combo_batch


class TestBRICSCacheIntegration(unittest.TestCase):

    def setUp(self):
        reset_global_brics_cache()
        self.cache = BRICSCache(cache_file=None, n_bits=2048)
        self.smiles_aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.smiles_caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        self.smiles_ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

    def tearDown(self):
        reset_global_brics_cache()

    def test_1_cache_miss(self):
        """Test 1: Uncached SMILES causes cache miss and populates cache."""
        self.assertEqual(self.cache.requests, 0)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 0)

        frags, fp_matrix = self.cache.get_or_compute_brics(self.smiles_aspirin)

        self.assertEqual(self.cache.requests, 1)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 1)
        self.assertGreater(len(frags), 0)
        self.assertEqual(fp_matrix.shape[1], 2048)

    def test_2_cache_hit(self):
        """Test 2: Subsequent lookup of same SMILES results in cache hit."""
        frags1, fp1 = self.cache.get_or_compute_brics(self.smiles_aspirin)
        initial_requests = self.cache.requests
        initial_hits = self.cache.hits

        # Second call for exact same SMILES
        frags2, fp2 = self.cache.get_or_compute_brics(self.smiles_aspirin)

        self.assertEqual(self.cache.requests, initial_requests + 1)
        self.assertEqual(self.cache.hits, initial_hits + 1)
        self.assertEqual(frags1, frags2)
        np.testing.assert_array_equal(fp1, fp2)

    def test_3_same_drug_across_combinations(self):
        """Test 3: Drug A in (A+B, A+C, A+D) is processed once and reused."""
        smiles_list_A = [self.smiles_aspirin, self.smiles_aspirin, self.smiles_aspirin]
        smiles_list_B = [self.smiles_caffeine, self.smiles_ibuprofen, self.smiles_aspirin]

        # Reset counters
        self.cache.reset_stats()

        collate_brics_fragments(smiles_list_A, brics_cache=self.cache, use_cache=True)
        collate_brics_fragments(smiles_list_B, brics_cache=self.cache, use_cache=True)

        stats = self.cache.get_stats()
        # Unique SMILES: aspirin, caffeine, ibuprofen (3 total)
        self.assertEqual(stats["unique_drugs"], 3)
        self.assertEqual(stats["requests"], 6)
        # First calls for aspirin, caffeine, ibuprofen are misses (3), subsequent calls are hits (3)
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["hit_rate"], 50.0)

    def test_4_numerical_equivalence(self):
        """Test 4: Cached representations are numerically identical to uncached path."""
        smiles_batch = [self.smiles_aspirin, self.smiles_caffeine, self.smiles_ibuprofen]

        # Uncached collation
        fp_uncached, mask_uncached, frags_uncached = collate_brics_fragments(
            smiles_batch, n_bits=2048, use_cache=False
        )

        # Cached collation
        fp_cached, mask_cached, frags_cached = collate_brics_fragments(
            smiles_batch, n_bits=2048, brics_cache=self.cache, use_cache=True
        )

        self.assertEqual(frags_uncached, frags_cached)
        torch.testing.assert_close(fp_uncached, fp_cached)
        torch.testing.assert_close(mask_uncached, mask_cached)

    def test_5_dataloader_integration(self):
        """Test 5: Active DataLoader pipeline uses BRICSCache."""
        drug_pairs = [
            ("A549", self.smiles_aspirin, self.smiles_caffeine),
            ("A549", self.smiles_aspirin, self.smiles_ibuprofen),
            ("HCT116", self.smiles_caffeine, self.smiles_aspirin),
        ]
        dose_grids = [(torch.zeros(4), torch.zeros(4)) for _ in drug_pairs]
        viab_surfaces = [torch.ones(4, 4) for _ in drug_pairs]
        cell_expr_dict = {
            "A549": torch.randn(976),
            "HCT116": torch.randn(976)
        }

        set_global_brics_cache(self.cache)
        dataset = CancerComboDataset(
            drug_pairs=drug_pairs,
            dose_grids=dose_grids,
            viability_surfaces=viab_surfaces,
            cell_expr_dict=cell_expr_dict,
            brics_cache=self.cache
        )

        # Precompute features
        unique_smiles = [self.smiles_aspirin, self.smiles_caffeine, self.smiles_ibuprofen]
        self.cache.precompute_dataset_drugs(unique_smiles)
        self.cache.reset_stats()

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_cancer_combo_batch
        )

        for batch in loader:
            self.assertIn("fp_A", batch)
            self.assertIn("fp_B", batch)
            self.assertEqual(batch["fp_A"].shape[0], len(batch["cell_expr"]))

        stats = self.cache.get_stats()
        self.assertGreater(stats["requests"], 0)
        self.assertEqual(stats["misses"], 0)  # 100% precomputed cache hits!
        self.assertEqual(stats["hit_rate"], 100.0)

    def test_6_multiple_workers(self):
        """Test 6: DataLoader compatibility with num_workers > 0."""
        drug_pairs = [
            ("A549", self.smiles_aspirin, self.smiles_caffeine),
            ("A549", self.smiles_ibuprofen, self.smiles_aspirin)
        ] * 4
        dose_grids = [(torch.zeros(4), torch.zeros(4)) for _ in drug_pairs]
        viab_surfaces = [torch.ones(4, 4) for _ in drug_pairs]
        cell_expr_dict = {"A549": torch.randn(976)}

        dataset = CancerComboDataset(
            drug_pairs=drug_pairs,
            dose_grids=dose_grids,
            viability_surfaces=viab_surfaces,
            cell_expr_dict=cell_expr_dict,
            brics_cache=self.cache
        )

        self.cache.precompute_dataset_drugs([self.smiles_aspirin, self.smiles_caffeine, self.smiles_ibuprofen])

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_cancer_combo_batch
        )

        batches = list(loader)
        self.assertEqual(len(batches), 4)
        for b in batches:
            self.assertEqual(b["fp_A"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
