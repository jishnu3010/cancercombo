"""
Benchmark Script for BRICSCache DataLoader Performance.

Compares active CancerCombo DataLoader throughput:
    BEFORE: Dynamic RDKit BRICS decomposition + Morgan FP on every sample (use_brics_cache=False)
    AFTER : Precomputed & cached SMILES lookups via BRICSCache (use_brics_cache=True)

Measures:
    - Precomputation / Dataset loading time
    - Batch generation time (N batches)
    - Batches per second & Samples per second
    - BRICSCache instrumentation statistics (hits, misses, hit rate)
"""

import time
import torch
from torch.utils.data import DataLoader

import config
from cancer_combo_brics import (
    load_cancer_combo_from_csv,
    collate_cancer_combo_batch
)
from cancer_combo_brics.brics_cache import BRICSCache, get_global_brics_cache, reset_global_brics_cache


def benchmark_dataloader(use_cache: bool, max_samples: int = 2000, num_batches: int = 15, num_workers: int = 2):
    reset_global_brics_cache()
    mode_str = "WITH BRICSCache (AFTER)" if use_cache else "WITHOUT Cache / Dynamic RDKit (BEFORE)"
    print("\n" + "=" * 70)
    print(f"  Benchmark: {mode_str}")
    print("=" * 70)

    # 1. Dataset Loading & Precomputation
    t0 = time.time()
    brics_cache = BRICSCache(cache_file=None, n_bits=config.FRAG_FP_DIM) if use_cache else None

    dataset = load_cancer_combo_from_csv(
        config.DATA_CSV,
        split=config.TRAIN_SPLIT,
        max_samples=max_samples,
        brics_cache=brics_cache,
        use_brics_cache=use_cache
    )
    load_time = time.time() - t0
    print(f"Dataset load & precompute time: {load_time:.3f} seconds ({len(dataset)} samples)")

    if brics_cache:
        brics_cache.reset_stats()

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cancer_combo_batch
    )

    # 2. Batch Generation Loop
    t_start = time.time()
    samples_processed = 0
    batches_processed = 0

    for i, batch in enumerate(loader):
        if batches_processed >= num_batches:
            break
        samples_processed += batch["cell_expr"].size(0)
        batches_processed += 1

    t_end = time.time()
    batch_time = t_end - t_start
    batches_per_sec = batches_processed / max(batch_time, 1e-6)
    samples_per_sec = samples_processed / max(batch_time, 1e-6)

    print(f"Processed {batches_processed} batches ({samples_processed} samples) in {batch_time:.3f} seconds.")
    print(f"Throughput: {batches_per_sec:.2f} batches/sec | {samples_per_sec:.2f} samples/sec")

    if brics_cache:
        brics_cache.print_stats()

    return {
        "mode": mode_str,
        "load_time": load_time,
        "batch_time": batch_time,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "stats": brics_cache.get_stats() if brics_cache else None
    }


def main():
    print("=" * 70)
    print("  CancerCombo BRICSCache Performance Benchmark")
    print("=" * 70)

    # Benchmark BEFORE (Uncached)
    res_before = benchmark_dataloader(use_cache=False, max_samples=2000, num_batches=15, num_workers=0)

    # Benchmark AFTER (Cached)
    res_after = benchmark_dataloader(use_cache=True, max_samples=2000, num_batches=15, num_workers=0)

    # Comparison Summary
    speedup_batch = res_after["batches_per_sec"] / max(res_before["batches_per_sec"], 1e-6)
    time_reduction = (1.0 - (res_after["batch_time"] / max(res_before["batch_time"], 1e-6))) * 100.0

    print("\n" + "=" * 70)
    print("         BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"BEFORE (Dynamic RDKit): {res_before['batches_per_sec']:.2f} batches/sec ({res_before['batch_time']:.3f}s)")
    print(f"AFTER  (BRICSCache)   : {res_after['batches_per_sec']:.2f} batches/sec ({res_after['batch_time']:.3f}s)")
    print(f"Batch Processing Speedup: {speedup_batch:.2f}x faster")
    print(f"Batch Processing Time Reduction: {time_reduction:.1f}%")
    if res_after["stats"]:
        print(f"Cache Hit Rate: {res_after['stats']['hit_rate']:.2f}% ({res_after['stats']['hits']} hits / {res_after['stats']['requests']} requests)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
