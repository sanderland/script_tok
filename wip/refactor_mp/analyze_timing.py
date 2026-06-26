# Compare 1 worker vs 8 workers timing

print("=" * 80)
print("TIMING COMPARISON: 1 Worker vs 8 Workers")
print("=" * 80)

# Worker startup
print("\n1. WORKER STARTUP:")
print("  1 worker:  24.14s total, 0.76s max load, 0.75s avg load")
print("  8 workers: 23.81s total, 0.86s max load, 0.77s avg load")
print("  → Similar startup time (good parallelism)")

# E-step timings (iteration 1.1)
print("\n2. E-STEP (Iteration 1.1, vocab size 165K):")
print("  1 worker:  total=15.01s, max_worker=13.14s, lattice=12.46s")
print("  8 workers: total=16.34s, max_worker=13.66s, lattice=12.80s")
print("  → 8 workers SLOWER! (16.34s vs 15.01s)")
print("  → Workers doing similar work time (~13s each)")

# E-step iteration 1.2
print("\n3. E-STEP (Iteration 1.2, vocab size 137K):")
print("  1 worker:  total=21.04s, max_worker=18.22s, lattice=17.35s")
print("  8 workers: total=22.21s, max_worker=18.89s, lattice=17.86s")
print("  → 8 workers SLOWER again (22.21s vs 21.04s)")

# Prune step
print("\n4. PRUNE (After iteration 1):")
print("  1 worker:  total=9.79s, max_worker=5.37s, viterbi=4.46s, loss_calc=2.06s")
print("  8 workers: total=10.92s, max_worker=5.78s, viterbi=4.66s, loss_calc=2.21s")
print("  → 8 workers SLOWER (10.92s vs 9.79s)")

# E-step iteration 2.1
print("\n5. E-STEP (Iteration 2.1, vocab size 85K):")
print("  1 worker:  total=18.46s, max_worker=16.61s, lattice=15.83s")
print("  8 workers: total=19.42s, max_worker=17.20s, lattice=16.35s")
print("  → 8 workers SLOWER (19.42s vs 18.46s)")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print("1. Max worker time ≈ 1-worker total time")
print("   → Workers ARE working in parallel (not queued)")
print("")
print("2. 8 workers consistently SLOWER than 1 worker")
print("   → NOT benefiting from parallelism at all")
print("   → Actually SLOWER due to overhead")
print("")
print("3. collect_time ≈ max_worker_time")
print("   → No queue contention")
print("   → Problem is in the worker compute itself")
print("")
print("4. The issue: Single worker processes ALL data in 13-18s")
print("   8 workers should process 1/8th of data in 1.6-2.3s each")
print("   But they take the SAME 13-18s!")
print("   → WORKERS ARE NOT PARTITIONING THE CORPUS!")
print("=" * 80)
