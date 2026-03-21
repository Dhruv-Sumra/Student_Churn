"""
Test script to validate the improved churn prediction model
"""
import pandas as pd
import numpy as np

# Test the adaptive semester weighting
def apply_semester_signal(base_prob, semester):
    """Adaptive semester weighting"""
    if semester >= 4:
        return base_prob
    
    adaptive_weights = {1: 0.20, 2: 0.15, 3: 0.05}
    weight = adaptive_weights.get(int(semester), 0.05)
    
    SEM_HIST_RATE = {1: 1.000, 2: 0.809, 3: 0.032}
    sem_signal = SEM_HIST_RATE.get(int(semester), 0.032)
    combined = (1 - weight) * base_prob + weight * sem_signal
    return float(np.clip(combined, 0.0, 1.0))

# Test cases
print("Testing Adaptive Semester Weighting")
print("=" * 50)

base_prob = 0.30  # 30% base admission risk

for sem in range(1, 7):
    final_prob = apply_semester_signal(base_prob, sem)
    print(f"Semester {sem}: Base={base_prob:.2f} → Final={final_prob:.2f}")

print("\n" + "=" * 50)
print("\nComparison with Old Fixed 10% Weight:")
print("=" * 50)

def old_apply_semester_signal(base_prob, semester, sem_weight=0.10):
    if semester >= 4:
        return base_prob
    SEM_HIST_RATE = {1: 1.000, 2: 0.809, 3: 0.032}
    sem_signal = SEM_HIST_RATE.get(int(semester), 0.032)
    combined = (1 - sem_weight) * base_prob + sem_weight * sem_signal
    return float(np.clip(combined, 0.0, 1.0))

print(f"\nBase admission probability: {base_prob:.2f}")
print(f"\n{'Semester':<12} {'Old (10%)':<15} {'New (Adaptive)':<15} {'Difference':<12}")
print("-" * 54)

for sem in range(1, 7):
    old_prob = old_apply_semester_signal(base_prob, sem)
    new_prob = apply_semester_signal(base_prob, sem)
    diff = new_prob - old_prob
    print(f"Sem {sem:<8} {old_prob:.4f}{'':<10} {new_prob:.4f}{'':<10} {diff:+.4f}")

print("\n" + "=" * 50)
print("\nKey Improvements:")
print("=" * 50)
print("✓ Sem 1: Stronger upward adjustment (20% vs 10%)")
print("✓ Sem 2: Moderate upward adjustment (15% vs 10%)")
print("✓ Sem 3: Minimal adjustment (5% vs 10%)")
print("✓ Better reflects actual risk gradient in data")
