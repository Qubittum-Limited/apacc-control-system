#!/usr/bin/env python3
"""Analyze Monte Carlo simulation results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_results(results_path):
    # Load results
    df = pd.read_csv(results_path)
    
    # Calculate statistics
    stats = {
        'total_scenarios': len(df['scenario_id'].unique()),
        'collision_rate': df['collision_occurred'].mean(),
        'mean_control_latency': df['control_loop_latency_ms'].mean(),
        'p95_control_latency': df['control_loop_latency_ms'].quantile(0.95),
        'safety_violation_rate': (df['safety_violations'] > 0).mean(),
    }
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Control latency histogram
    axes[0, 0].hist(df['control_loop_latency_ms'], bins=50, edgecolor='black')
    axes[0, 0].axvline(10, color='r', linestyle='--', label='10ms deadline')
    axes[0, 0].set_xlabel('Control Latency (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # TTC distribution
    axes[0, 1].hist(df['ttc_min'], bins=50, edgecolor='black')
    axes[0, 1].axvline(2.0, color='r', linestyle='--', label='Safety threshold')
    axes[0, 1].set_xlabel('Minimum TTC (s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Save results
    with open('analysis_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_analysis.png', dpi=300)
    
    return stats

if __name__ == "__main__":
    import sys
    results_path = sys.argv[1] if len(sys.argv) > 1 else 'results/monte_carlo_results.csv'
    stats = analyze_results(results_path)
    print(json.dumps(stats, indent=2))