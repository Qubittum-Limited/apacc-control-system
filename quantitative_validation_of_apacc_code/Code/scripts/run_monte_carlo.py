#!/usr/bin/env python3
"""Main script to run Monte Carlo simulations"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.carla_apacc_framework import MonteCarloSimulator
from simulation.importance_sampling_apacc import comprehensive_apacc_validation
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description='Run APACC Monte Carlo Simulations')
    parser.add_argument('--config', type=str, default='config/simulation.yaml',
                        help='Path to configuration file')
    parser.add_argument('--scenarios', type=int, default=10000,
                        help='Number of scenarios to run')
    parser.add_argument('--mode', choices=['carla', 'sumo', 'matlab', 'importance'],
                        default='carla', help='Simulation mode')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.mode == 'carla':
        simulator = MonteCarloSimulator()
        results = simulator.run_monte_carlo(
            num_scenarios=args.scenarios,
            parallel_runs=config.get('parallel_runs', 4)
        )
    elif args.mode == 'importance':
        results = comprehensive_apacc_validation()
    
    print(f"Simulation complete. Results saved.")

if __name__ == "__main__":
    main()