#!/usr/bin/env python3
"""
APACC Validation Framework - Main Runner
========================================
Paper: Quantitative Validation of Artificial Precognition Adaptive Cognised Control
Author: George Frangou
Institution: Cranfield University, School of Aerospace, Transport and Manufacturing
DOI: https://doi.org/10.5281/zenodo.8475

This script serves as the primary entry point for running APACC validation experiments
across multiple simulation environments (Monte Carlo, CARLA, SUMO, MATLAB).

Usage:
    python runner.py --scenario urban_8lane --episodes 10000 --gpu_monitor
    python runner.py --env carla --baseline mpc --parallel 8
    python runner.py --full_validation --output_format hdf5
"""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# GPU monitoring imports
try:
    import nvidia_ml_py as nvml
    import psutil
    import py3nvml.py3nvml as py3nvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPU monitoring libraries not available. Install nvidia-ml-py and py3nvml for GPU tracking.")

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing

# Custom APACC imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_monte_carlo import MonteCarloValidator
from simulation.carla_runner import CARLASimulation
from simulation.sumo_runner import SUMOSimulation
from simulation.matlab_bridge import MATLABValidator

# Data storage
import h5py
import zarr

# Profiling imports
try:
    from memory_profiler import profile
    from line_profiler import LineProfiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    warnings.warn("Profiling libraries not available. Install memory-profiler and line-profiler for profiling.")


class APACCRunner:
    """Main orchestrator for APACC validation experiments."""
    
    def __init__(self, config_path: str = "config/simulation.yaml"):
        """
        Initialize APACC Runner with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.results_dir = self._setup_results_directory()
        
        # Initialize GPU monitoring if available
        if GPU_AVAILABLE:
            nvml.nvmlInit()
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
        # Performance metrics
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_episodes': 0,
            'collision_count': 0,
            'avg_latency_ms': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'cpu_percent': []
        }
    
    def _load_config(self) -> Dict:
        """Load and validate configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required configuration sections
        required_sections = ['simulation', 'apacc', 'baselines', 'environments']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with both file and console output."""
        logger = logging.getLogger('APACC')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler (will be added after results directory is created)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_results_directory(self) -> Path:
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"results/{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler to logger
        log_file = results_dir / "apacc_validation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Save configuration to results directory
        config_copy = results_dir / "config.yaml"
        with open(config_copy, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info(f"Results directory created: {results_dir}")
        return results_dir
    
    def _monitor_gpu(self) -> Dict[str, float]:
        """Monitor GPU utilization and memory."""
        if not GPU_AVAILABLE:
            return {'gpu_util': 0.0, 'gpu_memory': 0.0, 'gpu_temp': 0.0}
        
        try:
            # GPU utilization
            util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            mem_used_gb = mem_info.used / 1024**3
            
            # Temperature
            temp = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
            
            return {
                'gpu_util': util.gpu,
                'gpu_memory': mem_used_gb,
                'gpu_temp': temp
            }
        except Exception as e:
            self.logger.warning(f"GPU monitoring error: {e}")
            return {'gpu_util': 0.0, 'gpu_memory': 0.0, 'gpu_temp': 0.0}
    
    def _monitor_system(self) -> Dict[str, float]:
        """Monitor system CPU and memory usage."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_gb': psutil.virtual_memory().used / 1024**3
        }
    
    def run_monte_carlo(self, scenarios: int, parallel: int = 1) -> pd.DataFrame:
        """
        Run Monte Carlo validation scenarios.
        
        Args:
            scenarios: Number of scenarios to run
            parallel: Number of parallel workers
            
        Returns:
            DataFrame with scenario results
        """
        self.logger.info(f"Starting Monte Carlo validation: {scenarios} scenarios")
        
        validator = MonteCarloValidator(self.config['simulation'])
        
        if parallel > 1:
            # Parallel execution using joblib
            scenario_batches = np.array_split(range(scenarios), parallel)
            
            def run_batch(batch_indices):
                results = []
                for i in batch_indices:
                    result = validator.run_scenario(i)
                    results.append(result)
                return results
            
            with tqdm(total=scenarios, desc="Monte Carlo Scenarios") as pbar:
                all_results = Parallel(n_jobs=parallel)(
                    delayed(run_batch)(batch) for batch in scenario_batches
                )
                pbar.update(scenarios)
            
            # Flatten results
            results = [r for batch in all_results for r in batch]
        else:
            # Sequential execution
            results = []
            for i in tqdm(range(scenarios), desc="Monte Carlo Scenarios"):
                result = validator.run_scenario(i)
                results.append(result)
                
                # Monitor system periodically
                if i % 100 == 0:
                    self.metrics['gpu_utilization'].append(self._monitor_gpu())
                    self.metrics['cpu_percent'].append(self._monitor_system()['cpu_percent'])
        
        return pd.DataFrame(results)
    
    def run_carla(self, episodes: int, scenario: str = "urban_8lane") -> pd.DataFrame:
        """
        Run CARLA simulation episodes.
        
        Args:
            episodes: Number of episodes to run
            scenario: Scenario name from config
            
        Returns:
            DataFrame with episode results
        """
        self.logger.info(f"Starting CARLA validation: {episodes} episodes, scenario: {scenario}")
        
        carla_config = self.config['environments']['carla']
        carla_config['scenario'] = scenario
        
        simulation = CARLASimulation(carla_config)
        results = []
        
        try:
            simulation.connect()
            
            for episode in tqdm(range(episodes), desc="CARLA Episodes"):
                # Run episode
                episode_result = simulation.run_episode()
                episode_result['episode'] = episode
                results.append(episode_result)
                
                # Update metrics
                self.metrics['total_episodes'] += 1
                if episode_result.get('collision', False):
                    self.metrics['collision_count'] += 1
                self.metrics['avg_latency_ms'].append(episode_result.get('avg_latency', 0))
                
                # Monitor resources
                if episode % 10 == 0:
                    gpu_stats = self._monitor_gpu()
                    self.metrics['gpu_utilization'].append(gpu_stats['gpu_util'])
                    self.logger.debug(f"Episode {episode}: GPU {gpu_stats['gpu_util']:.1f}%, "
                                    f"Temp {gpu_stats['gpu_temp']:.1f}°C")
        finally:
            simulation.cleanup()
        
        return pd.DataFrame(results)
    
    def run_sumo(self, duration: int, traffic_density: str = "medium") -> pd.DataFrame:
        """
        Run SUMO traffic simulation.
        
        Args:
            duration: Simulation duration in seconds
            traffic_density: Traffic density level
            
        Returns:
            DataFrame with simulation results
        """
        self.logger.info(f"Starting SUMO validation: {duration}s, density: {traffic_density}")
        
        sumo_config = self.config['environments']['sumo']
        sumo_config['traffic_density'] = traffic_density
        
        simulation = SUMOSimulation(sumo_config)
        return simulation.run(duration)
    
    def run_matlab(self, test_cases: List[str]) -> pd.DataFrame:
        """
        Run MATLAB mathematical validation.
        
        Args:
            test_cases: List of test case names
            
        Returns:
            DataFrame with validation results
        """
        self.logger.info(f"Starting MATLAB validation: {len(test_cases)} test cases")
        
        matlab_config = self.config['environments']['matlab']
        validator = MATLABValidator(matlab_config)
        
        results = []
        for test_case in test_cases:
            result = validator.validate(test_case)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def run_baseline_comparison(self, environment: str, episodes: int) -> pd.DataFrame:
        """
        Run baseline controller comparisons.
        
        Args:
            environment: Environment to use (carla, sumo, etc.)
            episodes: Number of episodes per baseline
            
        Returns:
            DataFrame comparing all controllers
        """
        self.logger.info(f"Running baseline comparison in {environment}")
        
        baselines = self.config['baselines']
        results = []
        
        for baseline_name, baseline_config in baselines.items():
            self.logger.info(f"Testing baseline: {baseline_name}")
            
            # Run episodes for this baseline
            if environment == "carla":
                baseline_results = self.run_carla(episodes, baseline_config.get('scenario', 'urban_8lane'))
            elif environment == "monte_carlo":
                baseline_results = self.run_monte_carlo(episodes)
            else:
                raise ValueError(f"Unsupported environment for baseline comparison: {environment}")
            
            baseline_results['controller'] = baseline_name
            results.append(baseline_results)
        
        return pd.concat(results, ignore_index=True)
    
    def save_results(self, results: pd.DataFrame, name: str, format: str = "hdf5"):
        """
        Save results in specified format.
        
        Args:
            results: DataFrame with results
            name: Base name for output file
            format: Output format (hdf5, zarr, csv)
        """
        output_path = self.results_dir / f"{name}.{format}"
        
        if format == "hdf5":
            results.to_hdf(output_path, key='results', mode='w', complevel=9)
            self.logger.info(f"Results saved to HDF5: {output_path}")
        elif format == "zarr":
            # Convert to zarr array
            zarr_store = zarr.open(str(output_path), mode='w')
            zarr_store.create_dataset('results', data=results.values, compression='gzip')
            zarr_store.attrs['columns'] = list(results.columns)
            self.logger.info(f"Results saved to Zarr: {output_path}")
        elif format == "csv":
            results.to_csv(output_path, index=False)
            self.logger.info(f"Results saved to CSV: {output_path}")
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def generate_summary_report(self):
        """Generate summary report of validation run."""
        summary = {
            'start_time': self.metrics['start_time'],
            'end_time': self.metrics['end_time'],
            'duration_hours': (self.metrics['end_time'] - self.metrics['start_time']).total_seconds() / 3600,
            'total_episodes': self.metrics['total_episodes'],
            'collision_rate': self.metrics['collision_count'] / max(self.metrics['total_episodes'], 1),
            'avg_latency_ms': np.mean(self.metrics['avg_latency_ms']) if self.metrics['avg_latency_ms'] else 0,
            'p99_latency_ms': np.percentile(self.metrics['avg_latency_ms'], 99) if self.metrics['avg_latency_ms'] else 0,
            'avg_gpu_utilization': np.mean(self.metrics['gpu_utilization']) if self.metrics['gpu_utilization'] else 0,
            'max_gpu_utilization': np.max(self.metrics['gpu_utilization']) if self.metrics['gpu_utilization'] else 0,
            'avg_cpu_percent': np.mean(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0
        }
        
        # Save summary
        summary_path = self.results_dir / "validation_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)
        
        return summary
    
    def run_full_validation(self):
        """Run complete validation suite as described in the paper."""
        self.logger.info("Starting full APACC validation suite")
        self.metrics['start_time'] = datetime.now()
        
        all_results = {}
        
        # 1. Monte Carlo (10,000 scenarios)
        self.logger.info("Phase 1: Monte Carlo Statistical Validation")
        mc_results = self.run_monte_carlo(
            scenarios=self.config['simulation']['monte_carlo']['scenarios'],
            parallel=self.config['simulation']['monte_carlo'].get('parallel_workers', 8)
        )
        all_results['monte_carlo'] = mc_results
        self.save_results(mc_results, 'monte_carlo_results', format='hdf5')
        
        # 2. CARLA (10,000 episodes)
        if self.config['environments']['carla'].get('enabled', True):
            self.logger.info("Phase 2: CARLA High-Fidelity Physics")
            carla_results = self.run_carla(
                episodes=self.config['simulation']['carla']['episodes'],
                scenario=self.config['simulation']['carla']['default_scenario']
            )
            all_results['carla'] = carla_results
            self.save_results(carla_results, 'carla_results', format='hdf5')
        
        # 3. SUMO (2,000 scenarios)
        if self.config['environments']['sumo'].get('enabled', True):
            self.logger.info("Phase 3: SUMO Large-Scale Traffic")
            sumo_results = self.run_sumo(
                duration=self.config['simulation']['sumo']['duration'],
                traffic_density=self.config['simulation']['sumo']['traffic_density']
            )
            all_results['sumo'] = sumo_results
            self.save_results(sumo_results, 'sumo_results', format='hdf5')
        
        # 4. MATLAB (2,500 scenarios)
        if self.config['environments']['matlab'].get('enabled', True):
            self.logger.info("Phase 4: MATLAB Mathematical Verification")
            matlab_results = self.run_matlab(
                test_cases=self.config['simulation']['matlab']['test_cases']
            )
            all_results['matlab'] = matlab_results
            self.save_results(matlab_results, 'matlab_results', format='hdf5')
        
        # 5. Baseline Comparisons
        self.logger.info("Phase 5: Baseline Controller Comparisons")
        baseline_results = self.run_baseline_comparison(
            environment=self.config['simulation']['baseline_comparison']['environment'],
            episodes=self.config['simulation']['baseline_comparison']['episodes_per_controller']
        )
        all_results['baselines'] = baseline_results
        self.save_results(baseline_results, 'baseline_comparison', format='hdf5')
        
        self.metrics['end_time'] = datetime.now()
        
        # Generate final report
        summary = self.generate_summary_report()
        
        return all_results, summary


def main():
    """Main entry point for APACC validation runner."""
    parser = argparse.ArgumentParser(
        description="APACC Validation Framework Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific scenario with GPU monitoring
  python runner.py --scenario urban_8lane --episodes 10000 --gpu_monitor
  
  # Run with specific environment and baseline
  python runner.py --env carla --baseline mpc --parallel 8
  
  # Run full validation suite
  python runner.py --full_validation --output_format hdf5
  
  # Run with custom config
  python runner.py --config config/custom_simulation.yaml --episodes 1000
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/simulation.yaml',
                      help='Path to configuration YAML file')
    
    # Execution modes
    parser.add_argument('--full_validation', action='store_true',
                      help='Run complete validation suite (24,500 scenarios)')
    parser.add_argument('--env', type=str, choices=['monte_carlo', 'carla', 'sumo', 'matlab'],
                      help='Specific environment to run')
    parser.add_argument('--scenario', type=str,
                      help='Scenario name (e.g., urban_8lane, highway_merge)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to run')
    
    # Baseline comparison
    parser.add_argument('--baseline', type=str, choices=['pid', 'mpc', 'drl_sac', 'drl_ppo'],
                      help='Baseline controller to test against APACC')
    parser.add_argument('--compare_all', action='store_true',
                      help='Compare APACC against all baselines')
    
    # Performance options
    parser.add_argument('--parallel', type=int, default=1,
                      help='Number of parallel workers (default: 1)')
    parser.add_argument('--gpu_monitor', action='store_true',
                      help='Enable GPU monitoring during execution')
    parser.add_argument('--profile', action='store_true',
                      help='Enable performance profiling')
    
    # Output options
    parser.add_argument('--output_format', type=str, choices=['hdf5', 'zarr', 'csv'],
                      default='hdf5', help='Output format for results')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                      help='Increase verbosity (use -vv for debug)')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize runner
    runner = APACCRunner(config_path=args.config)
    
    try:
        if args.full_validation:
            # Run complete validation suite
            results, summary = runner.run_full_validation()
            print(f"\nValidation complete! Results saved to: {runner.results_dir}")
            print(f"Total episodes: {summary['total_episodes']}")
            print(f"Collision rate: {summary['collision_rate']:.2%}")
            print(f"Average latency: {summary['avg_latency_ms']:.2f}ms")
            
        elif args.env:
            # Run specific environment
            if args.env == 'monte_carlo':
                results = runner.run_monte_carlo(args.episodes, args.parallel)
            elif args.env == 'carla':
                results = runner.run_carla(args.episodes, args.scenario or 'urban_8lane')
            elif args.env == 'sumo':
                results = runner.run_sumo(args.episodes)
            elif args.env == 'matlab':
                test_cases = runner.config['simulation']['matlab']['test_cases']
                results = runner.run_matlab(test_cases)
            
            # Save results
            runner.save_results(results, f"{args.env}_results", args.output_format)
            runner.generate_summary_report()
            
        elif args.compare_all or args.baseline:
            # Run baseline comparison
            results = runner.run_baseline_comparison('carla', args.episodes)
            runner.save_results(results, 'baseline_comparison', args.output_format)
            runner.generate_summary_report()
            
        else:
            # Default: run Monte Carlo with specified episodes
            results = runner.run_monte_carlo(args.episodes, args.parallel)
            runner.save_results(results, 'monte_carlo_results', args.output_format)
            runner.generate_summary_report()
            
    except KeyboardInterrupt:
        runner.logger.warning("Validation interrupted by user")
        runner.metrics['end_time'] = datetime.now()
        runner.generate_summary_report()
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup GPU monitoring
        if GPU_AVAILABLE:
            nvml.nvmlShutdown()


if __name__ == "__main__":
    main()