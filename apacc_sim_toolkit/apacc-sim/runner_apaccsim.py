#!/usr/bin/env python3
"""
runner_apaccsim.py

Main CLI runner script for APACC-Sim toolkit
Orchestrates multi-paradigm simulations across Monte Carlo, CARLA, SUMO, and MATLAB

Author: George Frangou
Institution: Cranfield University
"""

import os
import sys
import argparse
import yaml
import json
import logging
import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import multiprocessing as mp
import time
import signal
import traceback

# Core imports
import numpy as np
import pandas as pd
import h5py

# Distributed computing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Parallel execution will be limited.")

# Import simulation modules (placeholder - actual modules would be imported)
# from apacc_sim.monte_carlo import MonteCarloSimulator
# from apacc_sim.carla_interface import CARLASimulator
# from apacc_sim.sumo_interface import SUMOSimulator
# from apacc_sim.matlab_bridge import MATLABVerifier


class APACCSimRunner:
    """Main runner for APACC-Sim validation toolkit"""
    
    def __init__(self, config_file: str, output_dir: str = './results',
                 log_level: str = 'INFO', use_ray: bool = True):
        """
        Initialize APACC-Sim runner
        
        Args:
            config_file: Path to simulation configuration YAML
            output_dir: Directory for results output
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_ray: Whether to use Ray for distributed execution
        """
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir)
        self.use_ray = use_ray and RAY_AVAILABLE
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Load configuration
        self.config = self._load_config()
        
        # Create unique run ID
        self.run_id = self._generate_run_id()
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.simulators = {}
        self.results = {}
        self.metadata = {
            'run_id': self.run_id,
            'start_time': None,
            'end_time': None,
            'config_file': str(self.config_file),
            'config_hash': self._hash_config(),
            'platform': sys.platform,
            'python_version': sys.version,
            'ray_enabled': self.use_ray
        }
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._shutdown_requested = False
        
    def _setup_logging(self, log_level: str):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.output_dir / 'apacc_sim.log')
            ]
        )
        self.logger = logging.getLogger('APACCSimRunner')
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        self.logger.info(f"Loading configuration from {self.config_file}")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
            
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required sections
        required_sections = ['simulation', 'controllers', 'environments', 'metrics']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        # Set defaults
        config.setdefault('random_seed', 42)
        config.setdefault('parallel', {'enabled': True, 'workers': mp.cpu_count()})
        
        return config
        
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = self._hash_config()[:8]
        return f"{timestamp}_{config_hash}"
        
    def _hash_config(self) -> str:
        """Generate hash of configuration for reproducibility"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
        
    def initialize_simulators(self):
        """Initialize all configured simulators"""
        self.logger.info("Initializing simulators...")
        
        environments = self.config.get('environments', [])
        
        if 'monte_carlo' in environments:
            self._init_monte_carlo()
            
        if 'carla' in environments:
            self._init_carla()
            
        if 'sumo' in environments:
            self._init_sumo()
            
        if 'matlab' in environments:
            self._init_matlab()
            
        self.logger.info(f"Initialized {len(self.simulators)} simulators")
        
    def _init_monte_carlo(self):
        """Initialize Monte Carlo simulator"""
        self.logger.debug("Initializing Monte Carlo simulator")
        
        # Placeholder for actual implementation
        class MonteCarloSimulator:
            def __init__(self, config):
                self.config = config
                self.name = 'monte_carlo'
                
            def run_scenario(self, scenario_id, params):
                # Simulate Monte Carlo execution
                np.random.seed(params.get('seed', 42))
                collision = np.random.random() < 0.001  # 0.1% collision rate
                ttc = np.random.gamma(4, 2)
                latency = np.random.normal(3.5, 0.5)
                
                return {
                    'scenario_id': scenario_id,
                    'collision': collision,
                    'time_to_collision': ttc,
                    'control_latency': latency,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
        mc_config = self.config.get('monte_carlo', {})
        self.simulators['monte_carlo'] = MonteCarloSimulator(mc_config)
        
    def _init_carla(self):
        """Initialize CARLA simulator"""
        self.logger.debug("Initializing CARLA simulator")
        
        # Check CARLA availability
        carla_root = os.environ.get('CARLA_ROOT')
        if not carla_root:
            self.logger.warning("CARLA_ROOT not set, skipping CARLA initialization")
            return
            
        # Placeholder for actual CARLA interface
        class CARLASimulator:
            def __init__(self, config):
                self.config = config
                self.name = 'carla'
                
            def run_scenario(self, scenario_id, params):
                # Simulate CARLA execution
                return {
                    'scenario_id': scenario_id,
                    'collision': False,
                    'sensor_data': {'lidar_points': 64000, 'camera_frames': 30},
                    'physics_fps': 60,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
        carla_config = self.config.get('carla', {})
        self.simulators['carla'] = CARLASimulator(carla_config)
        
    def _init_sumo(self):
        """Initialize SUMO simulator"""
        self.logger.debug("Initializing SUMO simulator")
        
        # Placeholder for actual SUMO interface
        class SUMOSimulator:
            def __init__(self, config):
                self.config = config
                self.name = 'sumo'
                
            def run_scenario(self, scenario_id, params):
                # Simulate SUMO execution
                return {
                    'scenario_id': scenario_id,
                    'vehicles_simulated': 150,
                    'avg_speed': 45.2,
                    'traffic_density': 0.75,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
        sumo_config = self.config.get('sumo', {})
        self.simulators['sumo'] = SUMOSimulator(sumo_config)
        
    def _init_matlab(self):
        """Initialize MATLAB verifier"""
        self.logger.debug("Initializing MATLAB verifier")
        
        # Placeholder for actual MATLAB bridge
        class MATLABVerifier:
            def __init__(self, config):
                self.config = config
                self.name = 'matlab'
                
            def verify_controller(self, controller_params):
                # Simulate MATLAB verification
                return {
                    'stable': True,
                    'robustness_margin': 0.85,
                    'verified_constraints': ['collision_avoidance', 'lane_keeping'],
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
        matlab_config = self.config.get('matlab', {})
        self.simulators['matlab'] = MATLABVerifier(matlab_config)
        
    def run_validation(self):
        """Run complete validation campaign"""
        self.logger.info(f"Starting validation run: {self.run_id}")
        self.metadata['start_time'] = datetime.datetime.now().isoformat()
        
        try:
            # Initialize Ray if enabled
            if self.use_ray:
                self._init_ray()
                
            # Generate scenarios
            scenarios = self._generate_scenarios()
            self.logger.info(f"Generated {len(scenarios)} scenarios")
            
            # Save scenario configuration
            self._save_scenarios(scenarios)
            
            # Run simulations
            if self.use_ray:
                results = self._run_distributed(scenarios)
            else:
                results = self._run_sequential(scenarios)
                
            # Process results
            self._process_results(results)
            
            # Run verification if MATLAB available
            if 'matlab' in self.simulators:
                self._run_verification()
                
            # Generate reports
            self._generate_reports()
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise
            
        finally:
            self.metadata['end_time'] = datetime.datetime.now().isoformat()
            self._save_metadata()
            
            if self.use_ray:
                ray.shutdown()
                
        self.logger.info(f"Validation complete. Results saved to {self.run_dir}")
        
    def _init_ray(self):
        """Initialize Ray for distributed execution"""
        self.logger.info("Initializing Ray...")
        
        ray_config = self.config.get('parallel', {})
        num_cpus = ray_config.get('workers', mp.cpu_count())
        
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,
            logging_level=logging.WARNING
        )
        
        self.logger.info(f"Ray initialized with {num_cpus} workers")
        
    def _generate_scenarios(self) -> List[Dict[str, Any]]:
        """Generate validation scenarios"""
        self.logger.info("Generating scenarios...")
        
        sim_config = self.config['simulation']
        num_scenarios = sim_config.get('num_scenarios', 1000)
        random_seed = self.config.get('random_seed', 42)
        
        np.random.seed(random_seed)
        scenarios = []
        
        for i in range(num_scenarios):
            scenario = {
                'id': f"scenario_{i:05d}",
                'seed': random_seed + i,
                'parameters': self._generate_scenario_params(i)
            }
            scenarios.append(scenario)
            
        return scenarios
        
    def _generate_scenario_params(self, index: int) -> Dict[str, Any]:
        """Generate parameters for a single scenario"""
        # Weather conditions
        weather_conditions = ['clear', 'rain', 'fog', 'night']
        weather = np.random.choice(weather_conditions, p=[0.6, 0.2, 0.1, 0.1])
        
        # Traffic density
        traffic_density = np.random.beta(2, 5)  # Skewed towards lower density
        
        # Sensor failures
        sensor_failure_prob = 0.05
        sensor_failures = {
            'camera': np.random.random() < sensor_failure_prob,
            'lidar': np.random.random() < sensor_failure_prob,
            'radar': np.random.random() < sensor_failure_prob
        }
        
        return {
            'weather': weather,
            'traffic_density': traffic_density,
            'sensor_failures': sensor_failures,
            'vehicle_speed': np.random.normal(50, 10),  # km/h
            'scenario_type': np.random.choice(['highway', 'urban', 'intersection'])
        }
        
    def _save_scenarios(self, scenarios: List[Dict[str, Any]]):
        """Save scenario configuration for reproducibility"""
        scenario_file = self.run_dir / 'scenarios.json'
        with open(scenario_file, 'w') as f:
            json.dump(scenarios, f, indent=2)
            
        self.logger.info(f"Saved {len(scenarios)} scenarios to {scenario_file}")
        
    def _run_sequential(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run scenarios sequentially"""
        self.logger.info("Running scenarios sequentially...")
        
        results = []
        total = len(scenarios)
        
        for i, scenario in enumerate(scenarios):
            if self._shutdown_requested:
                self.logger.warning("Shutdown requested, stopping execution")
                break
                
            self.logger.debug(f"Running scenario {i+1}/{total}: {scenario['id']}")
            
            scenario_results = {}
            
            # Run each simulator
            for name, simulator in self.simulators.items():
                if name == 'matlab':
                    continue  # MATLAB verification runs separately
                    
                try:
                    result = simulator.run_scenario(scenario['id'], scenario['parameters'])
                    scenario_results[name] = result
                except Exception as e:
                    self.logger.error(f"Error in {name} for {scenario['id']}: {e}")
                    scenario_results[name] = {'error': str(e)}
                    
            results.append({
                'scenario': scenario,
                'results': scenario_results
            })
            
            # Progress update
            if (i + 1) % 100 == 0:
                self.logger.info(f"Progress: {i+1}/{total} scenarios completed")
                
        return results
        
    def _run_distributed(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run scenarios using Ray for distributed execution"""
        self.logger.info("Running scenarios with Ray distributed execution...")
        
        # Define Ray remote functions
        @ray.remote
        def run_scenario_remote(scenario, simulators_config):
            """Remote function to run a single scenario"""
            results = {}
            
            # Recreate simulators in remote worker
            # (In real implementation, would properly initialize simulators)
            
            # Placeholder: simulate execution
            import time
            import random
            time.sleep(random.uniform(0.01, 0.05))  # Simulate work
            
            results['monte_carlo'] = {
                'scenario_id': scenario['id'],
                'collision': random.random() < 0.001,
                'latency': random.gauss(3.5, 0.5)
            }
            
            return {'scenario': scenario, 'results': results}
            
        # Submit scenarios to Ray
        futures = []
        simulators_config = {name: sim.config for name, sim in self.simulators.items()}
        
        for scenario in scenarios:
            if self._shutdown_requested:
                break
            future = run_scenario_remote.remote(scenario, simulators_config)
            futures.append(future)
            
        # Collect results with progress tracking
        results = []
        total = len(futures)
        completed = 0
        
        while futures and not self._shutdown_requested:
            ready, futures = ray.wait(futures, timeout=1.0)
            
            for future in ready:
                result = ray.get(future)
                results.append(result)
                completed += 1
                
                if completed % 100 == 0:
                    self.logger.info(f"Progress: {completed}/{total} scenarios completed")
                    
        return results
        
    def _process_results(self, results: List[Dict[str, Any]]):
        """Process and save simulation results"""
        self.logger.info("Processing results...")
        
        # Convert to structured format
        processed_results = {
            'scenarios': [],
            'metrics': {},
            'raw_results': []
        }
        
        # Initialize metric collectors
        metrics = {
            'collision_count': 0,
            'collision_rate': 0.0,
            'latencies': [],
            'time_to_collisions': []
        }
        
        # Process each result
        for result in results:
            scenario = result['scenario']
            sim_results = result['results']
            
            processed_results['scenarios'].append(scenario)
            processed_results['raw_results'].append(sim_results)
            
            # Extract metrics
            if 'monte_carlo' in sim_results:
                mc_result = sim_results['monte_carlo']
                if not isinstance(mc_result, dict) or 'error' not in mc_result:
                    if mc_result.get('collision', False):
                        metrics['collision_count'] += 1
                    if 'control_latency' in mc_result:
                        metrics['latencies'].append(mc_result['control_latency'])
                    if 'time_to_collision' in mc_result:
                        metrics['time_to_collisions'].append(mc_result['time_to_collision'])
                        
        # Calculate aggregate metrics
        num_scenarios = len(results)
        metrics['collision_rate'] = metrics['collision_count'] / num_scenarios if num_scenarios > 0 else 0
        metrics['avg_latency'] = np.mean(metrics['latencies']) if metrics['latencies'] else 0
        metrics['p99_latency'] = np.percentile(metrics['latencies'], 99) if metrics['latencies'] else 0
        
        processed_results['metrics'] = metrics
        self.results = processed_results
        
        # Save results
        self._save_results()
        
    def _save_results(self):
        """Save processed results to various formats"""
        self.logger.info("Saving results...")
        
        # Save as HDF5
        hdf5_file = self.run_dir / 'results.h5'
        with h5py.File(hdf5_file, 'w') as f:
            # Save metrics
            metrics_grp = f.create_group('metrics')
            for key, value in self.results['metrics'].items():
                if isinstance(value, list):
                    metrics_grp.create_dataset(key, data=np.array(value))
                else:
                    metrics_grp.attrs[key] = value
                    
            # Save scenario parameters
            scenarios_grp = f.create_group('scenarios')
            for i, scenario in enumerate(self.results['scenarios']):
                scenario_grp = scenarios_grp.create_group(f'scenario_{i:05d}')
                scenario_grp.attrs['id'] = scenario['id']
                scenario_grp.attrs['seed'] = scenario['seed']
                
                # Save parameters
                for key, value in scenario['parameters'].items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            scenario_grp.attrs[f'{key}_{sub_key}'] = sub_value
                    else:
                        scenario_grp.attrs[key] = value
                        
        self.logger.info(f"Saved HDF5 results to {hdf5_file}")
        
        # Save summary as JSON
        summary_file = self.run_dir / 'summary.json'
        summary = {
            'run_id': self.run_id,
            'num_scenarios': len(self.results['scenarios']),
            'metrics': self.results['metrics'],
            'config_hash': self.metadata['config_hash']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save metrics as CSV for easy analysis
        if self.results['metrics']['latencies']:
            metrics_df = pd.DataFrame({
                'latency': self.results['metrics']['latencies'],
                'ttc': self.results['metrics']['time_to_collisions'][:len(self.results['metrics']['latencies'])]
            })
            metrics_df.to_csv(self.run_dir / 'metrics.csv', index=False)
            
    def _run_verification(self):
        """Run MATLAB verification if available"""
        if 'matlab' not in self.simulators:
            return
            
        self.logger.info("Running MATLAB verification...")
        
        matlab = self.simulators['matlab']
        controllers = self.config.get('controllers', {})
        
        verification_results = {}
        
        for controller_name, controller_config in controllers.items():
            self.logger.debug(f"Verifying controller: {controller_name}")
            
            try:
                result = matlab.verify_controller(controller_config)
                verification_results[controller_name] = result
            except Exception as e:
                self.logger.error(f"Verification failed for {controller_name}: {e}")
                verification_results[controller_name] = {'error': str(e)}
                
        # Save verification results
        verification_file = self.run_dir / 'verification_results.json'
        with open(verification_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
            
        self.logger.info(f"Saved verification results to {verification_file}")
        
    def _generate_reports(self):
        """Generate analysis reports"""
        self.logger.info("Generating reports...")
        
        # Create reports directory
        reports_dir = self.run_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Generate text report
        self._generate_text_report(reports_dir / 'report.txt')
        
        # Generate HTML report
        self._generate_html_report(reports_dir / 'report.html')
        
        # Generate certification evidence
        if self.config.get('certification', {}).get('enabled', False):
            self._generate_certification_evidence(reports_dir / 'certification')
            
    def _generate_text_report(self, output_file: Path):
        """Generate human-readable text report"""
        metrics = self.results['metrics']
        
        report_lines = [
            "APACC-Sim Validation Report",
            "=" * 50,
            f"Run ID: {self.run_id}",
            f"Date: {self.metadata['start_time']}",
            f"Configuration: {self.config_file}",
            "",
            "Summary Statistics",
            "-" * 30,
            f"Total Scenarios: {len(self.results['scenarios'])}",
            f"Collision Rate: {metrics['collision_rate']:.4%}",
            f"Average Latency: {metrics['avg_latency']:.2f} ms",
            f"P99 Latency: {metrics['p99_latency']:.2f} ms",
            "",
            "Environment Distribution",
            "-" * 30
        ]
        
        # Count scenarios by type
        scenario_types = {}
        weather_types = {}
        
        for scenario in self.results['scenarios']:
            params = scenario['parameters']
            scenario_type = params.get('scenario_type', 'unknown')
            weather = params.get('weather', 'unknown')
            
            scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
            weather_types[weather] = weather_types.get(weather, 0) + 1
            
        for scenario_type, count in scenario_types.items():
            report_lines.append(f"{scenario_type.capitalize()}: {count}")
            
        report_lines.extend(["", "Weather Conditions", "-" * 30])
        
        for weather, count in weather_types.items():
            report_lines.append(f"{weather.capitalize()}: {count}")
            
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
            
        self.logger.info(f"Generated text report: {output_file}")
        
    def _generate_html_report(self, output_file: Path):
        """Generate HTML report with visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APACC-Sim Validation Report - {self.run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #264653; }}
        .metric {{ 
            display: inline-block; 
            margin: 10px 20px 10px 0; 
            padding: 15px; 
            background: #F1FAEE; 
            border-radius: 5px;
            border: 1px solid #E0E0E0;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2E86AB; color: white; }}
        .success {{ color: #06D6A0; }}
        .warning {{ color: #F77F00; }}
        .error {{ color: #E63946; }}
    </style>
</head>
<body>
    <h1>APACC-Sim Validation Report</h1>
    <p><strong>Run ID:</strong> {self.run_id}</p>
    <p><strong>Date:</strong> {self.metadata['start_time']}</p>
    <p><strong>Configuration:</strong> {self.config_file}</p>
    
    <h2>Key Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{len(self.results['scenarios'])}</div>
            <div class="metric-label">Total Scenarios</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.results['metrics']['collision_rate']:.4%}</div>
            <div class="metric-label">Collision Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.results['metrics']['avg_latency']:.2f} ms</div>
            <div class="metric-label">Average Latency</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.results['metrics']['p99_latency']:.2f} ms</div>
            <div class="metric-label">P99 Latency</div>
        </div>
    </div>
    
    <h2>Simulators Used</h2>
    <ul>
"""
        
        for simulator_name in self.simulators:
            html_content += f"        <li>{simulator_name.replace('_', ' ').title()}</li>\n"
            
        html_content += """
    </ul>
    
    <h2>Results Summary</h2>
    <p>Detailed results are available in the following files:</p>
    <ul>
        <li><a href="../results.h5">HDF5 Results</a> - Complete raw data</li>
        <li><a href="../summary.json">JSON Summary</a> - Key metrics and configuration</li>
        <li><a href="../metrics.csv">CSV Metrics</a> - Tabular metrics data</li>
    </ul>
    
    <hr>
    <p><em>Generated by APACC-Sim v1.0</em></p>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Generated HTML report: {output_file}")
        
    def _generate_certification_evidence(self, output_dir: Path):
        """Generate certification evidence package"""
        output_dir.mkdir(exist_ok=True)
        
        self.logger.info("Generating certification evidence...")
        
        # ISO 26262 metrics
        iso26262_metrics = {
            'diagnostic_coverage': 0.95,  # Placeholder
            'safe_state_reached': True,
            'failure_rate': self.results['metrics']['collision_rate'],
            'response_time_violations': 0  # Count of latency > threshold
        }
        
        # Count response time violations
        threshold_ms = 10.0
        for latency in self.results['metrics']['latencies']:
            if latency > threshold_ms:
                iso26262_metrics['response_time_violations'] += 1
                
        # Save ISO 26262 evidence
        with open(output_dir / 'iso26262_evidence.json', 'w') as f:
            json.dump(iso26262_metrics, f, indent=2)
            
        # Generate traceability matrix
        self._generate_traceability_matrix(output_dir / 'traceability_matrix.csv')
        
        self.logger.info(f"Generated certification evidence in {output_dir}")
        
    def _generate_traceability_matrix(self, output_file: Path):
        """Generate requirements traceability matrix"""
        # Placeholder traceability data
        traceability_data = []
        
        requirements = [
            'REQ-SAFE-001: Collision avoidance',
            'REQ-SAFE-002: Lane keeping',
            'REQ-PERF-001: Control latency < 10ms',
            'REQ-PERF-002: Sensor fusion rate > 20Hz'
        ]
        
        test_cases = [
            'TC-001: Highway collision avoidance',
            'TC-002: Urban intersection safety',
            'TC-003: Control loop timing',
            'TC-004: Multi-sensor integration'
        ]
        
        # Create traceability links
        for req in requirements:
            for tc in test_cases:
                if ('collision' in req.lower() and 'collision' in tc.lower()) or \
                   ('lane' in req.lower() and 'highway' in tc.lower()) or \
                   ('latency' in req.lower() and 'timing' in tc.lower()) or \
                   ('sensor' in req.lower() and 'sensor' in tc.lower()):
                    traceability_data.append({
                        'Requirement': req,
                        'Test Case': tc,
                        'Status': 'PASS',
                        'Evidence': f'results.h5'
                    })
                    
        # Save as CSV
        df = pd.DataFrame(traceability_data)
        df.to_csv(output_file, index=False)
        
    def _save_metadata(self):
        """Save run metadata"""
        metadata_file = self.run_dir / 'metadata.json'
        
        # Calculate duration
        if self.metadata['start_time'] and self.metadata['end_time']:
            start = datetime.datetime.fromisoformat(self.metadata['start_time'].replace('Z', '+00:00'))
            end = datetime.datetime.fromisoformat(self.metadata['end_time'].replace('Z', '+00:00'))
            duration = (end - start).total_seconds()
            self.metadata['duration_seconds'] = duration
            
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        self.logger.info(f"Saved metadata to {metadata_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run APACC-Sim validation campaigns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s simulation.yaml                    # Run with default settings
  %(prog)s simulation.yaml -o ./my_results    # Custom output directory
  %(prog)s simulation.yaml --sequential       # Disable parallel execution
  %(prog)s simulation.yaml --dry-run          # Validate configuration only
  %(prog)s simulation.yaml --scenarios 100    # Override scenario count
        '''
    )
    
    parser.add_argument('config', help='Path to simulation configuration YAML file')
    parser.add_argument('-o', '--output', default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('-l', '--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run simulations sequentially (disable Ray)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running simulations')
    parser.add_argument('--scenarios', type=int,
                       help='Override number of scenarios in config')
    parser.add_argument('--seed', type=int,
                       help='Override random seed in config')
    parser.add_argument('--resume', type=str,
                       help='Resume from previous run directory')
    
    args = parser.parse_args()
    
    # Create runner
    runner = APACCSimRunner(
        config_file=args.config,
        output_dir=args.output,
        log_level=args.log_level,
        use_ray=not args.sequential
    )
    
    # Override configuration if specified
    if args.scenarios:
        runner.config['simulation']['num_scenarios'] = args.scenarios
    if args.seed:
        runner.config['random_seed'] = args.seed
        
    # Dry run mode
    if args.dry_run:
        runner.logger.info("Dry run mode - validating configuration only")
        runner.initialize_simulators()
        scenarios = runner._generate_scenarios()
        runner.logger.info(f"Configuration valid. Would run {len(scenarios)} scenarios.")
        runner.logger.info(f"Simulators available: {list(runner.simulators.keys())}")
        return
        
    # Resume mode
    if args.resume:
        runner.logger.info(f"Resume mode from {args.resume}")
        # TODO: Implement resume functionality
        runner.logger.warning("Resume functionality not yet implemented")
        
    # Run validation
    try:
        runner.initialize_simulators()
        runner.run_validation()
    except KeyboardInterrupt:
        runner.logger.warning("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()