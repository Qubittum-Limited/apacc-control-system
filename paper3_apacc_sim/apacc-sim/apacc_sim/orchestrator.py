"""
Simulation Orchestrator Module

Central coordination of multi-paradigm validation campaigns,
managing experiment flow across Monte Carlo, CARLA, SUMO, and MATLAB.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime

from modules.monte_carlo import MonteCarloSimulator
from modules.carla_integration import CarlaSimulator
from modules.sumo_wrapper import SumoSimulator
from modules.matlab_bridge import MatlabVerifier

from .metrics import MetricsCollector
from .explainability import ExplainabilityTracker

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for multi-paradigm validation scenario"""
    name: str = "default_validation"
    monte_carlo_runs: int = 1000
    carla_scenarios: List[str] = field(default_factory=lambda: ["urban_day"])
    sumo_traffic_density: str = "medium"
    matlab_validation: bool = True
    output_format: str = "parquet"
    parallel_workers: int = -1
    checkpoint_interval: int = 100
    

class SimulationOrchestrator:
    """
    Central orchestrator for multi-paradigm validation campaigns
    
    Coordinates execution across different simulation environments
    and aggregates results for comprehensive validation.
    """
    
    def __init__(self, config: Union[str, ScenarioConfig, Dict]):
        """
        Initialize orchestrator with configuration
        
        Args:
            config: Path to config file, ScenarioConfig object, or dict
        """
        if isinstance(config, str):
            self.config = self._load_config_file(config)
        elif isinstance(config, ScenarioConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = ScenarioConfig(**config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
            
        self.results_dir = Path("results") / self.config.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.explainability_tracker = ExplainabilityTracker()
        
        # Simulator instances (lazy loading)
        self._monte_carlo = None
        self._carla = None
        self._sumo = None
        self._matlab = None
        
    def _load_config_file(self, config_path: str) -> ScenarioConfig:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
                
        return ScenarioConfig(**config_dict)
        
    @property
    def monte_carlo(self) -> MonteCarloSimulator:
        """Lazy load Monte Carlo simulator"""
        if self._monte_carlo is None:
            config_path = Path("configs/monte_carlo/default.yaml")
            self._monte_carlo = MonteCarloSimulator(str(config_path))
        return self._monte_carlo
        
    @property
    def carla(self) -> CarlaSimulator:
        """Lazy load CARLA simulator"""
        if self._carla is None:
            config_path = Path("configs/carla/default.yaml")
            self._carla = CarlaSimulator(str(config_path))
            self._carla.connect()
        return self._carla
        
    @property
    def sumo(self) -> SumoSimulator:
        """Lazy load SUMO simulator"""
        if self._sumo is None:
            config_path = Path("configs/sumo/default.yaml")
            self._sumo = SumoSimulator(str(config_path))
        return self._sumo
        
    @property
    def matlab(self) -> MatlabVerifier:
        """Lazy load MATLAB verifier"""
        if self._matlab is None:
            config_path = Path("configs/matlab/verification_config.yaml")
            self._matlab = MatlabVerifier(str(config_path))
            self._matlab.start_engine()
        return self._matlab
        
    def run_validation_suite(self, controller, 
                           metrics: Optional[List[str]] = None,
                           save_raw_data: bool = False) -> pd.DataFrame:
        """
        Run complete validation suite across all simulators
        
        Args:
            controller: Controller object or function to validate
            metrics: List of metrics to collect (None for all)
            save_raw_data: Whether to save raw simulation data
            
        Returns:
            DataFrame with aggregated results
        """
        logger.info(f"Starting validation suite: {self.config.name}")
        
        all_results = []
        
        # Monte Carlo validation
        if self.config.monte_carlo_runs > 0:
            logger.info(f"Running Monte Carlo validation ({self.config.monte_carlo_runs} runs)...")
            mc_results = self._run_monte_carlo(controller)
            all_results.append(('monte_carlo', mc_results))
            
        # CARLA validation
        if self.config.carla_scenarios:
            logger.info(f"Running CARLA validation ({len(self.config.carla_scenarios)} scenarios)...")
            carla_results = self._run_carla(controller)
            all_results.append(('carla', carla_results))
            
        # SUMO validation
        if self.config.sumo_traffic_density:
            logger.info("Running SUMO traffic validation...")
            sumo_results = self._run_sumo(controller)
            all_results.append(('sumo', sumo_results))
            
        # MATLAB verification
        if self.config.matlab_validation:
            logger.info("Running MATLAB formal verification...")
            matlab_results = self._run_matlab(controller)
            all_results.append(('matlab', matlab_results))
            
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(aggregated, save_raw_data)
        
        # Generate summary report
        self._generate_summary(aggregated)
        
        logger.info(f"Validation suite complete. Results saved to {self.results_dir}")
        
        return aggregated
        
    def _run_monte_carlo(self, controller) -> pd.DataFrame:
        """Execute Monte Carlo validation"""
        def controller_wrapper(scenario):
            """Wrap controller to match Monte Carlo interface"""
            # Convert scenario to sensor data format expected by controller
            sensor_data = self._scenario_to_sensor_data(scenario)
            
            # Run controller
            control = controller(sensor_data)
            
            # Track explainability
            if hasattr(controller, 'get_decision_trace'):
                trace = controller.get_decision_trace()
                self.explainability_tracker.record_decision(trace)
                
            # Simulate and compute metrics
            metrics = self.metrics_collector.compute_scenario_metrics(
                sensor_data, control, scenario
            )
            
            return metrics
            
        results = self.monte_carlo.run_validation(
            controller_wrapper,
            num_runs=self.config.monte_carlo_runs
        )
        
        return results
        
    def _run_carla(self, controller) -> pd.DataFrame:
        """Execute CARLA validation"""
        results = []
        
        for scenario_name in self.config.carla_scenarios:
            logger.info(f"Running CARLA scenario: {scenario_name}")
            
            # Configure scenario
            self._configure_carla_scenario(scenario_name)
            
            # Spawn ego vehicle
            self.carla.spawn_ego_vehicle()
            
            # Run scenario
            try:
                def controller_wrapper(sensor_data):
                    """Wrap controller to match CARLA interface"""
                    control = controller(sensor_data)
                    
                    # Convert to CARLA control format
                    carla_control = carla.VehicleControl(
                        throttle=control.get('throttle', 0.0),
                        brake=control.get('brake', 0.0),
                        steer=control.get('steering', 0.0)
                    )
                    
                    return carla_control
                    
                metrics = self.carla.run_scenario(controller_wrapper)
                metrics['scenario'] = scenario_name
                results.append(metrics)
                
            finally:
                self.carla.cleanup()
                
        return pd.DataFrame(results)
        
    def _run_sumo(self, controller) -> pd.DataFrame:
        """Execute SUMO traffic validation"""
        # Generate network and traffic
        sumo_dir = self.results_dir / "sumo"
        sumo_dir.mkdir(exist_ok=True)
        
        self.sumo.generate_network(sumo_dir)
        self.sumo.generate_traffic_demand(sumo_dir)
        
        # Start simulation
        self.sumo.start_simulation()
        
        try:
            def controller_wrapper(vehicle_state):
                """Wrap controller to match SUMO interface"""
                # Convert SUMO state to sensor data format
                sensor_data = self._sumo_state_to_sensor_data(vehicle_state)
                
                control = controller(sensor_data)
                
                return control
                
            results = self.sumo.run_scenario(controller_wrapper)
            
        finally:
            self.sumo.close()
            
        return pd.DataFrame([results])
        
    def _run_matlab(self, controller) -> pd.DataFrame:
        """Execute MATLAB formal verification"""
        # Extract controller model for verification
        if hasattr(controller, 'get_system_matrices'):
            controller_model = {
                'system_matrices': controller.get_system_matrices(),
                'initial_state': np.zeros(controller.state_dim)
            }
        else:
            # Use example system for demonstration
            n, m = 4, 2  # State and control dimensions
            controller_model = {
                'system_matrices': {
                    'A': np.random.randn(n, n) * 0.1,
                    'B': np.random.randn(n, m),
                    'C': np.eye(n),
                    'D': np.zeros((n, m))
                },
                'initial_state': np.zeros(n)
            }
            
        results = self.matlab.run_comprehensive_verification(controller_model)
        
        # Flatten nested results for DataFrame
        flat_results = {}
        for category, values in results.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_results[f"{category}_{key}"] = value
            else:
                flat_results[category] = values
                
        return pd.DataFrame([flat_results])
        
    def _scenario_to_sensor_data(self, scenario):
        """Convert Monte Carlo scenario to sensor data format"""
        # This would be implemented based on controller interface
        return {
            'timestamp': 0.0,
            'weather': scenario.weather,
            'vehicle_count': scenario.vehicle_count,
            'pedestrian_count': scenario.pedestrian_count,
            'sensor_failures': scenario.sensor_failures,
            'communication_delay': scenario.communication_delay
        }
        
    def _sumo_state_to_sensor_data(self, vehicle_state):
        """Convert SUMO vehicle state to sensor data format"""
        return {
            'timestamp': 0.0,
            'position': vehicle_state['position'],
            'speed': vehicle_state['speed'],
            'nearby_vehicles': vehicle_state['nearby_vehicles']
        }
        
    def _configure_carla_scenario(self, scenario_name: str):
        """Configure CARLA for specific scenario"""
        scenario_configs = {
            'urban_day': {
                'map': 'Town04',
                'weather': 'ClearNoon',
                'num_vehicles': 30,
                'num_pedestrians': 10
            },
            'urban_rain': {
                'map': 'Town04',
                'weather': 'HardRainNoon',
                'num_vehicles': 20,
                'num_pedestrians': 5
            },
            'highway_fog': {
                'map': 'Town06',
                'weather': 'CloudyNoon',
                'num_vehicles': 50,
                'num_pedestrians': 0
            }
        }
        
        if scenario_name in scenario_configs:
            config = scenario_configs[scenario_name]
            # Apply configuration to CARLA
            # This would interface with CARLA config
            
    def _aggregate_results(self, results_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """Aggregate results from all simulators"""
        aggregated = []
        
        for simulator, results in results_list:
            results['simulator'] = simulator
            aggregated.append(results)
            
        return pd.concat(aggregated, ignore_index=True)
        
    def _save_results(self, results: pd.DataFrame, save_raw_data: bool):
        """Save results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated results
        if self.config.output_format == 'parquet':
            output_path = self.results_dir / f"results_{timestamp}.parquet"
            results.to_parquet(output_path, compression='snappy')
        elif self.config.output_format == 'csv':
            output_path = self.results_dir / f"results_{timestamp}.csv"
            results.to_csv(output_path, index=False)
        else:
            output_path = self.results_dir / f"results_{timestamp}.json"
            results.to_json(output_path, orient='records', indent=2)
            
        # Save explainability data
        explain_path = self.results_dir / f"explainability_{timestamp}.json"
        self.explainability_tracker.save_traces(explain_path)
        
        logger.info(f"Results saved to {output_path}")
        
    def _generate_summary(self, results: pd.DataFrame):
        """Generate summary statistics and plots"""
        summary = {
            'total_scenarios': len(results),
            'collision_rate': (results['collision'].sum() / len(results) * 100),
            'avg_control_latency': results['avg_control_latency'].mean(),
            'p99_control_latency': results['p99_control_latency'].mean()
        }
        
        # Group by simulator
        by_simulator = results.groupby('simulator').agg({
            'collision': 'mean',
            'avg_control_latency': 'mean',
            'total_distance': 'mean'
        })
        
        summary['by_simulator'] = by_simulator.to_dict()
        
        # Save summary
        summary_path = self.results_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Summary: {summary}")
        
    def cleanup(self):
        """Clean up resources"""
        if self._carla:
            self._carla.cleanup()
        if self._sumo:
            self._sumo.close()
        if self._matlab:
            self._matlab.close()