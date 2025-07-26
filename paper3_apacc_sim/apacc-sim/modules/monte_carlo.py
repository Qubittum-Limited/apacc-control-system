"""
Monte Carlo Statistical Validation Module

Generates parameterized scenarios with controlled stochastic elements
for statistical validation of autonomous control systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml
import hashlib
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScenarioParameters:
    """Container for scenario generation parameters"""
    weather: str
    lighting: str
    vehicle_count: int
    pedestrian_count: int
    sensor_failures: List[str]
    communication_delay: float
    map_region: str
    episode_duration: float
    

class MonteCarloSimulator:
    """
    Monte Carlo simulation framework for statistical validation
    
    Provides reproducible scenario generation with versioned seeds
    and parallel execution capabilities.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Monte Carlo simulator
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.rng = self._initialize_rng()
        self.scenario_cache = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = ['simulation', 'scenario_parameters', 'constraints']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
        
    def _initialize_rng(self) -> np.random.Generator:
        """Initialize random number generator with versioned seed"""
        base_seed = self.config['simulation']['seed']
        
        # Create versioned seed using SHA-256
        seed_string = f"apacc_sim_v1.0_{base_seed}"
        seed_hash = hashlib.sha256(seed_string.encode()).digest()
        seed = int.from_bytes(seed_hash[:4], 'big')
        
        logger.info(f"Initialized RNG with seed: {seed} (base: {base_seed})")
        return np.random.default_rng(seed)
        
    def generate_scenario(self, scenario_id: int) -> ScenarioParameters:
        """
        Generate a single scenario with controlled randomness
        
        Args:
            scenario_id: Unique identifier for reproducibility
            
        Returns:
            ScenarioParameters object
        """
        # Use scenario_id to ensure reproducibility
        local_rng = np.random.default_rng(self.rng.integers(0, 2**32) + scenario_id)
        
        # Sample weather
        weather_params = self.config['scenario_parameters']['weather']
        weather_prob = local_rng.beta(
            weather_params['alpha'], 
            weather_params['beta']
        )
        weather_idx = int(weather_prob * len(weather_params['categories']))
        weather = weather_params['categories'][min(weather_idx, len(weather_params['categories'])-1)]
        
        # Sample lighting
        lighting_params = self.config['scenario_parameters']['lighting']
        lighting = local_rng.choice(
            ['day', 'dusk', 'night', 'dawn'],
            p=lighting_params['probabilities']
        )
        
        # Sample traffic
        vehicle_params = self.config['scenario_parameters']['vehicle_density']
        vehicle_count = local_rng.poisson(vehicle_params['lambda'])
        vehicle_count = np.clip(vehicle_count, vehicle_params['min'], vehicle_params['max'])
        
        pedestrian_params = self.config['scenario_parameters']['pedestrian_density']
        pedestrian_count = local_rng.poisson(pedestrian_params['lambda'])
        pedestrian_count = np.clip(pedestrian_count, pedestrian_params['min'], pedestrian_params['max'])
        
        # Sample failures
        failure_params = self.config['scenario_parameters']['sensor_failures']
        sensor_failures = []
        for sensor in failure_params['types']:
            if local_rng.random() < failure_params['probability']:
                sensor_failures.append(sensor)
                
        # Sample communication delay
        delay_params = self.config['scenario_parameters']['communication_delay']
        if local_rng.random() < delay_params['trigger_probability']:
            comm_delay = local_rng.exponential(1/delay_params['lambda'])
        else:
            comm_delay = 0.0
            
        # Sample map and duration
        map_region = local_rng.choice(self.config['constraints']['map_regions'])
        episode_duration = self.config['constraints']['episode_duration']
        
        return ScenarioParameters(
            weather=weather,
            lighting=lighting,
            vehicle_count=int(vehicle_count),
            pedestrian_count=int(pedestrian_count),
            sensor_failures=sensor_failures,
            communication_delay=float(comm_delay),
            map_region=map_region,
            episode_duration=float(episode_duration)
        )
        
    def run_validation(self, controller_func, num_runs: Optional[int] = None) -> pd.DataFrame:
        """
        Run Monte Carlo validation campaign
        
        Args:
            controller_func: Function that takes ScenarioParameters and returns metrics
            num_runs: Number of scenarios to run (None for config default)
            
        Returns:
            DataFrame with results from all scenarios
        """
        if num_runs is None:
            num_runs = self.config['simulation']['runs']
            
        logger.info(f"Starting Monte Carlo validation with {num_runs} runs")
        
        # Determine number of workers
        n_workers = self.config['simulation']['parallel_workers']
        if n_workers == -1:
            n_workers = None  # Use all available cores
            
        results = []
        
        # Parallel execution with progress tracking
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            futures = []
            for i in range(num_runs):
                scenario = self.generate_scenario(i)
                future = executor.submit(self._run_single_scenario, 
                                       controller_func, scenario, i)
                futures.append(future)
                
            # Collect results with progress updates
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{num_runs} scenarios")
                    
                # Checkpointing
                if (i + 1) % self.config['simulation']['checkpoint_interval'] == 0:
                    self._save_checkpoint(results, i + 1)
                    
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save final results
        output_format = self.config['output']['format']
        output_path = Path(f"results/monte_carlo_{num_runs}_runs.{output_format}")
        output_path.parent.mkdir(exist_ok=True)
        
        if output_format == 'parquet':
            df_results.to_parquet(output_path, compression='snappy')
        elif output_format == 'hdf5':
            df_results.to_hdf(output_path, key='results', mode='w')
        else:
            df_results.to_csv(output_path, index=False)
            
        logger.info(f"Saved results to {output_path}")
        return df_results
        
    def _run_single_scenario(self, controller_func, scenario: ScenarioParameters, 
                           scenario_id: int) -> Dict[str, Any]:
        """Execute controller in single scenario and collect metrics"""
        try:
            # Run controller
            metrics = controller_func(scenario)
            
            # Add scenario metadata
            result = {
                'scenario_id': scenario_id,
                'weather': scenario.weather,
                'lighting': scenario.lighting,
                'vehicle_count': scenario.vehicle_count,
                'pedestrian_count': scenario.pedestrian_count,
                'sensor_failures': ','.join(scenario.sensor_failures),
                'communication_delay': scenario.communication_delay,
                'map_region': scenario.map_region,
                **metrics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario_id}: {str(e)}")
            return {
                'scenario_id': scenario_id,
                'error': str(e),
                'collision': True,  # Conservative failure
                'success': False
            }
            
    def _save_checkpoint(self, results: List[Dict], checkpoint_id: int):
        """Save intermediate results for fault tolerance"""
        checkpoint_path = Path(f"results/checkpoints/monte_carlo_checkpoint_{checkpoint_id}.parquet")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_checkpoint = pd.DataFrame(results)
        df_checkpoint.to_parquet(checkpoint_path, compression='snappy')
        logger.info(f"Saved checkpoint at scenario {checkpoint_id}")