"""
Integration tests for APACC-Sim toolkit
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import yaml

from apacc_sim import SimulationOrchestrator, ScenarioConfig


class MockController:
    """Mock controller for testing"""
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, sensor_data):
        """Simple control logic"""
        self.call_count += 1
        
        control = {
            'steering': 0.0,
            'throttle': 0.5,
            'brake': 0.0
        }
        
        # Add some variety based on sensor data
        if 'nearby_vehicles' in sensor_data and sensor_data['nearby_vehicles']:
            control['brake'] = 0.3
            control['throttle'] = 0.2
        
        return control
    
    def get_system_matrices(self):
        """Return example system matrices for MATLAB verification"""
        import numpy as np
        return {
            'A': np.array([[0, 1], [-1, -1]]),
            'B': np.array([[0], [1]]),
            'C': np.eye(2),
            'D': np.zeros((2, 1))
        }


class TestIntegration(unittest.TestCase):
    """Test complete validation pipeline"""
    
    def setUp(self):
        """Create test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.temp_dir) / "results"
        self.results_dir.mkdir()
        
        # Create mock configs directory structure
        self.configs_dir = Path(self.temp_dir) / "configs"
        for subdir in ['monte_carlo', 'carla', 'sumo', 'matlab']:
            (self.configs_dir / subdir).mkdir(parents=True)
        
        # Copy or create minimal configs
        self._create_test_configs()
    
    def _create_test_configs(self):
        """Create minimal test configurations"""
        # Monte Carlo config
        mc_config = {
            'simulation': {
                'runs': 10,
                'seed': 42,
                'parallel_workers': 1,
                'checkpoint_interval': 5
            },
            'scenario_parameters': {
                'weather': {
                    'distribution': 'categorical',
                    'categories': ['clear', 'rain']
                },
                'lighting': {
                    'distribution': 'categorical',
                    'probabilities': [0.5, 0.5]
                },
                'vehicle_density': {
                    'distribution': 'uniform',
                    'min': 5,
                    'max': 10
                },
                'pedestrian_density': {
                    'distribution': 'uniform',
                    'min': 0,
                    'max': 5
                },
                'sensor_failures': {
                    'distribution': 'bernoulli',
                    'probability': 0.1,
                    'types': ['camera']
                },
                'communication_delay': {
                    'distribution': 'exponential',
                    'lambda': 0.1,
                    'trigger_probability': 0.1
                }
            },
            'constraints': {
                'episode_duration': 10,
                'map_regions': ['urban']
            },
            'output': {
                'format': 'parquet'
            }
        }
        
        with open(self.configs_dir / 'monte_carlo' / 'default.yaml', 'w') as f:
            yaml.dump(mc_config, f)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized with config"""
        config = ScenarioConfig(
            name="test_run",
            monte_carlo_runs=5,
            carla_scenarios=[],
            sumo_traffic_density=None,
            matlab_validation=False
        )
        
        orchestrator = SimulationOrchestrator(config)
        self.assertEqual(orchestrator.config.name, "test_run")
        self.assertEqual(orchestrator.config.monte_carlo_runs, 5)
    
    def test_monte_carlo_only_validation(self):
        """Test running Monte Carlo validation only"""
        config = ScenarioConfig(
            name="mc_only_test",
            monte_carlo_runs=5,
            carla_scenarios=[],
            sumo_traffic_density=None,
            matlab_validation=False
        )
        
        # Mock configs path
        import sys
        import os
        old_path = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            orchestrator = SimulationOrchestrator(config)
            controller = MockController()
            
            results = orchestrator.run_validation_suite(controller)
            
            self.assertGreater(len(results), 0)
            self.assertIn('simulator', results.columns)
            self.assertIn('collision', results.columns)
            self.assertEqual(controller.call_count, 5)
            
        finally:
            os.chdir(old_path)
    
    def test_results_saving(self):
        """Test that results are saved correctly"""
        config = ScenarioConfig(
            name="save_test",
            monte_carlo_runs=3,
            carla_scenarios=[],
            sumo_traffic_density=None,
            matlab_validation=False,
            output_format='csv'
        )
        
        # Change to temp directory for config loading
        import os
        old_path = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            orchestrator = SimulationOrchestrator(config)
            controller = MockController()
            
            results = orchestrator.run_validation_suite(controller)
            
            # Check that results were saved
            results_files = list(orchestrator.results_dir.glob("*.csv"))
            self.assertGreater(len(results_files), 0)
            
            # Check summary was created
            summary_file = orchestrator.results_dir / "summary.json"
            self.assertTrue(summary_file.exists())
            
        finally:
            os.chdir(old_path)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()