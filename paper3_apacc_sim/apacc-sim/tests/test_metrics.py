"""
Unit tests for metrics computation module
"""

import unittest
import numpy as np
from typing import Dict, Any

from apacc_sim.metrics import MetricsCollector, SafetyMetrics, PerformanceMetrics


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection and computation"""
    
    def setUp(self):
        """Initialize metrics collector"""
        self.collector = MetricsCollector()
    
    def test_collision_detection(self):
        """Test collision detection from LiDAR data"""
        # Create sensor data with close obstacle
        sensor_data = {
            'lidar_points': np.array([
                [0.3, 0.0, 0.0, 1.0],  # Very close point
                [5.0, 1.0, 0.0, 1.0],  # Far point
                [10.0, 0.0, 0.0, 1.0]  # Far point
            ])
        }
        
        class MockScenario:
            scenario_id = 1
            speed_limit = 50
        
        metrics = self.collector.compute_scenario_metrics(
            sensor_data, {}, MockScenario()
        )
        
        self.assertTrue(metrics['collision'])
    
    def test_time_to_collision_computation(self):
        """Test TTC computation with nearby vehicles"""
        sensor_data = {
            'vehicle_state': {
                'x': 0.0, 'y': 0.0,
                'vx': 10.0, 'vy': 0.0
            },
            'nearby_vehicles': [{
                'position': [20.0, 0.0],
                'vx': 5.0, 'vy': 0.0
            }]
        }
        
        # Vehicle ahead going slower - should compute TTC
        metrics = self.collector._compute_safety_metrics(sensor_data, None)
        self.assertLess(metrics.time_to_collision, float('inf'))
        self.assertGreater(metrics.time_to_collision, 0)
    
    def test_performance_metrics(self):
        """Test performance metric computation"""
        sensor_data = {
            'control_latency': 0.005,  # 5ms
            'acceleration': [1.0, 0.5, 0.0],
            'vehicle_state': {
                'speed': 15.0,
                'distance': 100.0
            }
        }
        
        control = {
            'steering': 0.1,
            'throttle': 0.5,
            'brake': 0.0
        }
        
        metrics = self.collector._compute_performance_metrics(sensor_data, control)
        
        self.assertEqual(metrics.control_latency_ms, 5.0)
        self.assertGreater(metrics.comfort_score, 0)
        self.assertEqual(metrics.average_speed, 15.0)
    
    def test_aggregation(self):
        """Test metric aggregation across scenarios"""
        metrics_list = [
            {
                'collision': False,
                'time_to_collision': 5.0,
                'control_latency_ms': 4.0,
                'near_misses': 0
            },
            {
                'collision': True,
                'time_to_collision': 1.5,
                'control_latency_ms': 6.0,
                'near_misses': 1
            },
            {
                'collision': False,
                'time_to_collision': 3.0,
                'control_latency_ms': 5.0,
                'near_misses': 0
            }
        ]
        
        aggregated = self.collector.aggregate_metrics(metrics_list)
        
        self.assertEqual(aggregated['total_scenarios'], 3)
        self.assertAlmostEqual(aggregated['collision_rate'], 33.33, places=1)
        self.assertEqual(aggregated['collision_count'], 1)
        self.assertEqual(aggregated['total_near_misses'], 1)
        self.assertAlmostEqual(aggregated['avg_control_latency'], 5.0)
    
    def test_certification_metrics(self):
        """Test computation of certification-aligned metrics"""
        # Add some test data
        for i in range(100):
            self.collector.metrics_history.append({
                'collision': i < 2,  # 2% collision rate
                'time_to_collision': 0.5 if i < 5 else 3.0,
                'near_misses': 1 if i < 10 else 0,
                'safety_margin_violations': 3 if i < 3 else 0
            })
        
        cert_metrics = self.collector.compute_certification_metrics()
        
        self.assertIn('asil_d_collision_rate', cert_metrics)
        self.assertIn('sotif_edge_cases', cert_metrics)
        self.assertEqual(cert_metrics['sample_size'], 100)
        self.assertFalse(cert_metrics['asil_d_compliance'])  # 2% > threshold
    
    def test_confidence_interval(self):
        """Test confidence interval computation"""
        lower, upper = self.collector._compute_confidence_interval(0.05, 1000)
        
        self.assertLess(lower, 0.05)
        self.assertGreater(upper, 0.05)
        self.assertAlmostEqual(upper - lower, 0.027, places=2)  # Expected margin


if __name__ == '__main__':
    unittest.main()