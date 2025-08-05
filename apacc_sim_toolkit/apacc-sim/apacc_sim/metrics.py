"""
Metrics Collection and Analysis Module

Standardized metrics computation for safety, performance,
and certification compliance across all simulators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class SafetyMetrics:
    """Container for safety-related metrics"""
    collision: bool = False
    time_to_collision: float = float('inf')
    lane_deviation: float = 0.0
    speed_limit_violation: bool = False
    safety_margin_violations: int = 0
    near_misses: int = 0
    

@dataclass
class PerformanceMetrics:
    """Container for performance-related metrics"""
    control_latency_ms: float = 0.0
    trajectory_smoothness: float = 0.0
    fuel_efficiency: float = 0.0
    comfort_score: float = 0.0
    distance_traveled: float = 0.0
    average_speed: float = 0.0
    

class MetricsCollector:
    """
    Centralized metrics collection and computation
    
    Provides standardized metrics aligned with automotive
    safety standards (ISO 26262, ISO 21448).
    """
    
    def __init__(self):
        """Initialize metrics collector with thresholds"""
        self.thresholds = {
            'collision_distance': 0.5,  # meters
            'ttc_critical': 2.0,  # seconds
            'lane_deviation_max': 0.5,  # meters
            'speed_limit_tolerance': 0.1,  # 10% over limit
            'comfort_accel_max': 3.0,  # m/s^2
            'comfort_jerk_max': 5.0,  # m/s^3
        }
        
        self.metrics_history = []
        
    def compute_scenario_metrics(self, sensor_data: Dict[str, Any],
                               control: Dict[str, float],
                               scenario: Any) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a scenario
        
        Args:
            sensor_data: Current sensor readings
            control: Applied control commands
            scenario: Scenario parameters
            
        Returns:
            Dictionary of computed metrics
        """
        safety_metrics = self._compute_safety_metrics(sensor_data, scenario)
        performance_metrics = self._compute_performance_metrics(sensor_data, control)
        
        # Combine all metrics
        metrics = {
            **safety_metrics.__dict__,
            **performance_metrics.__dict__,
            'timestamp': sensor_data.get('timestamp', 0.0),
            'scenario_id': getattr(scenario, 'scenario_id', 0)
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
        
    def _compute_safety_metrics(self, sensor_data: Dict[str, Any],
                              scenario: Any) -> SafetyMetrics:
        """Compute safety-related metrics"""
        metrics = SafetyMetrics()
        
        # Collision detection
        if 'lidar_points' in sensor_data and sensor_data['lidar_points'] is not None:
            closest_distance = self._compute_closest_obstacle_distance(
                sensor_data['lidar_points']
            )
            metrics.collision = closest_distance < self.thresholds['collision_distance']
            
        # Time to collision
        if 'nearby_vehicles' in sensor_data:
            metrics.time_to_collision = self._compute_time_to_collision(
                sensor_data.get('vehicle_state', {}),
                sensor_data['nearby_vehicles']
            )
            
        # Lane deviation (simplified)
        if 'lane_position' in sensor_data:
            metrics.lane_deviation = abs(sensor_data['lane_position'])
            if metrics.lane_deviation > self.thresholds['lane_deviation_max']:
                metrics.safety_margin_violations += 1
                
        # Speed limit check
        if 'speed' in sensor_data and hasattr(scenario, 'speed_limit'):
            if sensor_data['speed'] > scenario.speed_limit * (1 + self.thresholds['speed_limit_tolerance']):
                metrics.speed_limit_violation = True
                metrics.safety_margin_violations += 1
                
        # Near miss detection
        if metrics.time_to_collision < self.thresholds['ttc_critical']:
            metrics.near_misses = 1
            
        return metrics
        
    def _compute_performance_metrics(self, sensor_data: Dict[str, Any],
                                   control: Dict[str, float]) -> PerformanceMetrics:
        """Compute performance-related metrics"""
        metrics = PerformanceMetrics()
        
        # Control latency (if provided)
        if 'control_latency' in sensor_data:
            metrics.control_latency_ms = sensor_data['control_latency'] * 1000
            
        # Trajectory smoothness (based on control changes)
        if hasattr(self, '_prev_control'):
            control_diff = sum(abs(control.get(k, 0) - self._prev_control.get(k, 0))
                             for k in ['steering', 'throttle', 'brake'])
            metrics.trajectory_smoothness = 1.0 / (1.0 + control_diff)
        else:
            metrics.trajectory_smoothness = 1.0
            
        self._prev_control = control.copy()
        
        # Comfort score (based on acceleration/jerk)
        if 'acceleration' in sensor_data:
            accel_magnitude = np.linalg.norm(sensor_data['acceleration'])
            comfort_accel = max(0, 1 - accel_magnitude / self.thresholds['comfort_accel_max'])
            
            if hasattr(self, '_prev_acceleration'):
                jerk = np.linalg.norm(
                    np.array(sensor_data['acceleration']) - np.array(self._prev_acceleration)
                ) / 0.01  # Assuming 100Hz
                comfort_jerk = max(0, 1 - jerk / self.thresholds['comfort_jerk_max'])
                metrics.comfort_score = (comfort_accel + comfort_jerk) / 2
            else:
                metrics.comfort_score = comfort_accel
                
            self._prev_acceleration = sensor_data['acceleration']
            
        # Distance and speed
        if 'vehicle_state' in sensor_data:
            state = sensor_data['vehicle_state']
            if 'speed' in state:
                metrics.average_speed = state['speed']
            if 'distance' in state:
                metrics.distance_traveled = state['distance']
                
        return metrics
        
    def _compute_closest_obstacle_distance(self, lidar_points: np.ndarray) -> float:
        """Compute distance to closest obstacle from LiDAR data"""
        if lidar_points is None or len(lidar_points) == 0:
            return float('inf')
            
        # Simple approach: find minimum distance in forward cone
        # Filter points in front of vehicle (x > 0)
        forward_points = lidar_points[lidar_points[:, 0] > 0]
        
        if len(forward_points) == 0:
            return float('inf')
            
        # Compute distances
        distances = np.linalg.norm(forward_points[:, :3], axis=1)
        
        return np.min(distances)
        
    def _compute_time_to_collision(self, ego_state: Dict[str, float],
                                  nearby_vehicles: List[Dict]) -> float:
        """Compute minimum time to collision with nearby vehicles"""
        if not nearby_vehicles or not ego_state:
            return float('inf')
            
        min_ttc = float('inf')
        
        ego_pos = np.array([ego_state.get('x', 0), ego_state.get('y', 0)])
        ego_vel = np.array([ego_state.get('vx', 0), ego_state.get('vy', 0)])
        
        for vehicle in nearby_vehicles:
            # Relative position and velocity
            rel_pos = np.array(vehicle['position'][:2]) - ego_pos
            rel_vel = np.array([vehicle.get('vx', 0), vehicle.get('vy', 0)]) - ego_vel
            
            # Check if vehicles are approaching
            if np.dot(rel_pos, rel_vel) < 0:
                # Compute TTC (simplified)
                distance = np.linalg.norm(rel_pos)
                approach_speed = -np.dot(rel_pos, rel_vel) / distance
                
                if approach_speed > 0:
                    ttc = distance / approach_speed
                    min_ttc = min(min_ttc, ttc)
                    
        return min_ttc
        
    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple scenarios
        
        Args:
            metrics_list: List of metrics dictionaries
            
        Returns:
            Aggregated statistics
        """
        df = pd.DataFrame(metrics_list)
        
        aggregated = {
            'total_scenarios': len(metrics_list),
            'collision_rate': df['collision'].mean() * 100,
            'collision_count': df['collision'].sum(),
            'avg_ttc': df['time_to_collision'].mean(),
            'min_ttc': df['time_to_collision'].min(),
            'avg_lane_deviation': df['lane_deviation'].mean(),
            'speed_violations': df['speed_limit_violation'].sum(),
            'total_near_misses': df['near_misses'].sum(),
            'avg_control_latency': df['control_latency_ms'].mean(),
            'p99_control_latency': df['control_latency_ms'].quantile(0.99),
            'avg_comfort_score': df['comfort_score'].mean(),
            'total_distance': df['distance_traveled'].sum(),
            'avg_speed': df['average_speed'].mean()
        }
        
        # Add safety score (0-100)
        safety_score = 100.0
        safety_score -= aggregated['collision_rate'] * 10  # -10 points per 1% collision
        safety_score -= aggregated['speed_violations'] / len(metrics_list) * 20
        safety_score -= aggregated['total_near_misses'] / len(metrics_list) * 5
        safety_score = max(0, safety_score)
        
        aggregated['safety_score'] = safety_score
        
        return aggregated
        
    def compute_certification_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics specifically for certification compliance
        
        Returns metrics aligned with ISO 26262 and ISO 21448
        """
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame(self.metrics_history)
        
        certification_metrics = {
            # ISO 26262 - Functional Safety
            'asil_d_collision_rate': df['collision'].mean(),  # Must be < 10^-9/hour
            'asil_d_compliance': df['collision'].mean() < 1e-6,  # Simplified threshold
            
            # ISO 21448 - SOTIF
            'sotif_near_miss_rate': df['near_misses'].sum() / len(df),
            'sotif_edge_cases': self._identify_edge_cases(df),
            'sotif_performance_limitations': self._assess_performance_limits(df),
            
            # Controllability metrics
            'controllability_c3_events': (df['time_to_collision'] < 1.0).sum(),
            'severity_s3_events': df['collision'].sum(),
            
            # Statistical confidence
            'sample_size': len(df),
            'collision_confidence_interval': self._compute_confidence_interval(
                df['collision'].mean(), len(df)
            )
        }
        
        return certification_metrics
        
    def _identify_edge_cases(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify edge cases for SOTIF analysis"""
        edge_cases = []
        
        # Multiple simultaneous failures
        multi_failure = df['safety_margin_violations'] > 2
        if multi_failure.any():
            edge_cases.append({
                'type': 'multiple_safety_violations',
                'count': multi_failure.sum(),
                'percentage': multi_failure.mean() * 100
            })
            
        # Very low TTC events
        critical_ttc = df['time_to_collision'] < 0.5
        if critical_ttc.any():
            edge_cases.append({
                'type': 'critical_ttc',
                'count': critical_ttc.sum(),
                'min_ttc': df['time_to_collision'].min()
            })
            
        return edge_cases
        
    def _assess_performance_limits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess performance limitations for SOTIF"""
        return {
            'max_control_latency': df['control_latency_ms'].max(),
            'latency_exceeds_100ms': (df['control_latency_ms'] > 100).sum(),
            'comfort_violations': (df['comfort_score'] < 0.5).sum(),
            'trajectory_instability': (df['trajectory_smoothness'] < 0.3).sum()
        }
        
    def _compute_confidence_interval(self, mean: float, n: int, 
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for binary metric"""
        import scipy.stats as stats
        
        if n == 0:
            return (0.0, 1.0)
            
        # Wilson score interval for binary proportion
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / n
        center = (mean + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
        
    def export_metrics_report(self, output_path: str, format: str = 'html'):
        """
        Export comprehensive metrics report
        
        Args:
            output_path: Path to save report
            format: Output format (html, pdf, markdown)
        """
        if not self.metrics_history:
            logger.warning("No metrics to export")
            return
            
        aggregated = self.aggregate_metrics(self.metrics_history)
        certification = self.compute_certification_metrics()
        
        if format == 'html':
            self._export_html_report(output_path, aggregated, certification)
        elif format == 'markdown':
            self._export_markdown_report(output_path, aggregated, certification)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _export_html_report(self, output_path: str, aggregated: Dict, 
                          certification: Dict):
        """Generate HTML metrics report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>APACC-Sim Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>APACC-Sim Validation Report</h1>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Scenarios</td><td>{total_scenarios}</td></tr>
                <tr><td>Collision Rate</td><td>{collision_rate:.2f}%</td></tr>
                <tr><td>Safety Score</td><td>{safety_score:.1f}/100</td></tr>
                <tr><td>Average Control Latency</td><td>{avg_control_latency:.2f} ms</td></tr>
            </table>
            
            <h2>Certification Compliance</h2>
            <table>
                <tr><th>Standard</th><th>Requirement</th><th>Status</th></tr>
                <tr>
                    <td>ISO 26262</td>
                    <td>ASIL-D Collision Rate < 10^-6</td>
                    <td class="{asil_status}">{asil_d_compliance}</td>
                </tr>
                <tr>
                    <td>ISO 21448</td>
                    <td>SOTIF Edge Cases Identified</td>
                    <td>{edge_case_count} found</td>
                </tr>
            </table>
            
            <p>Generated by APACC-Sim v1.0.0</p>
        </body>
        </html>
        """
        
        # Prepare template variables
        asil_status = 'pass' if certification['asil_d_compliance'] else 'fail'
        edge_case_count = len(certification.get('sotif_edge_cases', []))
        
        html_content = html_template.format(
            total_scenarios=aggregated['total_scenarios'],
            collision_rate=aggregated['collision_rate'],
            safety_score=aggregated['safety_score'],
            avg_control_latency=aggregated['avg_control_latency'],
            asil_d_compliance='PASS' if certification['asil_d_compliance'] else 'FAIL',
            asil_status=asil_status,
            edge_case_count=edge_case_count
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report saved to {output_path}")
        
    def _export_markdown_report(self, output_path: str, aggregated: Dict,
                              certification: Dict):
        """Generate Markdown metrics report"""
        md_content = f"""# APACC-Sim Validation Report

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Scenarios | {aggregated['total_scenarios']} |
| Collision Rate | {aggregated['collision_rate']:.2f}% |
| Safety Score | {aggregated['safety_score']:.1f}/100 |
| Avg Control Latency | {aggregated['avg_control_latency']:.2f} ms |
| P99 Control Latency | {aggregated['p99_control_latency']:.2f} ms |

## Safety Metrics

- Collision Count: {aggregated['collision_count']}
- Near Misses: {aggregated['total_near_misses']}
- Speed Violations: {aggregated['speed_violations']}
- Average TTC: {aggregated['avg_ttc']:.2f}s
- Minimum TTC: {aggregated['min_ttc']:.2f}s

## Certification Compliance

### ISO 26262 (Functional Safety)
- ASIL-D Compliance: {'✓ PASS' if certification['asil_d_compliance'] else '✗ FAIL'}
- Collision Rate: {certification['asil_d_collision_rate']:.2e}

### ISO 21448 (SOTIF)
- Edge Cases Identified: {len(certification.get('sotif_edge_cases', []))}
- Performance Limitations Found: {len(certification.get('sotif_performance_limitations', {}))}

Generated by APACC-Sim v1.0.0
"""
        
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        logger.info(f"Markdown report saved to {output_path}")