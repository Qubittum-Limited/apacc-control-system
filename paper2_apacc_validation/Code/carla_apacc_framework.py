#!/usr/bin/env python3
"""
CARLA-APACC Monte Carlo Simulation Framework
For Paper 2: Quantitative Validation of APACC
"""

import carla
import numpy as np
import json
import csv
import random
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('APACC_Simulation')

@dataclass
class ScenarioConfig:
    """Configuration for a single test scenario"""
    scenario_id: str
    weather: Dict[str, float]
    num_vehicles: int
    num_pedestrians: int
    time_of_day: float
    occlusion_zones: List[Tuple[float, float, float, float]]
    fault_injections: List[Dict[str, any]]
    duration_seconds: float
    
@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    scenario_id: str
    timestamp: float
    control_loop_latency_ms: float
    safety_violations: int
    collision_occurred: bool
    ttc_min: float  # Minimum time-to-collision
    lateral_deviation_max: float
    acceleration_jerk_max: float
    rule_activations: Dict[str, int]
    mpc_optimization_time_ms: float
    fuzzy_inference_time_ms: float
    total_distance_traveled: float
    scenario_success: bool
    
class APACCController:
    """APACC control implementation for CARLA"""
    
    def __init__(self, vehicle, config_path='apacc_config.json'):
        self.vehicle = vehicle
        self.config = self._load_config(config_path)
        self.rule_base = self._initialize_rules()
        self.precognition_horizon = 2.0  # seconds
        self.control_frequency = 100  # Hz
        self.metrics = []
        
    def _load_config(self, path):
        """Load APACC configuration"""
        # Default config if file doesn't exist
        default_config = {
            'fuzzy_rules': 50,
            'mpc_horizon': 20,
            'safety_thresholds': {
                'min_ttc': 2.0,
                'max_lateral_deviation': 0.5,
                'max_acceleration': 3.0,
                'max_jerk': 2.0
            }
        }
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return default_config
            
    def _initialize_rules(self):
        """Initialize fuzzy rule base"""
        rules = {}
        # Example rules - expand based on Paper 1
        rules['pedestrian_proximity'] = {
            'activation_threshold': 10.0,  # meters
            'output_weight': 0.8
        }
        rules['lane_keeping'] = {
            'activation_threshold': 0.3,  # meters from center
            'output_weight': 0.6
        }
        rules['collision_avoidance'] = {
            'activation_threshold': 3.0,  # seconds TTC
            'output_weight': 1.0
        }
        return rules
        
    def control_step(self, sensor_data):
        """Execute one control cycle"""
        start_time = time.time()
        
        # Phase 1: Fuzzy inference (coarse-tuning)
        fuzzy_start = time.time()
        fuzzy_output = self._fuzzy_inference(sensor_data)
        fuzzy_time = (time.time() - fuzzy_start) * 1000
        
        # Phase 2: MPC optimization (fine-tuning)
        mpc_start = time.time()
        control_output = self._mpc_optimization(fuzzy_output, sensor_data)
        mpc_time = (time.time() - mpc_start) * 1000
        
        # Phase 3: Apply control
        self._apply_control(control_output)
        
        # Record metrics
        total_time = (time.time() - start_time) * 1000
        return {
            'latency_ms': total_time,
            'fuzzy_time_ms': fuzzy_time,
            'mpc_time_ms': mpc_time,
            'control': control_output
        }
        
    def _fuzzy_inference(self, sensor_data):
        """Fuzzy logic controller (simplified)"""
        activations = {}
        
        # Check each rule
        for rule_name, rule_params in self.rule_base.items():
            if rule_name == 'pedestrian_proximity':
                # Check pedestrian distances
                min_dist = float('inf')
                for ped in sensor_data.get('pedestrians', []):
                    dist = np.linalg.norm([ped['x'] - sensor_data['ego_x'], 
                                         ped['y'] - sensor_data['ego_y']])
                    min_dist = min(min_dist, dist)
                
                if min_dist < rule_params['activation_threshold']:
                    activations[rule_name] = 1.0 - (min_dist / rule_params['activation_threshold'])
                    
        return activations
        
    def _mpc_optimization(self, fuzzy_output, sensor_data):
        """MPC trajectory optimization (simplified)"""
        # Placeholder for MPC - implement based on Paper 1 equations
        steering = 0.0
        throttle = 0.6
        brake = 0.0
        
        # Modify based on fuzzy activations
        if 'pedestrian_proximity' in fuzzy_output:
            brake = fuzzy_output['pedestrian_proximity'] * 0.8
            throttle = 0.0
            
        return {
            'steering': np.clip(steering, -1.0, 1.0),
            'throttle': np.clip(throttle, 0.0, 1.0),
            'brake': np.clip(brake, 0.0, 1.0)
        }
        
    def _apply_control(self, control):
        """Apply control commands to vehicle"""
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=control['throttle'],
            steer=control['steering'],
            brake=control['brake']
        ))

class MonteCarloSimulator:
    """Main Monte Carlo simulation orchestrator"""
    
    def __init__(self, carla_host='localhost', carla_port=2000):
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.results = []
        
    def generate_scenarios(self, num_scenarios=10000):
        """Generate randomized test scenarios"""
        scenarios = []
        
        for i in range(num_scenarios):
            scenario = ScenarioConfig(
                scenario_id=f"MC_{i:06d}",
                weather={
                    'cloudiness': random.uniform(0, 100),
                    'precipitation': random.uniform(0, 100),
                    'sun_altitude_angle': random.uniform(-90, 90),
                    'fog_density': random.uniform(0, 100),
                    'wetness': random.uniform(0, 100)
                },
                num_vehicles=random.randint(5, 30),
                num_pedestrians=random.randint(0, 20),
                time_of_day=random.uniform(0, 24),
                occlusion_zones=self._generate_occlusions(),
                fault_injections=self._generate_faults(),
                duration_seconds=random.uniform(30, 120)
            )
            scenarios.append(scenario)
            
        return scenarios
        
    def _generate_occlusions(self):
        """Generate random occlusion zones"""
        num_occlusions = random.randint(0, 5)
        occlusions = []
        
        for _ in range(num_occlusions):
            # x, y, width, height
            occlusions.append((
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(2, 10),
                random.uniform(2, 10)
            ))
            
        return occlusions
        
    def _generate_faults(self):
        """Generate random fault injections"""
        faults = []
        
        # Sensor failures
        if random.random() < 0.1:  # 10% chance
            faults.append({
                'type': 'sensor_failure',
                'sensor': random.choice(['lidar', 'camera', 'radar']),
                'duration': random.uniform(0.1, 2.0)
            })
            
        # Communication delays
        if random.random() < 0.2:  # 20% chance
            faults.append({
                'type': 'comm_delay',
                'delay_ms': random.uniform(10, 100)
            })
            
        return faults
        
    def run_scenario(self, scenario: ScenarioConfig):
        """Execute a single scenario"""
        logger.info(f"Running scenario {scenario.scenario_id}")
        
        try:
            # Setup world
            self._setup_weather(scenario.weather)
            ego_vehicle = self._spawn_ego_vehicle()
            self._spawn_traffic(scenario.num_vehicles, scenario.num_pedestrians)
            
            # Initialize APACC controller
            controller = APACCController(ego_vehicle)
            
            # Setup sensors
            sensors = self._setup_sensors(ego_vehicle)
            
            # Run simulation
            metrics = self._run_simulation_loop(
                controller, 
                sensors, 
                scenario
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario.scenario_id}: {e}")
            return None
            
        finally:
            # Cleanup
            self._cleanup_scenario()
            
    def _setup_weather(self, weather_params):
        """Configure weather conditions"""
        weather = carla.WeatherParameters(
            cloudiness=weather_params['cloudiness'],
            precipitation=weather_params['precipitation'],
            sun_altitude_angle=weather_params['sun_altitude_angle'],
            fog_density=weather_params['fog_density'],
            wetness=weather_params['wetness']
        )
        self.world.set_weather(weather)
        
    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle with APACC control"""
        bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        
        vehicle = self.world.spawn_actor(bp, spawn_point)
        return vehicle
        
    def _spawn_traffic(self, num_vehicles, num_pedestrians):
        """Spawn traffic participants"""
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Spawn vehicles
        for i in range(min(num_vehicles, len(spawn_points)-1)):
            bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            spawn_point = spawn_points[i+1]
            
            vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                
        # Spawn pedestrians
        for _ in range(num_pedestrians):
            bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
            spawn_point = carla.Transform(
                carla.Location(
                    x=random.uniform(-100, 100),
                    y=random.uniform(-100, 100),
                    z=1.0
                )
            )
            
            walker = self.world.try_spawn_actor(bp, spawn_point)
            if walker:
                # Add walker AI controller
                walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
                self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                
    def _setup_sensors(self, vehicle):
        """Attach sensors to vehicle"""
        sensors = {}
        
        # Camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        sensors['camera'] = camera
        
        # LiDAR
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        sensors['lidar'] = lidar
        
        # Collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        sensors['collision'] = collision
        
        return sensors
        
    def _run_simulation_loop(self, controller, sensors, scenario):
        """Main simulation loop"""
        metrics_list = []
        collision_detected = False
        
        # Setup collision callback
        def on_collision(event):
            nonlocal collision_detected
            collision_detected = True
            
        sensors['collision'].listen(on_collision)
        
        # Run for specified duration
        start_time = time.time()
        while (time.time() - start_time) < scenario.duration_seconds:
            # Get sensor data
            sensor_data = self._collect_sensor_data(sensors)
            
            # Inject faults if specified
            self._inject_faults(scenario.fault_injections, sensor_data)
            
            # Execute control
            control_metrics = controller.control_step(sensor_data)
            
            # Collect metrics
            metrics = SimulationMetrics(
                scenario_id=scenario.scenario_id,
                timestamp=time.time() - start_time,
                control_loop_latency_ms=control_metrics['latency_ms'],
                safety_violations=self._check_safety_violations(sensor_data),
                collision_occurred=collision_detected,
                ttc_min=self._calculate_ttc(sensor_data),
                lateral_deviation_max=self._calculate_lateral_deviation(),
                acceleration_jerk_max=self._calculate_jerk(),
                rule_activations=control_metrics.get('rule_activations', {}),
                mpc_optimization_time_ms=control_metrics['mpc_time_ms'],
                fuzzy_inference_time_ms=control_metrics['fuzzy_time_ms'],
                total_distance_traveled=0.0,  # Calculate from odometry
                scenario_success=not collision_detected
            )
            
            metrics_list.append(metrics)
            
            # Synchronize with simulator
            self.world.tick()
            
        return metrics_list
        
    def _collect_sensor_data(self, sensors):
        """Aggregate sensor data"""
        # Simplified - expand based on actual sensor processing
        ego_transform = sensors['camera'].parent.get_transform()
        ego_velocity = sensors['camera'].parent.get_velocity()
        
        # Get nearby vehicles and pedestrians
        vehicles = []
        pedestrians = []
        
        for actor in self.world.get_actors():
            if actor.type_id.startswith('vehicle.'):
                if actor.id != sensors['camera'].parent.id:
                    vehicles.append({
                        'x': actor.get_location().x,
                        'y': actor.get_location().y,
                        'vx': actor.get_velocity().x,
                        'vy': actor.get_velocity().y
                    })
            elif actor.type_id.startswith('walker.pedestrian.'):
                pedestrians.append({
                    'x': actor.get_location().x,
                    'y': actor.get_location().y
                })
                
        return {
            'ego_x': ego_transform.location.x,
            'ego_y': ego_transform.location.y,
            'ego_vx': ego_velocity.x,
            'ego_vy': ego_velocity.y,
            'vehicles': vehicles,
            'pedestrians': pedestrians
        }
        
    def _inject_faults(self, fault_list, sensor_data):
        """Inject faults into sensor data"""
        for fault in fault_list:
            if fault['type'] == 'sensor_failure':
                # Simulate sensor failure by removing data
                if fault['sensor'] == 'lidar':
                    sensor_data['vehicles'] = []
                elif fault['sensor'] == 'camera':
                    sensor_data['pedestrians'] = []
                    
            elif fault['type'] == 'comm_delay':
                # Simulate delay (simplified)
                time.sleep(fault['delay_ms'] / 1000.0)
                
    def _check_safety_violations(self, sensor_data):
        """Count safety violations"""
        violations = 0
        
        # Check minimum distances
        for vehicle in sensor_data.get('vehicles', []):
            dist = np.sqrt((vehicle['x'] - sensor_data['ego_x'])**2 + 
                          (vehicle['y'] - sensor_data['ego_y'])**2)
            if dist < 2.0:  # 2 meter minimum
                violations += 1
                
        return violations
        
    def _calculate_ttc(self, sensor_data):
        """Calculate minimum time-to-collision"""
        min_ttc = float('inf')
        
        ego_pos = np.array([sensor_data['ego_x'], sensor_data['ego_y']])
        ego_vel = np.array([sensor_data['ego_vx'], sensor_data['ego_vy']])
        
        for vehicle in sensor_data.get('vehicles', []):
            other_pos = np.array([vehicle['x'], vehicle['y']])
            other_vel = np.array([vehicle['vx'], vehicle['vy']])
            
            rel_pos = other_pos - ego_pos
            rel_vel = other_vel - ego_vel
            
            # Simple TTC calculation
            if np.dot(rel_pos, rel_vel) < 0:  # Approaching
                dist = np.linalg.norm(rel_pos)
                speed = np.linalg.norm(rel_vel)
                if speed > 0:
                    ttc = dist / speed
                    min_ttc = min(min_ttc, ttc)
                    
        return min_ttc
        
    def _calculate_lateral_deviation(self):
        """Calculate lateral deviation from lane center"""
        # Placeholder - implement based on lane detection
        return random.uniform(0, 0.5)
        
    def _calculate_jerk(self):
        """Calculate maximum jerk"""
        # Placeholder - implement based on acceleration history
        return random.uniform(0, 2.0)
        
    def _cleanup_scenario(self):
        """Clean up actors after scenario"""
        for actor in self.world.get_actors():
            if actor.type_id.startswith('vehicle.') or \
               actor.type_id.startswith('walker.') or \
               actor.type_id.startswith('sensor.'):
                actor.destroy()
                
    def run_monte_carlo(self, num_scenarios=10000, parallel_runs=4):
        """Run full Monte Carlo simulation"""
        logger.info(f"Starting Monte Carlo simulation with {num_scenarios} scenarios")
        
        # Generate scenarios
        scenarios = self.generate_scenarios(num_scenarios)
        
        # Save scenario configurations
        with open('scenarios.json', 'w') as f:
            json.dump([asdict(s) for s in scenarios], f, indent=2)
            
        # Run scenarios (with parallelization for multiple CARLA instances)
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
            # Note: This assumes multiple CARLA servers on different ports
            futures = []
            
            for i, scenario in enumerate(scenarios):
                # Distribute across different CARLA instances
                port = 2000 + (i % parallel_runs)
                future = executor.submit(self._run_scenario_wrapper, scenario, port)
                futures.append(future)
                
            # Collect results
            for i, future in enumerate(futures):
                if i % 100 == 0:
                    logger.info(f"Progress: {i}/{num_scenarios} scenarios completed")
                    
                result = future.result()
                if result:
                    all_metrics.extend(result)
                    
        # Save results
        self._save_results(all_metrics)
        
        return all_metrics
        
    def _run_scenario_wrapper(self, scenario, port):
        """Wrapper to run scenario on specific CARLA instance"""
        try:
            sim = MonteCarloSimulator(carla_port=port)
            return sim.run_scenario(scenario)
        except Exception as e:
            logger.error(f"Error in scenario wrapper: {e}")
            return None
            
    def _save_results(self, metrics):
        """Save simulation results"""
        # Convert to DataFrame for analysis
        data = []
        for metric in metrics:
            data.append(asdict(metric))
            
        df = pd.DataFrame(data)
        
        # Save as CSV
        df.to_csv(f'monte_carlo_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', 
                  index=False)
        
        # Save summary statistics
        summary = {
            'total_scenarios': df['scenario_id'].nunique(),
            'total_collisions': df['collision_occurred'].sum(),
            'collision_rate': df['collision_occurred'].mean(),
            'avg_control_latency_ms': df['control_loop_latency_ms'].mean(),
            'max_control_latency_ms': df['control_loop_latency_ms'].max(),
            'safety_violation_rate': (df['safety_violations'] > 0).mean(),
            'success_rate': df['scenario_success'].mean()
        }
        
        with open('simulation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Results saved. Summary: {summary}")

# Main execution
if __name__ == "__main__":
    # Configure simulation parameters
    NUM_SCENARIOS = 10000
    PARALLEL_RUNS = 4  # Number of parallel CARLA instances
    
    # Initialize simulator
    simulator = MonteCarloSimulator()
    
    # Run Monte Carlo simulation
    results = simulator.run_monte_carlo(
        num_scenarios=NUM_SCENARIOS,
        parallel_runs=PARALLEL_RUNS
    )
    
    logger.info("Monte Carlo simulation completed!")