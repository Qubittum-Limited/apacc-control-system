"""
CARLA High-Fidelity Physics Simulation Module

Integrates with CARLA simulator for realistic sensor simulation
and physics-based validation of autonomous controllers.
"""

import carla
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import yaml
import logging
import time
from dataclasses import dataclass
from collections import deque
import queue

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Container for multi-modal sensor data"""
    timestamp: float
    rgb_images: List[np.ndarray]
    semantic_images: List[np.ndarray]
    lidar_points: np.ndarray
    radar_detections: List[Dict]
    vehicle_state: Dict[str, float]


class CarlaSimulator:
    """
    CARLA simulation wrapper for high-fidelity testing
    
    Provides synchronized sensor data collection and 
    physics-based validation of control algorithms.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize CARLA simulator connection
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}
        self.sensor_queues = {}
        self._sensor_data_buffer = deque(maxlen=100)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate CARLA configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required = ['carla_server', 'world', 'sensors', 'traffic']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
        
    def connect(self):
        """Establish connection to CARLA server"""
        try:
            self.client = carla.Client(
                self.config['carla_server']['host'],
                self.config['carla_server']['port']
            )
            self.client.set_timeout(self.config['carla_server']['timeout'])
            
            # Load world
            self.world = self.client.load_world(self.config['world']['map'])
            
            # Configure synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = self.config['carla_server']['synchronous_mode']
            settings.fixed_delta_seconds = self.config['carla_server']['fixed_delta_seconds']
            self.world.apply_settings(settings)
            
            logger.info(f"Connected to CARLA server at "
                       f"{self.config['carla_server']['host']}:"
                       f"{self.config['carla_server']['port']}")
                       
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {str(e)}")
            raise
            
    def spawn_ego_vehicle(self, spawn_point: Optional[carla.Transform] = None) -> carla.Vehicle:
        """Spawn ego vehicle with sensors"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Get vehicle blueprint
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Get spawn point
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)
            
        # Spawn vehicle
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"Spawned ego vehicle at {spawn_point.location}")
        
        # Attach sensors
        self._attach_sensors()
        
        return self.ego_vehicle
        
    def _attach_sensors(self):
        """Attach and configure sensors according to config"""
        blueprint_library = self.world.get_blueprint_library()
        
        # RGB Cameras
        for cam_config in self.config['sensors']['cameras']:
            if cam_config['type'] == 'sensor.camera.rgb':
                self._attach_rgb_camera(cam_config, blueprint_library)
            elif cam_config['type'] == 'sensor.camera.semantic_segmentation':
                self._attach_semantic_camera(cam_config, blueprint_library)
                
        # LiDAR
        lidar_config = self.config['sensors']['lidar']
        self._attach_lidar(lidar_config, blueprint_library)
        
        # Radar
        for radar_config in self.config['sensors']['radar']:
            self._attach_radar(radar_config, blueprint_library)
            
    def _attach_rgb_camera(self, config: Dict, blueprint_library):
        """Attach RGB camera sensor"""
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config['resolution'][0]))
        camera_bp.set_attribute('image_size_y', str(config['resolution'][1]))
        camera_bp.set_attribute('fov', str(config['fov']))
        
        transform = carla.Transform(
            carla.Location(x=config['position'][0], 
                          y=config['position'][1], 
                          z=config['position'][2]),
            carla.Rotation(pitch=config['rotation'][0], 
                         yaw=config['rotation'][1], 
                         roll=config['rotation'][2])
        )
        
        camera = self.world.spawn_actor(camera_bp, transform, 
                                       attach_to=self.ego_vehicle)
        
        # Create queue for this sensor
        q = queue.Queue()
        self.sensor_queues[f'rgb_{len(self.sensors)}'] = q
        camera.listen(lambda data: q.put(data))
        
        self.sensors[f'rgb_{len(self.sensors)}'] = camera
        logger.info(f"Attached RGB camera at position {config['position']}")
        
    def _attach_semantic_camera(self, config: Dict, blueprint_library):
        """Attach semantic segmentation camera"""
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(config['resolution'][0]))
        camera_bp.set_attribute('image_size_y', str(config['resolution'][1]))
        camera_bp.set_attribute('fov', str(config['fov']))
        
        transform = carla.Transform(
            carla.Location(x=config['position'][0], 
                          y=config['position'][1], 
                          z=config['position'][2]),
            carla.Rotation(pitch=config['rotation'][0], 
                         yaw=config['rotation'][1], 
                         roll=config['rotation'][2])
        )
        
        camera = self.world.spawn_actor(camera_bp, transform, 
                                       attach_to=self.ego_vehicle)
        
        q = queue.Queue()
        self.sensor_queues[f'semantic_{len(self.sensors)}'] = q
        camera.listen(lambda data: q.put(data))
        
        self.sensors[f'semantic_{len(self.sensors)}'] = camera
        logger.info(f"Attached semantic camera at position {config['position']}")
        
    def _attach_lidar(self, config: Dict, blueprint_library):
        """Attach LiDAR sensor"""
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(config['channels']))
        lidar_bp.set_attribute('range', str(config['range']))
        lidar_bp.set_attribute('rotation_frequency', str(config['rotation_frequency']))
        lidar_bp.set_attribute('points_per_second', str(config['points_per_second']))
        
        transform = carla.Transform(
            carla.Location(x=config['position'][0], 
                          y=config['position'][1], 
                          z=config['position'][2]),
            carla.Rotation(pitch=config['rotation'][0], 
                         yaw=config['rotation'][1], 
                         roll=config['rotation'][2])
        )
        
        lidar = self.world.spawn_actor(lidar_bp, transform, 
                                      attach_to=self.ego_vehicle)
        
        q = queue.Queue()
        self.sensor_queues['lidar'] = q
        lidar.listen(lambda data: q.put(data))
        
        self.sensors['lidar'] = lidar
        logger.info(f"Attached LiDAR with {config['channels']} channels")
        
    def _attach_radar(self, config: Dict, blueprint_library):
        """Attach radar sensor"""
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(config['horizontal_fov']))
        radar_bp.set_attribute('vertical_fov', str(config['vertical_fov']))
        
        transform = carla.Transform(
            carla.Location(x=config['position'][0], 
                          y=config['position'][1], 
                          z=config['position'][2]),
            carla.Rotation(pitch=config['rotation'][0], 
                         yaw=config['rotation'][1], 
                         roll=config['rotation'][2])
        )
        
        radar = self.world.spawn_actor(radar_bp, transform, 
                                      attach_to=self.ego_vehicle)
        
        q = queue.Queue()
        self.sensor_queues[f'radar_{len(self.sensors)}'] = q
        radar.listen(lambda data: q.put(data))
        
        self.sensors[f'radar_{len(self.sensors)}'] = radar
        logger.info(f"Attached radar at position {config['position']}")
        
    def spawn_traffic(self):
        """Spawn NPC vehicles and pedestrians"""
        # Spawn vehicles
        num_vehicles = self.config['traffic']['number_of_vehicles']
        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
        
        spawn_points = self.world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)
        
        for i in range(min(num_vehicles, len(spawn_points)-1)):
            blueprint = np.random.choice(vehicle_bps)
            spawn_point = spawn_points[i+1]  # Skip ego vehicle spawn
            
            npc = self.world.try_spawn_actor(blueprint, spawn_point)
            if npc:
                npc.set_autopilot(True)
                
        logger.info(f"Spawned {num_vehicles} NPC vehicles")
        
        # TODO: Add pedestrian spawning when CARLA API supports it better
        
    def set_weather(self, weather_preset: Optional[str] = None):
        """Set weather conditions"""
        if weather_preset is None:
            # Random selection based on probabilities
            weather_configs = self.config['world']['weather_presets']
            probs = [w['probability'] for w in weather_configs]
            weather_preset = np.random.choice(
                [w['preset'] for w in weather_configs],
                p=probs
            )
            
        # Apply weather
        weather = getattr(carla.WeatherParameters, weather_preset)
        self.world.set_weather(weather)
        logger.info(f"Set weather to {weather_preset}")
        
    def collect_sensor_data(self, timeout: float = 1.0) -> SensorData:
        """Collect synchronized sensor data from all sensors"""
        sensor_data = {
            'rgb_images': [],
            'semantic_images': [],
            'lidar_points': None,
            'radar_detections': []
        }
        
        # Collect from all sensor queues
        for sensor_name, sensor_queue in self.sensor_queues.items():
            try:
                data = sensor_queue.get(timeout=timeout)
                
                if 'rgb' in sensor_name:
                    # Convert to numpy array
                    array = np.frombuffer(data.raw_data, dtype=np.uint8)
                    array = array.reshape((data.height, data.width, 4))[:, :, :3]
                    sensor_data['rgb_images'].append(array)
                    
                elif 'semantic' in sensor_name:
                    array = np.frombuffer(data.raw_data, dtype=np.uint8)
                    array = array.reshape((data.height, data.width, 4))[:, :, 2]
                    sensor_data['semantic_images'].append(array)
                    
                elif sensor_name == 'lidar':
                    points = np.frombuffer(data.raw_data, dtype=np.float32)
                    points = points.reshape([-1, 4])  # x, y, z, intensity
                    sensor_data['lidar_points'] = points
                    
                elif 'radar' in sensor_name:
                    detections = []
                    for detection in data:
                        detections.append({
                            'velocity': detection.velocity,
                            'azimuth': detection.azimuth,
                            'altitude': detection.altitude,
                            'depth': detection.depth
                        })
                    sensor_data['radar_detections'].extend(detections)
                    
            except queue.Empty:
                logger.warning(f"Timeout collecting data from {sensor_name}")
                
        # Get vehicle state
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        
        vehicle_state = {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'roll': transform.rotation.roll,
            'vx': velocity.x,
            'vy': velocity.y,
            'vz': velocity.z
        }
        
        return SensorData(
            timestamp=self.world.get_snapshot().timestamp.elapsed_seconds,
            rgb_images=sensor_data['rgb_images'],
            semantic_images=sensor_data['semantic_images'],
            lidar_points=sensor_data['lidar_points'],
            radar_detections=sensor_data['radar_detections'],
            vehicle_state=vehicle_state
        )
        
    def apply_control(self, control: carla.VehicleControl):
        """Apply control command to ego vehicle"""
        self.ego_vehicle.apply_control(control)
        
    def tick(self):
        """Advance simulation by one timestep"""
        self.world.tick()
        
    def run_scenario(self, controller_func, duration: float = 120.0) -> Dict[str, Any]:
        """
        Run a complete scenario with given controller
        
        Args:
            controller_func: Function that takes SensorData and returns VehicleControl
            duration: Scenario duration in seconds
            
        Returns:
            Dictionary of metrics from the scenario
        """
        start_time = time.time()
        metrics = {
            'collision': False,
            'lane_violations': 0,
            'total_distance': 0.0,
            'avg_speed': 0.0,
            'control_latencies': []
        }
        
        # Spawn traffic
        self.spawn_traffic()
        
        # Set random weather
        self.set_weather()
        
        # Main control loop
        prev_location = self.ego_vehicle.get_location()
        speeds = []
        
        while time.time() - start_time < duration:
            # Collect sensor data
            sensor_data = self.collect_sensor_data()
            
            # Get control from controller
            control_start = time.time()
            control = controller_func(sensor_data)
            control_latency = (time.time() - control_start) * 1000  # ms
            metrics['control_latencies'].append(control_latency)
            
            # Apply control
            self.apply_control(control)
            
            # Tick simulation
            self.tick()
            
            # Update metrics
            current_location = self.ego_vehicle.get_location()
            distance = current_location.distance(prev_location)
            metrics['total_distance'] += distance
            prev_location = current_location
            
            velocity = self.ego_vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2)
            speeds.append(speed)
            
            # Check for collisions
            if self._check_collision():
                metrics['collision'] = True
                logger.warning("Collision detected!")
                break
                
        # Calculate final metrics
        metrics['avg_speed'] = np.mean(speeds) if speeds else 0.0
        metrics['avg_control_latency'] = np.mean(metrics['control_latencies'])
        metrics['p99_control_latency'] = np.percentile(metrics['control_latencies'], 99)
        
        # Clean up latency list to save space
        del metrics['control_latencies']
        
        return metrics
        
    def _check_collision(self) -> bool:
        """Check if ego vehicle has collided"""
        # Simple collision check using actor distance
        # In production, use collision sensor
        for actor in self.world.get_actors():
            if actor.id == self.ego_vehicle.id:
                continue
            if actor.get_location().distance(self.ego_vehicle.get_location()) < 2.0:
                return True
        return False
        
    def cleanup(self):
        """Clean up spawned actors and sensors"""
        # Destroy sensors
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.destroy()
                
        # Destroy ego vehicle
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            
        # Reset to asynchronous mode
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
        logger.info("Cleaned up CARLA simulation")