"""
SUMO Large-Scale Traffic Simulation Module

Integrates with SUMO (Simulation of Urban Mobility) for
large-scale traffic flow validation and multi-agent scenarios.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import yaml
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ImportError("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

logger = logging.getLogger(__name__)


class SumoSimulator:
    """
    SUMO traffic simulation wrapper for large-scale validation
    
    Provides TraCI API integration for multi-agent traffic scenarios
    and traffic flow analysis.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize SUMO simulator
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.net = None
        self.ego_vehicles = {}
        self.metrics_buffer = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate SUMO configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required = ['sumo_configuration', 'network', 'traffic_flow']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        return config
        
    def generate_network(self, output_dir: Path):
        """Generate SUMO network based on configuration"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        network_config = self.config['network']
        
        if network_config['type'] == 'grid':
            self._generate_grid_network(output_dir, network_config)
        elif network_config['type'] == 'spider':
            self._generate_spider_network(output_dir, network_config)
        else:
            raise ValueError(f"Unknown network type: {network_config['type']}")
            
        logger.info(f"Generated {network_config['type']} network in {output_dir}")
        
    def _generate_grid_network(self, output_dir: Path, config: Dict):
        """Generate grid network using netgenerate"""
        cmd = [
            'netgenerate',
            '--grid',
            '--grid.x-number', '10',
            '--grid.y-number', '10',
            '--grid.x-length', str(config['size'][0] / 10),
            '--grid.y-length', str(config['size'][1] / 10),
            '--default.lanenumber', str(config['lanes_per_road']),
            '--default.speed', str(config['speed_limit'] / 3.6),  # km/h to m/s
            '--tls.guess', 'true' if config['traffic_lights'] else 'false',
            '--output-file', str(output_dir / 'network.net.xml')
        ]
        
        import subprocess
        subprocess.run(cmd, check=True)
        
        # Load generated network
        self.net = sumolib.net.readNet(str(output_dir / 'network.net.xml'))
        
    def _generate_spider_network(self, output_dir: Path, config: Dict):
        """Generate spider/radial network"""
        # TODO: Implement spider network generation
        raise NotImplementedError("Spider network generation not yet implemented")
        
    def generate_traffic_demand(self, output_dir: Path):
        """Generate traffic demand (routes) based on configuration"""
        traffic_config = self.config['traffic_flow']
        
        # Generate random trips
        cmd = [
            'randomTrips.py',
            '-n', str(output_dir / 'network.net.xml'),
            '-r', str(output_dir / 'routes.rou.xml'),
            '--begin', '0',
            '--end', '3600',  # 1 hour simulation
            '--period', str(traffic_config['route_generation']['period']),
            '--seed', str(self.config['sumo_configuration']['seed'])
        ]
        
        # Add vehicle type probabilities
        for vtype in traffic_config['vehicle_types']:
            cmd.extend(['--vehicle-class', vtype['id']])
            
        import subprocess
        subprocess.run(cmd, check=True)
        
        # Post-process routes to add vehicle types
        self._add_vehicle_types(output_dir / 'routes.rou.xml', traffic_config)
        
        logger.info(f"Generated traffic demand in {output_dir}")
        
    def _add_vehicle_types(self, route_file: Path, traffic_config: Dict):
        """Add vehicle type definitions to route file"""
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        # Add vehicle types at the beginning
        for vtype in traffic_config['vehicle_types']:
            vtype_elem = ET.Element('vType', attrib={
                'id': vtype['id'],
                'length': str(vtype['length']),
                'maxSpeed': str(vtype['max_speed'] / 3.6),  # km/h to m/s
                'accel': str(vtype['accel']),
                'decel': str(vtype['decel'])
            })
            root.insert(0, vtype_elem)
            
        tree.write(route_file)
        
    def start_simulation(self, config_file: Optional[str] = None, gui: bool = False):
        """Start SUMO simulation with TraCI"""
        sumo_binary = 'sumo-gui' if gui else self.config['sumo_configuration']['binary']
        
        # Build command
        sumo_cmd = [
            sumo_binary,
            '--net-file', 'network.net.xml',
            '--route-files', 'routes.rou.xml',
            '--step-length', str(self.config['sumo_configuration']['step_length']),
            '--seed', str(self.config['sumo_configuration']['seed']),
            '--no-warnings', str(self.config['sumo_configuration']['no_warnings'])
        ]
        
        if config_file:
            sumo_cmd.extend(['-c', config_file])
            
        # Start TraCI
        traci.start(sumo_cmd)
        logger.info("Started SUMO simulation with TraCI")
        
    def spawn_ego_vehicles(self, controller_configs: List[Dict]) -> List[str]:
        """Spawn ego vehicles that will be controlled"""
        ego_ids = []
        
        for i, config in enumerate(controller_configs):
            ego_id = f"ego_{i}"
            
            # Get a valid route
            route_ids = traci.route.getIDList()
            if not route_ids:
                raise RuntimeError("No routes available")
                
            route_id = route_ids[i % len(route_ids)]
            
            # Add vehicle
            traci.vehicle.add(
                vehID=ego_id,
                routeID=route_id,
                typeID='passenger',  # Default type
                depart='now'
            )
            
            # Set color for visualization
            traci.vehicle.setColor(ego_id, (255, 0, 0, 255))  # Red
            
            self.ego_vehicles[ego_id] = config
            ego_ids.append(ego_id)
            
        logger.info(f"Spawned {len(ego_ids)} ego vehicles")
        return ego_ids
        
    def get_vehicle_state(self, vehicle_id: str) -> Dict[str, Any]:
        """Get current state of a vehicle"""
        try:
            state = {
                'position': traci.vehicle.getPosition(vehicle_id),
                'speed': traci.vehicle.getSpeed(vehicle_id),
                'angle': traci.vehicle.getAngle(vehicle_id),
                'road_id': traci.vehicle.getRoadID(vehicle_id),
                'lane_index': traci.vehicle.getLaneIndex(vehicle_id),
                'distance': traci.vehicle.getDistance(vehicle_id)
            }
            
            # Get nearby vehicles
            nearby_vehicles = self._get_nearby_vehicles(vehicle_id, radius=50.0)
            state['nearby_vehicles'] = nearby_vehicles
            
            return state
            
        except traci.TraCIException as e:
            logger.error(f"Error getting state for {vehicle_id}: {str(e)}")
            return None
            
    def _get_nearby_vehicles(self, vehicle_id: str, radius: float) -> List[Dict]:
        """Get information about nearby vehicles"""
        ego_pos = traci.vehicle.getPosition(vehicle_id)
        nearby = []
        
        for veh_id in traci.vehicle.getIDList():
            if veh_id == vehicle_id:
                continue
                
            veh_pos = traci.vehicle.getPosition(veh_id)
            distance = np.sqrt((veh_pos[0] - ego_pos[0])**2 + 
                             (veh_pos[1] - ego_pos[1])**2)
            
            if distance <= radius:
                nearby.append({
                    'id': veh_id,
                    'distance': distance,
                    'position': veh_pos,
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'angle': traci.vehicle.getAngle(veh_id)
                })
                
        return sorted(nearby, key=lambda x: x['distance'])
        
    def apply_control(self, vehicle_id: str, control: Dict[str, float]):
        """Apply control commands to ego vehicle"""
        try:
            # SUMO uses different control interface than CARLA
            # Convert throttle/brake to acceleration
            if 'throttle' in control and 'brake' in control:
                if control['brake'] > 0:
                    acceleration = -control['brake'] * 5.0  # Max decel
                else:
                    acceleration = control['throttle'] * 3.0  # Max accel
                    
                target_speed = max(0, traci.vehicle.getSpeed(vehicle_id) + 
                                 acceleration * self.config['sumo_configuration']['step_length'])
                traci.vehicle.setSpeed(vehicle_id, target_speed)
                
            # Apply steering as lane change
            if 'steering' in control and abs(control['steering']) > 0.1:
                current_lane = traci.vehicle.getLaneIndex(vehicle_id)
                if control['steering'] > 0 and current_lane > 0:
                    traci.vehicle.changeLane(vehicle_id, current_lane - 1, 2.0)
                elif control['steering'] < 0:
                    # Check if right lane exists
                    num_lanes = traci.edge.getLaneNumber(traci.vehicle.getRoadID(vehicle_id))
                    if current_lane < num_lanes - 1:
                        traci.vehicle.changeLane(vehicle_id, current_lane + 1, 2.0)
                        
        except traci.TraCIException as e:
            logger.error(f"Error applying control to {vehicle_id}: {str(e)}")
            
    def step(self):
        """Advance simulation by one timestep"""
        traci.simulationStep()
        
        # Collect metrics
        self._collect_metrics()
        
    def _collect_metrics(self):
        """Collect traffic flow metrics"""
        metrics = {
            'time': traci.simulation.getTime(),
            'vehicle_count': traci.vehicle.getIDCount(),
            'departed': traci.simulation.getDepartedNumber(),
            'arrived': traci.simulation.getArrivedNumber(),
            'collisions': traci.simulation.getCollidingVehiclesNumber()
        }
        
        # Calculate average speed
        if metrics['vehicle_count'] > 0:
            speeds = [traci.vehicle.getSpeed(veh_id) 
                     for veh_id in traci.vehicle.getIDList()]
            metrics['avg_speed'] = np.mean(speeds) * 3.6  # m/s to km/h
        else:
            metrics['avg_speed'] = 0.0
            
        self.metrics_buffer.append(metrics)
        
    def run_scenario(self, controller_func, duration: float = 3600.0) -> Dict[str, Any]:
        """
        Run simulation scenario with given controller
        
        Args:
            controller_func: Function that takes vehicle state and returns control
            duration: Simulation duration in seconds
            
        Returns:
            Dictionary of aggregated metrics
        """
        start_time = traci.simulation.getTime()
        
        # Spawn ego vehicles
        ego_configs = [{}] * self.config['traffic_flow']['ego_vehicles']
        ego_ids = self.spawn_ego_vehicles(ego_configs)
        
        # Metrics tracking
        ego_metrics = {ego_id: {
            'distance': 0.0,
            'collisions': 0,
            'lane_changes': 0,
            'avg_speed': 0.0,
            'control_latencies': []
        } for ego_id in ego_ids}
        
        step_count = 0
        
        # Main simulation loop
        while traci.simulation.getTime() - start_time < duration:
            # Control ego vehicles
            for ego_id in ego_ids:
                if ego_id in traci.vehicle.getIDList():
                    # Get state
                    state = self.get_vehicle_state(ego_id)
                    
                    if state:
                        # Get control from controller
                        import time
                        control_start = time.time()
                        control = controller_func(state)
                        control_latency = (time.time() - control_start) * 1000
                        
                        ego_metrics[ego_id]['control_latencies'].append(control_latency)
                        
                        # Apply control
                        self.apply_control(ego_id, control)
                        
            # Step simulation
            self.step()
            step_count += 1
            
            # Log progress
            if step_count % 1000 == 0:
                current_time = traci.simulation.getTime()
                logger.info(f"Simulation time: {current_time:.1f}s / {duration}s")
                
        # Aggregate metrics
        results = {
            'total_vehicles': traci.simulation.getDepartedNumber(),
            'total_arrived': traci.simulation.getArrivedNumber(),
            'total_collisions': sum([m['collisions'] for m in self.metrics_buffer]),
            'avg_network_speed': np.mean([m['avg_speed'] for m in self.metrics_buffer])
        }
        
        # Ego vehicle specific metrics
        for ego_id, metrics in ego_metrics.items():
            if ego_id in traci.vehicle.getIDList():
                metrics['distance'] = traci.vehicle.getDistance(ego_id)
                metrics['avg_speed'] = metrics['distance'] / duration * 3.6
                
            metrics['avg_control_latency'] = np.mean(metrics['control_latencies'])
            metrics['p99_control_latency'] = np.percentile(metrics['control_latencies'], 99)
            
            # Add to results
            results[f'{ego_id}_distance'] = metrics['distance']
            results[f'{ego_id}_avg_speed'] = metrics['avg_speed']
            results[f'{ego_id}_avg_latency'] = metrics['avg_control_latency']
            
        return results
        
    def close(self):
        """Close TraCI connection"""
        traci.close()
        logger.info("Closed SUMO simulation")
        
    def export_metrics(self, output_path: Path):
        """Export collected metrics to file"""
        df = pd.DataFrame(self.metrics_buffer)
        
        output_path.parent.mkdir(exist_ok=True)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix == '.parquet':
            df.to_parquet(output_path, compression='snappy')
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
            
        logger.info(f"Exported metrics to {output_path}")