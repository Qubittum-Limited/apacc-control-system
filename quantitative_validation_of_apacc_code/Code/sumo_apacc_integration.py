#!/usr/bin/env python3
"""
SUMO-APACC Integration for Large-Scale Traffic Simulation
Focuses on multi-vehicle scenarios and traffic flow analysis
"""

import os
import sys
import traci
import sumolib
import numpy as np
import json
import random
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET

class SUMOAPACCSimulator:
    """SUMO integration for APACC validation"""
    
    def __init__(self, sumo_config_path: str):
        self.sumo_config = sumo_config_path
        self.step_length = 0.01  # 10ms for 100Hz control
        self.apacc_vehicles = {}
        
    def generate_random_network(self, num_intersections=20, grid_size=1000):
        """Generate random urban network"""
        nodes = []
        edges = []
        
        # Create grid of intersections
        for i in range(num_intersections):
            x = random.uniform(0, grid_size)
            y = random.uniform(0, grid_size)
            nodes.append({
                'id': f'n{i}',
                'x': x,
                'y': y,
                'type': 'traffic_light' if random.random() > 0.5 else 'priority'
            })
            
        # Connect nodes with edges
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                dist = np.sqrt((nodes[i]['x'] - nodes[j]['x'])**2 + 
                              (nodes[i]['y'] - nodes[j]['y'])**2)
                if dist < 200:  # Connect nearby nodes
                    edges.append({
                        'id': f'e{i}_{j}',
                        'from': nodes[i]['id'],
                        'to': nodes[j]['id'],
                        'numLanes': random.randint(1, 3),
                        'speed': random.choice([30, 50, 70]) / 3.6  # m/s
                    })
                    
        return nodes, edges
        
    def create_sumo_scenario(self, scenario_id: str, config: Dict):
        """Create SUMO network and route files"""
        # Generate network
        nodes, edges = self.generate_random_network()
        
        # Write node file
        node_root = ET.Element('nodes')
        for node in nodes:
            ET.SubElement(node_root, 'node', attrib={
                'id': node['id'],
                'x': str(node['x']),
                'y': str(node['y']),
                'type': node['type']
            })
            
        node_tree = ET.ElementTree(node_root)
        node_file = f'{scenario_id}.nod.xml'
        node_tree.write(node_file)
        
        # Write edge file
        edge_root = ET.Element('edges')
        for edge in edges:
            ET.SubElement(edge_root, 'edge', attrib={
                'id': edge['id'],
                'from': edge['from'],
                'to': edge['to'],
                'numLanes': str(edge['numLanes']),
                'speed': str(edge['speed'])
            })
            
        edge_tree = ET.ElementTree(edge_root)
        edge_file = f'{scenario_id}.edg.xml'
        edge_tree.write(edge_file)
        
        # Generate network
        net_file = f'{scenario_id}.net.xml'
        os.system(f'netconvert --node-files={node_file} --edge-files={edge_file} --output-file={net_file}')
        
        # Generate routes
        self._generate_routes(scenario_id, config['num_vehicles'], edges)
        
        # Create SUMO config
        self._create_sumo_config(scenario_id, net_file)
        
        return f'{scenario_id}.sumocfg'
        
    def _generate_routes(self, scenario_id: str, num_vehicles: int, edges: List):
        """Generate vehicle routes"""
        route_root = ET.Element('routes')
        
        # Vehicle types
        vtype_apacc = ET.SubElement(route_root, 'vType', attrib={
            'id': 'apacc_vehicle',
            'accel': '3.0',
            'decel': '6.0',
            'length': '4.5',
            'maxSpeed': '50.0',
            'color': '1,0,0'  # Red for APACC vehicles
        })
        
        vtype_normal = ET.SubElement(route_root, 'vType', attrib={
            'id': 'normal_vehicle',
            'accel': '2.5',
            'decel': '4.5',
            'length': '4.5',
            'maxSpeed': '40.0',
            'color': '0,1,0'  # Green for normal vehicles
        })
        
        # Generate vehicles
        for i in range(num_vehicles):
            # Random route
            route_edges = random.sample([e['id'] for e in edges], 
                                      min(5, len(edges)))
            route_id = f'route_{i}'
            
            route = ET.SubElement(route_root, 'route', attrib={
                'id': route_id,
                'edges': ' '.join(route_edges)
            })
            
            # Vehicle
            vehicle_type = 'apacc_vehicle' if i < 10 else 'normal_vehicle'
            vehicle = ET.SubElement(route_root, 'vehicle', attrib={
                'id': f'veh_{i}',
                'type': vehicle_type,
                'route': route_id,
                'depart': str(random.uniform(0, 100))
            })
            
        route_tree = ET.ElementTree(route_root)
        route_file = f'{scenario_id}.rou.xml'
        route_tree.write(route_file)
        
    def _create_sumo_config(self, scenario_id: str, net_file: str):
        """Create SUMO configuration file"""
        config_root = ET.Element('configuration')
        
        input_elem = ET.SubElement(config_root, 'input')
        ET.SubElement(input_elem, 'net-file', value=net_file)
        ET.SubElement(input_elem, 'route-files', value=f'{scenario_id}.rou.xml')
        
        time_elem = ET.SubElement(config_root, 'time')
        ET.SubElement(time_elem, 'begin', value='0')
        ET.SubElement(time_elem, 'end', value='300')  # 5 minutes
        ET.SubElement(time_elem, 'step-length', value=str(self.step_length))
        
        config_tree = ET.ElementTree(config_root)
        config_file = f'{scenario_id}.sumocfg'
        config_tree.write(config_file)
        
    def run_simulation(self, config_file: str):
        """Run SUMO simulation with APACC control"""
        # Start SUMO
        sumo_cmd = ['sumo', '-c', config_file, '--step-length', str(self.step_length)]
        traci.start(sumo_cmd)
        
        metrics = []
        
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                # Get current simulation time
                sim_time = traci.simulation.getTime()
                
                # Update APACC-controlled vehicles
                apacc_vehicles = [v for v in traci.vehicle.getIDList() 
                                 if traci.vehicle.getTypeID(v) == 'apacc_vehicle']
                
                for veh_id in apacc_vehicles:
                    # Get sensor data
                    sensor_data = self._get_vehicle_sensors(veh_id)
                    
                    # Run APACC control
                    control = self._apacc_control_step(veh_id, sensor_data)
                    
                    # Apply control
                    self._apply_control(veh_id, control)
                    
                    # Collect metrics
                    metrics.append({
                        'time': sim_time,
                        'vehicle_id': veh_id,
                        'position': traci.vehicle.getPosition(veh_id),
                        'speed': traci.vehicle.getSpeed(veh_id),
                        'acceleration': traci.vehicle.getAcceleration(veh_id),
                        'lane_position': traci.vehicle.getLateralLanePosition(veh_id),
                        'ttc': self._calculate_ttc(veh_id, sensor_data)
                    })
                    
                # Advance simulation
                traci.simulationStep()
                
        finally:
            traci.close()
            
        return metrics
        
    def _get_vehicle_sensors(self, veh_id: str):
        """Get sensor data for vehicle"""
        # Get nearby vehicles
        x, y = traci.vehicle.getPosition(veh_id)
        nearby_vehicles = []
        
        for other_id in traci.vehicle.getIDList():
            if other_id != veh_id:
                other_x, other_y = traci.vehicle.getPosition(other_id)
                dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                
                if dist < 50:  # 50m sensing range
                    nearby_vehicles.append({
                        'id': other_id,
                        'distance': dist,
                        'position': (other_x, other_y),
                        'speed': traci.vehicle.getSpeed(other_id),
                        'angle': traci.vehicle.getAngle(other_id)
                    })
                    
        return {
            'position': (x, y),
            'speed': traci.vehicle.getSpeed(veh_id),
            'angle': traci.vehicle.getAngle(veh_id),
            'nearby_vehicles': nearby_vehicles,
            'traffic_light': self._get_upcoming_traffic_light(veh_id)
        }
        
    def _get_upcoming_traffic_light(self, veh_id: str):
        """Get upcoming traffic light state"""
        tls = traci.vehicle.getNextTLS(veh_id)
        if tls:
            tl_id, tl_index, dist, state = tls[0]
            return {
                'distance': dist,
                'state': state
            }
        return None
        
    def _apacc_control_step(self, veh_id: str, sensor_data: Dict):
        """Simplified APACC control for SUMO"""
        # Target speed based on conditions
        target_speed = 13.89  # 50 km/h default
        
        # Check for nearby vehicles
        min_dist = float('inf')
        for vehicle in sensor_data['nearby_vehicles']:
            if vehicle['distance'] < min_dist:
                min_dist = vehicle['distance']
                
        # Adjust speed based on distance
        if min_dist < 10:
            target_speed = 5.0
        elif min_dist < 20:
            target_speed = 8.0
            
        # Check traffic light
        tl = sensor_data['traffic_light']
        if tl and tl['state'] in ['r', 'y'] and tl['distance'] < 30:
            target_speed = 0.0
            
        return {
            'speed': target_speed,
            'lane_change': 0  # No lane change for now
        }
        
    def _apply_control(self, veh_id: str, control: Dict):
        """Apply APACC control to vehicle"""
        traci.vehicle.setSpeed(veh_id, control['speed'])
        
        if control['lane_change'] != 0:
            current_lane = traci.vehicle.getLaneIndex(veh_id)
            target_lane = current_lane + control['lane_change']
            traci.vehicle.changeLane(veh_id, target_lane, 2.0)
            
    def _calculate_ttc(self, veh_id: str, sensor_data: Dict):
        """Calculate time-to-collision"""
        ego_speed = sensor_data['speed']
        min_ttc = float('inf')
        
        for vehicle in sensor_data['nearby_vehicles']:
            if vehicle['distance'] < 50:
                relative_speed = ego_speed - vehicle['speed']
                if relative_speed > 0:
                    ttc = vehicle['distance'] / relative_speed
                    min_ttc = min(min_ttc, ttc)
                    
        return min_ttc

# Example usage
if __name__ == "__main__":
    simulator = SUMOAPACCSimulator('config.sumocfg')
    
    # Generate test scenario
    scenario_config = {
        'num_vehicles': 100,
        'num_intersections': 15
    }
    
    config_file = simulator.create_sumo_scenario('test_scenario', scenario_config)
    
    # Run simulation
    metrics = simulator.run_simulation(config_file)
    
    print(f"Simulation completed with {len(metrics)} data points")