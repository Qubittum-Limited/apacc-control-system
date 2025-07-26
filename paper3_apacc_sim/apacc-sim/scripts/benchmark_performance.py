#!/usr/bin/env python3
"""
Benchmark Performance Script

Measures computational performance and resource usage
of controllers during validation.
"""

import argparse
import logging
import sys
import time
import psutil
import GPUtil
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.validate_controller import load_controller
from modules.monte_carlo import MonteCarloSimulator


@dataclass
class PerformanceSample:
    """Single performance measurement"""
    timestamp: float
    scenario_id: int
    control_latency_ms: float
    cpu_percent: float
    memory_mb: float
    gpu_percent: float
    gpu_memory_mb: float
    

class PerformanceBenchmark:
    """
    Benchmark controller performance and resource usage
    """
    
    def __init__(self):
        """Initialize benchmark tracker"""
        self.samples: List[PerformanceSample] = []
        self.process = psutil.Process()
        self.has_gpu = len(GPUtil.getGPUs()) > 0
        
    def measure_controller_performance(self, controller, num_scenarios: int = 100):
        """
        Run performance benchmark on controller
        
        Args:
            controller: Controller to benchmark
            num_scenarios: Number of scenarios to test
        """
        logging.info(f"Starting performance benchmark with {num_scenarios} scenarios")
        
        # Initialize Monte Carlo for consistent scenarios
        mc_sim = MonteCarloSimulator('configs/monte_carlo/default.yaml')
        
        # Warmup
        logging.info("Running warmup...")
        for i in range(10):
            scenario = mc_sim.generate_scenario(i)
            sensor_data = self._scenario_to_sensor_data(scenario)
            _ = controller(sensor_data)
        
        # Main benchmark loop
        logging.info("Running benchmark...")
        for i in range(num_scenarios):
            scenario = mc_sim.generate_scenario(i)
            sensor_data = self._scenario_to_sensor_data(scenario)
            
            # Measure before
            cpu_before = self.process.cpu_percent(interval=0)
            mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
            
            if self.has_gpu:
                gpu = GPUtil.getGPUs()[0]
                gpu_before = gpu.load * 100
                gpu_mem_before = gpu.memoryUsed
            else:
                gpu_before = 0
                gpu_mem_before = 0
            
            # Time controller execution
            start_time = time.perf_counter()
            _ = controller(sensor_data)
            control_latency = (time.perf_counter() - start_time) * 1000  # ms
            
            # Measure after
            cpu_after = self.process.cpu_percent(interval=0)
            mem_after = self.process.memory_info().rss / 1024 / 1024
            
            if self.has_gpu:
                gpu = GPUtil.getGPUs()[0]
                gpu_after = gpu.load * 100
                gpu_mem_after = gpu.memoryUsed
            else:
                gpu_after = 0
                gpu_mem_after = 0
            
            # Record sample
            sample = PerformanceSample(
                timestamp=time.time(),
                scenario_id=i,
                control_latency_ms=control_latency,
                cpu_percent=max(cpu_after, cpu_before),  # Take peak
                memory_mb=mem_after,
                gpu_percent=max(gpu_after, gpu_before),
                gpu_memory_mb=gpu_mem_after
            )
            
            self.samples.append(sample)
            
            # Progress update
            if (i + 1) % 20 == 0:
                logging.info(f"Progress: {i + 1}/{num_scenarios}")
                self._print_current_stats()
    
    def _scenario_to_sensor_data(self, scenario) -> Dict[str, Any]:
        """Convert scenario to sensor data format"""
        # Simplified conversion for benchmarking
        return {
            'timestamp': time.time(),
            'lidar_points': np.random.randn(1000, 4),  # Dummy LiDAR
            'rgb_images': [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)],
            'vehicle_state': {
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'vx': 10.0, 'vy': 0.0, 'vz': 0.0
            },
            'nearby_vehicles': []
        }
    
    def _print_current_stats(self):
        """Print current performance statistics"""
        latencies = [s.control_latency_ms for s in self.samples[-20:]]
        logging.info(f"  Avg latency: {np.mean(latencies):.2f} ms")
        logging.info(f"  CPU usage: {self.samples[-1].cpu_percent:.1f}%")
        logging.info(f"  Memory: {self.samples[-1].memory_mb:.1f} MB")
        if self.has_gpu:
            logging.info(f"  GPU usage: {self.samples[-1].gpu_percent:.1f}%")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results"""
        latencies = [s.control_latency_ms for s in self.samples]
        cpu_usage = [s.cpu_percent for s in self.samples]
        memory_usage = [s.memory_mb for s in self.samples]
        
        analysis = {
            'num_samples': len(self.samples),
            'latency': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            },
            'cpu': {
                'mean': np.mean(cpu_usage),
                'max': np.max(cpu_usage)
            },
            'memory': {
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage),
                'growth': memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
            }
        }
        
        if self.has_gpu:
            gpu_usage = [s.gpu_percent for s in self.samples]
            gpu_memory = [s.gpu_memory_mb for s in self.samples]
            
            analysis['gpu'] = {
                'mean': np.mean(gpu_usage),
                'max': np.max(gpu_usage)
            }
            analysis['gpu_memory'] = {
                'mean': np.mean(gpu_memory),
                'max': np.max(gpu_memory)
            }
        
        # Real-time compliance
        analysis['realtime_compliance'] = {
            '10ms': np.sum(np.array(latencies) < 10) / len(latencies) * 100,
            '20ms': np.sum(np.array(latencies) < 20) / len(latencies) * 100,
            '100ms': np.sum(np.array(latencies) < 100) / len(latencies) * 100
        }
        
        return analysis
    
    def save_results(self, output_path: Path):
        """Save benchmark results"""
        # Convert samples to dict
        samples_dict = [asdict(s) for s in self.samples]
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Create output
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_scenarios': len(self.samples),
                'has_gpu': self.has_gpu
            },
            'analysis': analysis,
            'samples': samples_dict
        }
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
        
        # Print summary
        self.print_summary(analysis)
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\nControl Latency:")
        print(f"  Mean: {analysis['latency']['mean']:.2f} ms")
        print(f"  Std:  {analysis['latency']['std']:.2f} ms")
        print(f"  P95:  {analysis['latency']['p95']:.2f} ms")
        print(f"  P99:  {analysis['latency']['p99']:.2f} ms")
        print(f"  Max:  {analysis['latency']['max']:.2f} ms")
        
        print(f"\nResource Usage:")
        print(f"  CPU (mean): {analysis['cpu']['mean']:.1f}%")
        print(f"  CPU (max):  {analysis['cpu']['max']:.1f}%")
        print(f"  Memory (mean): {analysis['memory']['mean']:.1f} MB")
        print(f"  Memory (max):  {analysis['memory']['max']:.1f} MB")
        
        if self.has_gpu:
            print(f"  GPU (mean): {analysis['gpu']['mean']:.1f}%")
            print(f"  GPU (max):  {analysis['gpu']['max']:.1f}%")
        
        print(f"\nReal-time Compliance:")
        print(f"  <10ms:  {analysis['realtime_compliance']['10ms']:.1f}%")
        print(f"  <20ms:  {analysis['realtime_compliance']['20ms']:.1f}%")
        print(f"  <100ms: {analysis['realtime_compliance']['100ms']:.1f}%")
        print("="*60)


def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(
        description="Benchmark controller performance and resource usage"
    )
    
    parser.add_argument(
        '--controller', '-c',
        required=True,
        help='Path to controller implementation'
    )
    
    parser.add_argument(
        '--scenarios', '-n',
        type=int,
        default=100,
        help='Number of scenarios to benchmark (default: 100)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load controller
    logging.info(f"Loading controller from {args.controller}")
    try:
        controller = load_controller(args.controller)
    except Exception as e:
        logging.error(f"Failed to load controller: {e}")
        return 1
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    
    try:
        benchmark.measure_controller_performance(controller, args.scenarios)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"benchmark_{Path(args.controller).stem}_{timestamp}.json")
        
        benchmark.save_results(output_path)
        
        return 0
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())