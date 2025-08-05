#!/usr/bin/env python3
"""
generate_figures_apaccsim.py

Generates publication-quality figures from APACC-Sim validation results
Recreates all figures from Paper 3 and Chapter 5

Author: George Frangou
Institution: Cranfield University
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime

# Scientific computing imports
import numpy as np
import pandas as pd
import h5py

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns

# Optional Plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be skipped.")

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Color palette for consistency
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#E63946',
    'tertiary': '#F77F00',
    'quaternary': '#06D6A0',
    'dark': '#264653',
    'light': '#F1FAEE',
    'grid': '#CCCCCC'
}

# Controller colors for comparisons
CONTROLLER_COLORS = {
    'APACC': COLORS['primary'],
    'MPC': COLORS['secondary'],
    'DRL': COLORS['tertiary'],
    'PID': COLORS['quaternary']
}


class FigureGenerator:
    """Generates all figures for APACC-Sim paper"""
    
    def __init__(self, data_dir: str = './results', output_dir: str = './figures',
                 formats: List[str] = ['png', 'svg', 'pdf']):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.formats = formats
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.metrics_data = None
        self.controller_data = None
        self.latency_data = None
        self.explainability_data = None
        
    def load_data(self, hdf5_file: Optional[str] = None):
        """Load validation data from HDF5 or generate synthetic data"""
        if hdf5_file and Path(hdf5_file).exists():
            print(f"Loading data from {hdf5_file}")
            with h5py.File(hdf5_file, 'r') as f:
                # Load actual data structure
                self.metrics_data = {
                    'collision_rates': f['metrics/collision_rates'][:],
                    'ttc_values': f['metrics/time_to_collision'][:],
                    'lane_deviations': f['metrics/lane_deviations'][:]
                }
                self.controller_data = {
                    'controllers': ['APACC', 'MPC', 'DRL', 'PID'],
                    'collision_rates': f['comparison/collision_rates'][:],
                    'latencies': f['comparison/latencies'][:]
                }
                self.latency_data = f['performance/latency_distribution'][:]
                self.explainability_data = {
                    'rule_activations': f['explainability/rule_activations'][:],
                    'confidence_scores': f['explainability/confidence_scores'][:]
                }
        else:
            print("Generating synthetic demonstration data...")
            self._generate_synthetic_data()
            
    def _generate_synthetic_data(self):
        """Generate synthetic data matching paper results"""
        np.random.seed(42)  # For reproducibility
        
        # Safety metrics data
        self.metrics_data = {
            'collision_rates': np.random.beta(1, 1000, 10000) * 0.1,  # ~0.06% average
            'ttc_values': np.random.gamma(4, 2, 10000),  # Time-to-collision
            'lane_deviations': np.random.normal(0, 0.15, 10000)  # Lane center deviation
        }
        
        # Controller comparison data
        self.controller_data = {
            'controllers': ['APACC', 'MPC', 'DRL', 'PID'],
            'collision_rates': [0.06, 0.90, 1.20, 2.80],  # From paper
            'latencies': [3.61, 8.87, 4.60, 0.82],  # From paper
            'explainability': ['Full', 'Partial', 'None', 'Full']
        }
        
        # Latency distribution data (milliseconds)
        self.latency_data = {
            'p50': 3.42,
            'p90': 6.90,
            'p99': 7.24,
            'distribution': np.random.gamma(3.5, 1, 10000)
        }
        
        # Explainability data
        num_rules = 25
        num_scenarios = 1000
        self.explainability_data = {
            'rule_activations': np.random.poisson(8.3, (num_scenarios, num_rules)),
            'confidence_scores': np.random.beta(8, 2, num_scenarios),
            'rule_names': [f'R{i:02d}' for i in range(num_rules)]
        }
        
    def save_figure(self, fig, name: str):
        """Save figure in multiple formats"""
        for fmt in self.formats:
            filepath = self.output_dir / f"{name}.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            
    def figure_1_architecture_stack(self):
        """Generate Figure 5.1: APACC-Sim Architecture Stack"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define layers
        layers = [
            {'name': 'User API & Configuration\nInterface', 'color': COLORS['primary'], 'height': 1},
            {'name': 'Orchestration Engine -\nRay/Dask', 'color': COLORS['tertiary'], 'height': 1.2},
            {'name': 'Simulation Modules', 'color': COLORS['light'], 'height': 1.5},
            {'name': 'Metrics Collection & Analysis\nPipeline', 'color': COLORS['secondary'], 'height': 1}
        ]
        
        # Draw layers
        y_pos = 0
        for i, layer in enumerate(layers):
            rect = FancyBboxPatch((0.5, y_pos), 9, layer['height'],
                                boxstyle="round,pad=0.1",
                                facecolor=layer['color'],
                                edgecolor='black',
                                linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(5, y_pos + layer['height']/2, layer['name'],
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
            y_pos += layer['height'] + 0.3
            
        # Add simulation modules
        modules = ['Monte Carlo', 'CARLA', 'SUMO', 'MATLAB']
        module_width = 2
        module_y = 1.5 + 0.3 + 0.3
        for i, module in enumerate(modules):
            x = 0.8 + i * 2.2
            rect = Rectangle((x, module_y), module_width, 0.9,
                           facecolor=COLORS['quaternary'],
                           edgecolor='black',
                           linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + module_width/2, module_y + 0.45, module,
                   ha='center', va='center', fontsize=10)
            
        # Add arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        
        # User config to API
        ax.annotate('', xy=(5, 4.5), xytext=(5, 6),
                   arrowprops=arrow_props)
        
        # API to Orchestration
        ax.annotate('', xy=(5, 3.2), xytext=(5, 4.3),
                   arrowprops=arrow_props)
        
        # Orchestration to modules
        for i in range(4):
            x = 1.8 + i * 2.2
            ax.annotate('', xy=(x, 2.5), xytext=(5, 3),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
            
        # Modules to metrics
        ax.annotate('', xy=(5, 0.5), xytext=(5, 1.4),
                   arrowprops=arrow_props)
        
        # Add labels
        ax.text(0.2, 6.5, 'User Configuration', fontsize=10, style='italic')
        ax.text(9.5, -0.2, 'Certification Reports', fontsize=10, style='italic')
        
        # Formatting
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 7)
        ax.axis('off')
        ax.set_title('APACC-Sim Architecture Stack', fontsize=16, fontweight='bold', pad=20)
        
        self.save_figure(fig, 'figure_1_architecture_stack')
        plt.close()
        
    def figure_2_scenario_generation(self):
        """Generate Figure 5.2: Scenario Generation and Distribution Flow"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define components
        components = {
            'crypto_seed': {'pos': (1, 8), 'size': (2, 1), 'color': COLORS['tertiary']},
            'param_space': {'pos': (4, 8), 'size': (3, 1.5), 'color': COLORS['primary']},
            'correlation': {'pos': (8, 8), 'size': (3, 1.5), 'color': COLORS['primary']},
            'generator': {'pos': (5, 5.5), 'size': (3, 1), 'color': COLORS['quaternary']},
            'queue': {'pos': (2, 3), 'size': (2, 1.5), 'color': COLORS['secondary']},
            'checkpoint': {'pos': (8, 3.5), 'size': (2.5, 1), 'color': COLORS['tertiary']},
            'scheduler': {'pos': (5, 2), 'size': (3, 1), 'color': COLORS['tertiary']},
            'workers': {'pos': (1, 0), 'size': (10, 1), 'color': COLORS['light']},
            'collector': {'pos': (5, -2), 'size': (3, 1), 'color': COLORS['quaternary']},
            'analysis': {'pos': (5, -4), 'size': (3, 1.5), 'color': COLORS['primary']}
        }
        
        # Draw components
        for name, comp in components.items():
            rect = FancyBboxPatch(comp['pos'], comp['size'][0], comp['size'][1],
                                boxstyle="round,pad=0.05",
                                facecolor=comp['color'],
                                edgecolor='black',
                                linewidth=2)
            ax.add_patch(rect)
            
        # Add labels
        ax.text(2, 8.5, 'Cryptographic Seed', ha='center', va='center', fontweight='bold')
        ax.text(5.5, 8.75, 'Parameter Space\n• Weather: Beta dist\n• Traffic: Poisson dist\n• Failures: Bernoulli',
               ha='center', va='center', fontsize=9)
        ax.text(9.5, 8.75, 'Correlation Matrix\nRain ↔ Visibility\nTraffic ↔ Complexity',
               ha='center', va='center', fontsize=9)
        ax.text(6.5, 6, 'Scenario Generator', ha='center', va='center', fontweight='bold')
        ax.text(3, 3.75, 'Scenario Queue', ha='center', va='center', fontweight='bold')
        ax.text(9.25, 4, 'Checkpoint Manager', ha='center', va='center', fontweight='bold')
        ax.text(6.5, 2.5, 'Ray Scheduler', ha='center', va='center', fontweight='bold')
        ax.text(6.5, -1.5, 'Result Collector', ha='center', va='center', fontweight='bold')
        ax.text(6.5, -3.5, 'Statistical Analysis\n• Importance Sampling\n• Bootstrap CI\n• Sequential Testing',
               ha='center', va='center', fontsize=9)
        
        # Add worker nodes
        for i in range(3):
            x = 1.5 + i * 4
            ax.text(x, 0.5, f'Worker {i+1}\nGPU' if i == 0 else f'Worker {i+1}\nCPU',
                   ha='center', va='center', fontsize=9)
            if i < 2:
                ax.text(x + 2, 0.5, '...', ha='center', va='center')
                
        ax.text(9.5, 0.5, 'Worker N\nSpot Instance', ha='center', va='center', fontsize=9)
        
        # Add arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        
        # Inputs to generator
        ax.annotate('', xy=(5.5, 5.5), xytext=(2, 8),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(6, 5.5), xytext=(5.5, 8),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(6.5, 5.5), xytext=(9.5, 8),
                   arrowprops=arrow_props)
        
        # Generator to distribution
        ax.annotate('', xy=(3, 3.5), xytext=(5.5, 5.5),
                   arrowprops=arrow_props)
        ax.text(4, 4.5, 'Distribution Layer', fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
        
        # Queue to scheduler
        ax.annotate('', xy=(5, 2), xytext=(3, 3),
                   arrowprops=arrow_props)
        
        # Checkpoint recovery
        ax.annotate('', xy=(6.5, 2.5), xytext=(8.5, 3.5),
                   arrowprops=dict(arrowstyle='-->', lw=1.5, color='gray', linestyle='dashed'))
        ax.text(7.5, 3, 'Recovery', fontsize=8, style='italic')
        
        # Scheduler to workers
        ax.annotate('', xy=(3, 1), xytext=(5, 2),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(6.5, 1), xytext=(6.5, 2),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(9, 1), xytext=(7, 2),
                   arrowprops=arrow_props)
        
        # Workers to collector
        ax.annotate('', xy=(6.5, -2), xytext=(6.5, 0),
                   arrowprops=arrow_props)
        ax.text(7, -1, 'Result Aggregation', fontsize=8, style='italic')
        
        # Collector to analysis
        ax.annotate('', xy=(6.5, -4), xytext=(6.5, -2),
                   arrowprops=arrow_props)
        
        # Formatting
        ax.set_xlim(0, 11)
        ax.set_ylim(-5, 10)
        ax.axis('off')
        ax.set_title('Scenario Generation and Distribution Flow', fontsize=16, fontweight='bold')
        
        self.save_figure(fig, 'figure_2_scenario_generation')
        plt.close()
        
    def figure_3_carla_pipeline(self):
        """Generate Figure 5.3: CARLA Integration Pipeline"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Pipeline stages
        stages = [
            {'name': 'Config Parser', 'pos': (1, 2), 'inputs': ['Scenario'], 'outputs': ['CARLA Client']},
            {'name': 'CARLA Client', 'pos': (3, 2), 'inputs': ['Config'], 'outputs': ['Sensor Suite']},
            {'name': 'Sensor Suite', 'pos': (5.5, 2), 'inputs': ['Client'], 'outputs': ['Data Collector']},
            {'name': 'Data Collector', 'pos': (8, 2), 'inputs': ['Raw Data'], 'outputs': ['Processed']},
            {'name': 'Sensor Data Package', 'pos': (10.5, 2), 'inputs': ['Processed'], 'outputs': []}
        ]
        
        # Draw pipeline
        for i, stage in enumerate(stages):
            # Draw box
            rect = FancyBboxPatch((stage['pos'][0]-0.7, stage['pos'][1]-0.5), 1.4, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['primary'] if i % 2 == 0 else COLORS['secondary'],
                                edgecolor='black',
                                linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(stage['pos'][0], stage['pos'][1], stage['name'],
                   ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add arrow to next stage
            if i < len(stages) - 1:
                ax.annotate('', xy=(stages[i+1]['pos'][0]-0.7, stage['pos'][1]),
                           xytext=(stage['pos'][0]+0.7, stage['pos'][1]),
                           arrowprops=dict(arrowstyle='->', lw=2))
                
        # Add data flow labels
        ax.text(0.5, 3, 'Raw Data', fontsize=9, style='italic')
        ax.text(2, 2.8, 'Scenario', fontsize=9)
        ax.text(4.25, 2.8, 'Connection', fontsize=9)
        ax.text(6.75, 2.8, 'RGB, LiDAR, Radar\nSemantic', fontsize=8, ha='center')
        ax.text(9.25, 2.8, 'Synchronized', fontsize=9)
        
        # Add sensor details box
        sensor_rect = FancyBboxPatch((4.5, 0.2), 3, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['light'],
                                   edgecolor='gray',
                                   linewidth=1.5)
        ax.add_patch(sensor_rect)
        ax.text(6, 0.7, 'Sensors:\n• RGB Camera\n• LiDAR\n• Radar\n• Semantic Seg',
               ha='center', va='center', fontsize=8)
        
        # Formatting
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')
        ax.set_title('CARLA Integration Pipeline', fontsize=16, fontweight='bold')
        
        self.save_figure(fig, 'figure_3_carla_pipeline')
        plt.close()
        
    def figure_4_safety_metrics(self):
        """Generate Figure 5.4: Safety Metrics Computation Pipeline"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define data sources
        sources = [
            {'name': 'Vehicle Position', 'pos': (1, 7)},
            {'name': 'Object Detections', 'pos': (1, 5.5)},
            {'name': 'Trajectories', 'pos': (1, 4)},
            {'name': 'Sensor Status', 'pos': (1, 2.5)}
        ]
        
        # Primary metrics
        primary = [
            {'name': 'Collision Detection\n• Geometric\n• Temporal\n• Predictive', 'pos': (4, 6.5)},
            {'name': 'Lane Deviation\n• Lateral Error\n• TLC Prediction\n• Lane Change Safety', 'pos': (4, 4)},
            {'name': 'Time-to-Collision\n• Constant Velocity\n• Constant Accel\n• Probabilistic', 'pos': (4, 1.5)}
        ]
        
        # Analysis layer
        analysis = [
            {'name': 'Composite Score', 'pos': (7, 5.5)},
            {'name': 'Safety Margins', 'pos': (7, 3.5)},
            {'name': 'Probabilistic Assessment', 'pos': (7, 1.5)}
        ]
        
        # Output
        outputs = [
            {'name': 'ISO 26262\nMetrics', 'pos': (10, 5)},
            {'name': 'Evidence\nPackage', 'pos': (10, 3)},
            {'name': 'ISO 21448\nAnalysis', 'pos': (10, 1)}
        ]
        
        # Draw components
        for src in sources:
            rect = Rectangle((src['pos'][0]-0.7, src['pos'][1]-0.3), 1.4, 0.6,
                           facecolor=COLORS['light'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(src['pos'][0], src['pos'][1], src['name'],
                   ha='center', va='center', fontsize=9)
            
        for prim in primary:
            rect = FancyBboxPatch((prim['pos'][0]-1, prim['pos'][1]-0.7), 2, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['primary'],
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(prim['pos'][0], prim['pos'][1], prim['name'],
                   ha='center', va='center', fontsize=8)
            
        for anal in analysis:
            rect = FancyBboxPatch((anal['pos'][0]-0.8, anal['pos'][1]-0.5), 1.6, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['quaternary'],
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(anal['pos'][0], anal['pos'][1], anal['name'],
                   ha='center', va='center', fontsize=9)
            
        for out in outputs:
            rect = FancyBboxPatch((out['pos'][0]-0.7, out['pos'][1]-0.5), 1.4, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['tertiary'],
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(out['pos'][0], out['pos'][1], out['name'],
                   ha='center', va='center', fontsize=9, fontweight='bold')
            
        # Add arrows
        arrow_props = dict(arrowstyle='->', lw=1.5, color='gray')
        
        # Sources to primary metrics
        for src in sources[:2]:
            ax.annotate('', xy=(3, 6.5), xytext=(src['pos'][0]+0.7, src['pos'][1]),
                       arrowprops=arrow_props)
        ax.annotate('', xy=(3, 4), xytext=(1.7, 4),
                   arrowprops=arrow_props)
        ax.annotate('', xy=(3, 1.5), xytext=(1.7, 2.5),
                   arrowprops=arrow_props)
        
        # Primary to analysis
        for i, prim in enumerate(primary):
            for j, anal in enumerate(analysis):
                ax.annotate('', xy=(6.2, anal['pos'][1]), 
                           xytext=(5, prim['pos'][1]),
                           arrowprops=dict(arrowstyle='->', lw=0.8, color='lightgray'))
                
        # Analysis to outputs
        for anal in analysis:
            for out in outputs:
                ax.annotate('', xy=(9.3, out['pos'][1]),
                           xytext=(7.8, anal['pos'][1]),
                           arrowprops=dict(arrowstyle='->', lw=0.8, color='lightgray'))
                
        # Add labels
        ax.text(1, 8, 'Raw Data Sources', fontsize=10, fontweight='bold', style='italic')
        ax.text(4, 8, 'Primary Metrics', fontsize=10, fontweight='bold', style='italic')
        ax.text(7, 7, 'Analysis Layer', fontsize=10, fontweight='bold', style='italic')
        ax.text(10, 6.5, 'Certification Output', fontsize=10, fontweight='bold', style='italic')
        
        # Add safety envelope
        ax.text(7, 0.2, 'Safety Envelope', ha='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['light']))
        
        # Formatting
        ax.set_xlim(0, 11.5)
        ax.set_ylim(-0.5, 8.5)
        ax.axis('off')
        ax.set_title('Safety Metrics Computation Pipeline', fontsize=16, fontweight='bold')
        
        self.save_figure(fig, 'figure_4_safety_metrics')
        plt.close()
        
    def figure_5_explainability(self):
        """Generate Figure 5.5: Explainability Instrumentation Pipeline"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Decision context box
        context_rect = Rectangle((1, 6), 8, 2, facecolor=COLORS['light'],
                               edgecolor='black', linewidth=2)
        ax.add_patch(context_rect)
        ax.text(5, 7.5, 'Decision Context', ha='center', fontweight='bold', fontsize=12)
        
        # Context components
        contexts = [
            {'name': 'Active Rules\nR17: Pedestrian\nR23: Lane Keep', 'pos': (2.5, 6.8)},
            {'name': 'Vehicle State\nv=15m/s, θ=0.2rad', 'pos': (5, 6.8)},
            {'name': 'Sensor Inputs\nt = 0.125s', 'pos': (7.5, 6.8)}
        ]
        
        for ctx in contexts:
            rect = Rectangle((ctx['pos'][0]-1, ctx['pos'][1]-0.5), 2, 1,
                           facecolor=COLORS['tertiary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(ctx['pos'][0], ctx['pos'][1], ctx['name'],
                   ha='center', va='center', fontsize=8)
            
        # Decision process
        process_rect = Rectangle((1, 3.5), 8, 2, facecolor='white',
                               edgecolor='black', linewidth=2)
        ax.add_patch(process_rect)
        ax.text(5, 5, 'Decision Process', ha='center', fontweight='bold', fontsize=12)
        
        # Process components
        processes = [
            {'name': 'Fuzzy Inference\nw₁₇ = 0.8\nw₂₃ = 0.3', 'pos': (3, 4.2)},
            {'name': 'MPC Optimization\nHorizon: 2.0s\nConstraints: Active', 'pos': (7, 4.2)}
        ]
        
        for proc in processes:
            rect = FancyBboxPatch((proc['pos'][0]-1.2, proc['pos'][1]-0.5), 2.4, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['primary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(proc['pos'][0], proc['pos'][1], proc['name'],
                   ha='center', va='center', fontsize=8)
            
        # Control decision
        decision_rect = FancyBboxPatch((3.5, 2.2), 3, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=COLORS['quaternary'], edgecolor='black', linewidth=2)
        ax.add_patch(decision_rect)
        ax.text(5, 2.6, 'Control Decision\nδ = -0.15 rad\na = -2.1 m/s²',
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Logging layer
        log_rect = Rectangle((1, 0.5), 8, 1.2, facecolor='white',
                           edgecolor='black', linewidth=2)
        ax.add_patch(log_rect)
        ax.text(5, 1.3, 'Logging & Analysis', ha='center', fontweight='bold', fontsize=12)
        
        # Log components
        logs = [
            {'name': 'Decision Log\nCryptographic Hash', 'pos': (2.5, 0.8)},
            {'name': 'Uncertainty\nσ = 0.12', 'pos': (5, 0.8)},
            {'name': 'Activation Trace', 'pos': (7.5, 0.8)}
        ]
        
        for log in logs:
            rect = FancyBboxPatch((log['pos'][0]-0.9, log['pos'][1]-0.3), 1.8, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['secondary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(log['pos'][0], log['pos'][1], log['name'],
                   ha='center', va='center', fontsize=8)
            
        # Explanation generation
        exp_rect = Rectangle((1, -1.5), 8, 1.5, facecolor=COLORS['light'],
                           edgecolor='black', linewidth=2)
        ax.add_patch(exp_rect)
        ax.text(5, -0.3, 'Explanation Generation', ha='center', fontweight='bold', fontsize=12)
        
        exp_components = [
            {'name': 'Template Engine', 'pos': (3, -1)},
            {'name': 'Visualization', 'pos': (7, -1)}
        ]
        
        for exp in exp_components:
            rect = Rectangle((exp['pos'][0]-1, exp['pos'][1]-0.3), 2, 0.6,
                           facecolor=COLORS['primary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(exp['pos'][0], exp['pos'][1], exp['name'],
                   ha='center', va='center', fontsize=9)
            
        # Natural language output
        output_rect = FancyBboxPatch((2, -3), 6, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['quaternary'], edgecolor='black', linewidth=2)
        ax.add_patch(output_rect)
        ax.text(5, -2.5, 'Natural Language:\n"Moderate braking due to pedestrian detected 15m ahead.\nConfidence: 87%"',
               ha='center', va='center', fontsize=8, style='italic')
        
        # Add arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        
        # Context to process
        ax.annotate('', xy=(5, 3.5), xytext=(5, 6),
                   arrowprops=arrow_props)
        
        # Process to decision
        ax.annotate('', xy=(5, 2.2), xytext=(5, 3.5),
                   arrowprops=arrow_props)
        
        # Decision to logging
        ax.annotate('', xy=(5, 1.7), xytext=(5, 2.2),
                   arrowprops=arrow_props)
        
        # Logging to explanation
        ax.annotate('', xy=(5, -0.5), xytext=(5, 0.5),
                   arrowprops=arrow_props)
        
        # Explanation to output
        ax.annotate('', xy=(5, -3), xytext=(5, -1.5),
                   arrowprops=arrow_props)
        
        # Formatting
        ax.set_xlim(0, 10)
        ax.set_ylim(-3.5, 8.5)
        ax.axis('off')
        ax.set_title('Explainability Instrumentation Pipeline', fontsize=16, fontweight='bold')
        
        self.save_figure(fig, 'figure_5_explainability')
        plt.close()
        
    def figure_6_cloud_edge_deployment(self):
        """Generate Figure 5.6: Cloud-Edge Deployment Architecture"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Vehicle edge layer
        vehicle_rect = Rectangle((1, 0), 10, 2, facecolor=COLORS['light'],
                               edgecolor='black', linewidth=2)
        ax.add_patch(vehicle_rect)
        ax.text(6, 1.7, 'Vehicle Edge Layer', ha='center', fontweight='bold', fontsize=12)
        
        vehicle_components = [
            {'name': 'Vehicle ECU\nReal-time Control', 'pos': (2.5, 0.8)},
            {'name': 'Edge GPU\nNVIDIA Orin', 'pos': (5.5, 0.8)},
            {'name': 'Local Cache\nScenario Patterns', 'pos': (8.5, 0.8)}
        ]
        
        for comp in vehicle_components:
            rect = Rectangle((comp['pos'][0]-1, comp['pos'][1]-0.4), 2, 0.8,
                           facecolor=COLORS['tertiary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
                   ha='center', va='center', fontsize=8)
            
        # Roadside edge
        roadside_rect = Rectangle((1, 3), 10, 2, facecolor='#E8E8E8',
                                edgecolor='black', linewidth=2)
        ax.add_patch(roadside_rect)
        ax.text(6, 4.7, 'Roadside Edge', ha='center', fontweight='bold', fontsize=12)
        
        roadside_components = [
            {'name': 'Road Side Unit\n5G/C-V2X', 'pos': (3, 3.8)},
            {'name': 'Multi-Access Edge\nComputing', 'pos': (6, 3.8)},
            {'name': 'Local Aggregation\nTraffic Patterns', 'pos': (9, 3.8)}
        ]
        
        for comp in roadside_components:
            rect = Rectangle((comp['pos'][0]-1.2, comp['pos'][1]-0.4), 2.4, 0.8,
                           facecolor=COLORS['secondary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
                   ha='center', va='center', fontsize=8)
            
        # Regional edge
        regional_rect = Rectangle((1, 6), 10, 1.5, facecolor='#D0D0D0',
                                edgecolor='black', linewidth=2)
        ax.add_patch(regional_rect)
        ax.text(6, 7.2, 'Regional Edge', ha='center', fontweight='bold', fontsize=12)
        
        regional_components = [
            {'name': 'Regional DC\nSim Caching', 'pos': (3.5, 6.6)},
            {'name': 'Model Updates\nOTA Deploy', 'pos': (7.5, 6.6)}
        ]
        
        for comp in regional_components:
            rect = Rectangle((comp['pos'][0]-1.3, comp['pos'][1]-0.3), 2.6, 0.6,
                           facecolor=COLORS['primary'], edgecolor='black')
            ax.add_patch(rect)
            ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
                   ha='center', va='center', fontsize=8)
            
        # Cloud core
        cloud_rect = Rectangle((1, 8.5), 10, 1.5, facecolor=COLORS['primary'],
                             edgecolor='black', linewidth=2)
        ax.add_patch(cloud_rect)
        ax.text(6, 9.7, 'Cloud Core', ha='center', fontweight='bold', fontsize=12, color='white')
        
        cloud_components = [
            {'name': 'HPC Clusters\nMATLAB Verify', 'pos': (2.5, 9)},
            {'name': 'GPU Clusters\nCARLA Simulation', 'pos': (5.5, 9)},
            {'name': 'Spot Fleet\nMonte Carlo', 'pos': (8.5, 9)}
        ]
        
        for comp in cloud_components:
            rect = Rectangle((comp['pos'][0]-1, comp['pos'][1]-0.3), 2, 0.6,
                           facecolor='white