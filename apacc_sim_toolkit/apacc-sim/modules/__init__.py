"""
APACC-Sim Validation Toolkit - Core Simulation Modules

This package contains the four primary simulation paradigms:
- Monte Carlo statistical validation
- CARLA high-fidelity physics
- SUMO large-scale traffic
- MATLAB symbolic verification
"""

from .monte_carlo import MonteCarloSimulator
from .carla_integration import CarlaSimulator
from .sumo_wrapper import SumoSimulator
from .matlab_bridge import MatlabVerifier

__all__ = [
    'MonteCarloSimulator',
    'CarlaSimulator', 
    'SumoSimulator',
    'MatlabVerifier'
]

__version__ = '1.0.0'