"""
APACC-Sim: Open-Source Autonomous Control Validation Toolkit

A comprehensive simulation framework for validating neuro-symbolic
control architectures across multiple simulation paradigms.
"""

from .orchestrator import SimulationOrchestrator, ScenarioConfig
from .metrics import MetricsCollector, SafetyMetrics
from .explainability import ExplainabilityTracker, DecisionTrace

__version__ = '1.0.0'
__author__ = 'APACC-Sim Contributors'
__license__ = 'MIT'

__all__ = [
    'SimulationOrchestrator',
    'ScenarioConfig',
    'MetricsCollector',
    'SafetyMetrics',
    'ExplainabilityTracker',
    'DecisionTrace'
]

# Package-level configuration
DEFAULT_CONFIG = {
    'logging_level': 'INFO',
    'results_directory': 'results',
    'checkpoint_enabled': True,
    'parallel_execution': True
}


def setup_logging(level='INFO'):
    """Configure package-wide logging"""
    import logging
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger('apacc_sim')
    logger.setLevel(getattr(logging, level))
    logger.addHandler(handler)
    
    return logger


# Initialize package logger
logger = setup_logging(DEFAULT_CONFIG['logging_level'])