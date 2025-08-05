#!/usr/bin/env python3
"""
Validate Controller Script

Main entry point for running validation campaigns
on custom autonomous controllers.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from apacc_sim import SimulationOrchestrator, ScenarioConfig
from apacc_sim.metrics import MetricsCollector


def setup_logging(verbose: bool):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_controller(controller_path: str):
    """
    Load controller from Python file
    
    Args:
        controller_path: Path to controller implementation
        
    Returns:
        Controller object or function
    """
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("controller", controller_path)
    controller_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller_module)
    
    # Look for Controller class or control function
    if hasattr(controller_module, 'Controller'):
        return controller_module.Controller()
    elif hasattr(controller_module, 'control'):
        return controller_module.control
    else:
        raise ValueError(f"No Controller class or control function found in {controller_path}")


def main():
    """Main validation execution"""
    parser = argparse.ArgumentParser(
        description="APACC-Sim Controller Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full validation suite
  %(prog)s --controller my_controller.py --config configs/full_validation.yaml
  
  # Quick test with Monte Carlo only
  %(prog)s --controller my_controller.py --monte-carlo 1000
  
  # CARLA-only validation
  %(prog)s --controller my_controller.py --carla-only --scenarios urban_rain highway_fog
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--controller', '-c',
        required=True,
        help='Path to controller implementation (.py file)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-f',
        help='Path to validation configuration file (YAML/JSON)'
    )
    
    # Simulation options
    parser.add_argument(
        '--monte-carlo', '-m',
        type=int,
        default=0,
        help='Number of Monte Carlo scenarios (0 to disable)'
    )
    
    parser.add_argument(
        '--carla-only',
        action='store_true',
        help='Run only CARLA validation'
    )
    
    parser.add_argument(
        '--sumo-only',
        action='store_true',
        help='Run only SUMO validation'
    )
    
    parser.add_argument(
        '--scenarios', '-s',
        nargs='+',
        default=['urban_day'],
        help='CARLA scenarios to help='CARLA scenarios to run (urban_day, urban_rain, highway_fog)'
    )
    
    parser.add_argument(
        '--skip-matlab',
        action='store_true',
        help='Skip MATLAB formal verification'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Directory for results (default: results)'
    )
    
    parser.add_argument(
        '--format',
        choices=['parquet', 'csv', 'json'],
        default='parquet',
        help='Output format for results'
    )
    
    parser.add_argument(
        '--save-raw',
        action='store_true',
        help='Save raw simulation data (warning: large files)'
    )
    
    # Execution options
    parser.add_argument(
        '--parallel', '-j',
        type=int,
        default=-1,
        help='Number of parallel workers (-1 for auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running validation'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load controller
    logger.info(f"Loading controller from {args.controller}")
    try:
        controller = load_controller(args.controller)
    except Exception as e:
        logger.error(f"Failed to load controller: {e}")
        return 1
    
    # Build configuration
    if args.config:
        config = args.config
    else:
        # Build config from command line arguments
        config = ScenarioConfig(
            name=Path(args.controller).stem,
            monte_carlo_runs=args.monte_carlo,
            carla_scenarios=args.scenarios if not args.sumo_only else [],
            sumo_traffic_density="medium" if not args.carla_only else None,
            matlab_validation=not args.skip_matlab,
            output_format=args.format,
            parallel_workers=args.parallel
        )
        
        # Handle exclusive modes
        if args.carla_only:
            config.monte_carlo_runs = 0
            config.sumo_traffic_density = None
            config.matlab_validation = False
        elif args.sumo_only:
            config.monte_carlo_runs = 0
            config.carla_scenarios = []
            config.matlab_validation = False
    
    # Dry run - just print config
    if args.dry_run:
        logger.info("Dry run - configuration:")
        logger.info(f"  Controller: {args.controller}")
        logger.info(f"  Monte Carlo runs: {config.monte_carlo_runs}")
        logger.info(f"  CARLA scenarios: {config.carla_scenarios}")
        logger.info(f"  SUMO enabled: {config.sumo_traffic_density is not None}")
        logger.info(f"  MATLAB verification: {config.matlab_validation}")
        logger.info(f"  Output format: {config.output_format}")
        logger.info(f"  Parallel workers: {config.parallel_workers}")
        return 0
    
    # Create orchestrator
    logger.info("Initializing simulation orchestrator")
    orchestrator = SimulationOrchestrator(config)
    
    # Run validation
    try:
        logger.info("Starting validation suite")
        results = orchestrator.run_validation_suite(
            controller,
            save_raw_data=args.save_raw
        )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*60)
        
        # Extract key metrics
        total_scenarios = len(results)
        collision_rate = results['collision'].mean() * 100 if 'collision' in results else 0
        avg_latency = results['avg_control_latency'].mean() if 'avg_control_latency' in results else 0
        
        logger.info(f"Total scenarios: {total_scenarios}")
        logger.info(f"Collision rate: {collision_rate:.2f}%")
        logger.info(f"Average latency: {avg_latency:.2f} ms")
        logger.info(f"Results saved to: {orchestrator.results_dir}")
        
        # Return success/failure based on collision rate
        return 0 if collision_rate < 1.0 else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        orchestrator.cleanup()


if __name__ == '__main__':
    sys.exit(main())