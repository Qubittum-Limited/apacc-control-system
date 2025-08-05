#!/usr/bin/env python3
"""
APACC Metadata Generation Script
================================
Creates comprehensive metadata from simulation configuration and runtime environment
for reproducibility and archival purposes.

Usage:
    python generate_metadata.py --config config/simulation.yaml --results results/20250128_143052
    python generate_metadata.py --config config/simulation.yaml --output metadata.json

Author: George Frangou
Institution: Cranfield University
DOI: https://doi.org/10.5281/zenodo.8475
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Try to import optional dependencies
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

try:
    import GPUtil
    import nvidia_ml_py as nvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class MetadataGenerator:
    """Generate comprehensive metadata for APACC validation runs."""
    
    def __init__(self, config_path: Path, results_dir: Optional[Path] = None):
        self.config_path = Path(config_path)
        self.results_dir = results_dir
        self.metadata = {
            'apacc_metadata_version': '1.0',
            'generation_timestamp': datetime.now().isoformat(),
            'doi': 'https://doi.org/10.5281/zenodo.8475',
            'paper_title': 'Quantitative Validation of Artificial Precognition Adaptive Cognised Control',
            'author': 'George Frangou',
            'institution': 'Cranfield University'
        }
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete metadata."""
        self.metadata['configuration'] = self._get_configuration_metadata()
        self.metadata['environment'] = self._get_environment_metadata()
        self.metadata['system'] = self._get_system_metadata()
        self.metadata['dependencies'] = self._get_dependencies_metadata()
        self.metadata['git'] = self._get_git_metadata()
        
        if self.results_dir:
            self.metadata['execution'] = self._get_execution_metadata()
        
        return self.metadata
    
    def _get_configuration_metadata(self) -> Dict[str, Any]:
        """Extract metadata from configuration file."""
        config_meta = {}
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Configuration file hash
        with open(self.config_path, 'rb') as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()
        
        config_meta['file_path'] = str(self.config_path)
        config_meta['file_hash_sha256'] = config_hash
        config_meta['file_size_bytes'] = self.config_path.stat().st_size
        
        # Extract key parameters
        config_meta['total_scenarios'] = {
            'monte_carlo': config['simulation']['monte_carlo']['scenarios'],
            'carla': config['simulation']['carla']['episodes'],
            'sumo': f"{config['simulation']['sumo']['duration']}s simulation",
            'matlab': len(config['simulation']['matlab']['test_cases'])
        }
        
        config_meta['control_frequency_hz'] = config['apacc']['control_frequency']
        config_meta['gpu_acceleration'] = config['apacc']['gpu_acceleration']
        config_meta['parallel_workers'] = config['simulation']['monte_carlo']['parallel_workers']
        
        # Environment flags
        config_meta['enabled_environments'] = {
            env: cfg.get('enabled', True) 
            for env, cfg in config['environments'].items()
        }
        
        return config_meta
    
    def _get_environment_metadata(self) -> Dict[str, Any]:
        """Get runtime environment metadata."""
        env_meta = {}
        
        # Python environment
        env_meta['python'] = {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'executable': sys.executable
        }
        
        # Operating system
        env_meta['os'] = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        # Environment variables
        important_vars = [
            'SUMO_HOME', 'CARLA_ROOT', 'CUDA_HOME', 'CUDA_PATH',
            'LD_LIBRARY_PATH', 'PATH', 'PYTHONPATH'
        ]
        env_meta['environment_variables'] = {
            var: os.environ.get(var, 'NOT SET')
            for var in important_vars
        }
        
        # Check for specific tools
        env_meta['tools'] = {}
        
        # CARLA version
        try:
            import carla
            env_meta['tools']['carla_version'] = getattr(carla, '__version__', 'unknown')
        except ImportError:
            env_meta['tools']['carla_version'] = 'not installed'
        
        # SUMO version
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home:
            sumo_binary = os.path.join(sumo_home, 'bin', 'sumo')
            if platform.system() == 'Windows':
                sumo_binary += '.exe'
            
            try:
                result = subprocess.run([sumo_binary, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    env_meta['tools']['sumo_version'] = result.stdout.split('\n')[0]
                else:
                    env_meta['tools']['sumo_version'] = 'binary found but version check failed'
            except Exception:
                env_meta['tools']['sumo_version'] = 'SUMO_HOME set but binary not accessible'
        else:
            env_meta['tools']['sumo_version'] = 'SUMO_HOME not set'
        
        # MATLAB version
        try:
            import matlab.engine
            eng = matlab.engine.start_matlab()
            matlab_version = eng.version()
            eng.quit()
            env_meta['tools']['matlab_version'] = matlab_version
        except Exception:
            env_meta['tools']['matlab_version'] = 'not accessible'
        
        # CUDA version
        try:
            import torch
            if torch.cuda.is_available():
                env_meta['tools']['cuda_version'] = torch.version.cuda
                env_meta['tools']['cudnn_version'] = torch.backends.cudnn.version()
            else:
                env_meta['tools']['cuda_version'] = 'CUDA runtime found but no GPU'
        except ImportError:
            try:
                import cupy
                env_meta['tools']['cuda_version'] = f"CUDA {cupy.cuda.runtime.runtimeGetVersion()}"
            except ImportError:
                env_meta['tools']['cuda_version'] = 'not detected'
        
        return env_meta
    
    def _get_system_metadata(self) -> Dict[str, Any]:
        """Get system hardware metadata."""
        sys_meta = {}
        
        # Basic system info
        sys_meta['hostname'] = platform.node()
        sys_meta['architecture'] = platform.machine()
        sys_meta['cpu_count'] = os.cpu_count()
        
        # Detailed CPU info if available
        if CPUINFO_AVAILABLE:
            cpu_info = cpuinfo.get_cpu_info()
            sys_meta['cpu'] = {
                'brand': cpu_info.get('brand_raw', 'unknown'),
                'arch': cpu_info.get('arch', 'unknown'),
                'bits': cpu_info.get('bits', 0),
                'count': cpu_info.get('count', os.cpu_count()),
                'frequency_mhz': cpu_info.get('hz_actual_raw', [0])[0] / 1e6
            }
        else:
            sys_meta['cpu'] = {
                'info': 'install py-cpuinfo for detailed CPU information'
            }
        
        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            sys_meta['memory'] = {
                'total_gb': round(mem.total / (1024**3), 2),
                'available_gb': round(mem.available / (1024**3), 2),
                'percent_used': mem.percent
            }
        except ImportError:
            sys_meta['memory'] = {'info': 'install psutil for memory information'}
        
        # GPU info
        if GPU_AVAILABLE:
            try:
                nvml.nvmlInit()
                gpu_count = nvml.nvmlDeviceGetCount()
                sys_meta['gpu'] = {
                    'count': gpu_count,
                    'devices': []
                }
                
                for i in range(gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    sys_meta['gpu']['devices'].append({
                        'index': i,
                        'name': name,
                        'memory_gb': round(mem_info.total / (1024**3), 2),
                        'driver_version': nvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    })
                
                nvml.nvmlShutdown()
            except Exception as e:
                sys_meta['gpu'] = {'error': str(e)}
        else:
            sys_meta['gpu'] = {'info': 'GPU libraries not available'}
        
        return sys_meta
    
    def _get_dependencies_metadata(self) -> Dict[str, Any]:
        """Get Python package dependencies."""
        deps_meta = {}
        
        # Get installed packages
        try:
            import pkg_resources
            installed_packages = {
                pkg.key: pkg.version
                for pkg in pkg_resources.working_set
            }
            
            # Filter to relevant packages
            relevant_packages = [
                'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
                'carla', 'traci', 'sumolib', 'torch', 'tensorflow',
                'h5py', 'zarr', 'yaml', 'pyyaml', 'joblib',
                'psutil', 'tqdm', 'plotly', 'pytest', 'coverage'
            ]
            
            deps_meta['python_packages'] = {
                pkg: installed_packages.get(pkg, 'not installed')
                for pkg in relevant_packages
                if pkg in installed_packages or pkg in ['carla', 'traci', 'sumolib']
            }
            
            deps_meta['total_packages_installed'] = len(installed_packages)
            
        except ImportError:
            # Fallback to pip list
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    packages = json.loads(result.stdout)
                    deps_meta['python_packages'] = {
                        pkg['name']: pkg['version']
                        for pkg in packages
                        if pkg['name'].lower() in relevant_packages
                    }
                else:
                    deps_meta['python_packages'] = {'error': 'pip list failed'}
            except Exception as e:
                deps_meta['python_packages'] = {'error': str(e)}
        
        return deps_meta
    
    def _get_git_metadata(self) -> Dict[str, Any]:
        """Get git repository metadata."""
        git_meta = {}
        
        if not GIT_AVAILABLE:
            git_meta['info'] = 'GitPython not installed'
            return git_meta
        
        try:
            repo = git.Repo(search_parent_directories=True)
            
            git_meta['commit_hash'] = repo.head.commit.hexsha
            git_meta['commit_date'] = repo.head.commit.committed_datetime.isoformat()
            git_meta['branch'] = repo.active_branch.name
            git_meta['remotes'] = [
                {'name': remote.name, 'url': list(remote.urls)[0]}
                for remote in repo.remotes
            ]
            git_meta['is_dirty'] = repo.is_dirty()
            git_meta['untracked_files'] = len(repo.untracked_files)
            
            # Get recent commits
            git_meta['recent_commits'] = [
                {
                    'hash': commit.hexsha[:8],
                    'message': commit.message.strip().split('\n')[0],
                    'author': str(commit.author),
                    'date': commit.committed_datetime.isoformat()
                }
                for commit in list(repo.iter_commits(max_count=5))
            ]
            
        except Exception as e:
            git_meta['error'] = f"Not a git repository or error: {e}"
        
        return git_meta
    
    def _get_execution_metadata(self) -> Dict[str, Any]:
        """Get execution-specific metadata from results directory."""
        exec_meta = {}
        
        if not self.results_dir or not self.results_dir.exists():
            return exec_meta
        
        # Results directory info
        exec_meta['results_directory'] = str(self.results_dir)
        exec_meta['results_directory_size_mb'] = self._get_directory_size(self.results_dir)
        
        # List result files
        result_files = []
        for pattern in ['*.hdf5', '*.csv', '*.json', '*.log']:
            for file in self.results_dir.glob(pattern):
                result_files.append({
                    'name': file.name,
                    'size_mb': round(file.stat().st_size / (1024**2), 2),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                })
        
        exec_meta['result_files'] = result_files
        
        # Parse validation summary if it exists
        summary_file = self.results_dir / 'validation_summary.yaml'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                exec_meta['validation_summary'] = yaml.safe_load(f)
        
        return exec_meta
    
    def _get_directory_size(self, directory: Path) -> float:
        """Calculate total size of directory in MB."""
        total_size = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return round(total_size / (1024**2), 2)
    
    def save(self, output_path: Path):
        """Save metadata to file."""
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        elif output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(self.metadata, f, default_flow_style=False, default=str)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        print(f"Metadata saved to: {output_path}")
        
        # Also create a hash of the metadata file itself
        with open(output_path, 'rb') as f:
            metadata_hash = hashlib.sha256(f.read()).hexdigest()
        
        print(f"Metadata file hash (SHA-256): {metadata_hash}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive metadata for APACC validation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='config/simulation.yaml',
                      help='Path to simulation configuration file')
    parser.add_argument('--results', type=str,
                      help='Path to results directory (optional)')
    parser.add_argument('--output', type=str,
                      help='Output file path (default: results_dir/metadata.json)')
    parser.add_argument('--format', type=str, choices=['json', 'yaml'], default='json',
                      help='Output format')
    parser.add_argument('--print', action='store_true',
                      help='Print metadata to console')
    
    args = parser.parse_args()
    
    # Determine paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    results_dir = Path(args.results) if args.results else None
    
    # Generate metadata
    generator = MetadataGenerator(config_path, results_dir)
    metadata = generator.generate()
    
    # Print if requested
    if args.print:
        print(json.dumps(metadata, indent=2, default=str))
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif results_dir:
        output_path = results_dir / f"metadata.{args.format}"
    else:
        output_path = Path(f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}")
    
    # Save metadata
    generator.save(output_path)


if __name__ == "__main__":
    main()