#!/usr/bin/env python3
"""
APACC Environment Validation Script
===================================
Validates that all required dependencies and environment settings are correctly configured
for running the APACC validation framework.

Usage:
    python validate_environment.py
    python validate_environment.py --verbose
    python validate_environment.py --fix-instructions

Author: George Frangou
Institution: Cranfield University
DOI: https://doi.org/10.5281/zenodo.8475
"""

import argparse
import os
import platform
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
ENDC = '\033[0m'
BOLD = '\033[1m'


class EnvironmentValidator:
    """Validates APACC runtime environment and dependencies."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.validation_results = {}
        self.has_errors = False
        self.has_warnings = False
    
    def print_status(self, component: str, status: str, message: str = ""):
        """Print colored status message."""
        if status == "OK":
            color = GREEN
            symbol = "✓"
        elif status == "WARNING":
            color = YELLOW
            symbol = "⚠"
            self.has_warnings = True
        else:  # ERROR
            color = RED
            symbol = "✗"
            self.has_errors = True
        
        print(f"{color}{symbol} {component:<30} {status:<10}{ENDC} {message}")
    
    def print_header(self, text: str):
        """Print section header."""
        print(f"\n{BOLD}{BLUE}{'='*60}{ENDC}")
        print(f"{BOLD}{BLUE}{text}{ENDC}")
        print(f"{BOLD}{BLUE}{'='*60}{ENDC}\n")
    
    def check_python_version(self) -> Tuple[str, str]:
        """Validate Python version is 3.10."""
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version.major == 3 and python_version.minor == 10:
            return "OK", f"Python {version_str}"
        elif python_version.major == 3 and python_version.minor > 10:
            return "WARNING", f"Python {version_str} (3.10 recommended)"
        else:
            return "ERROR", f"Python {version_str} (3.10 required)"
    
    def check_package(self, package_name: str, import_name: str = None, 
                     version_attr: str = "__version__", min_version: str = None) -> Tuple[str, str]:
        """Check if a Python package is installed and meets version requirements."""
        if import_name is None:
            import_name = package_name
        
        try:
            module = import_module(import_name)
            
            # Get version if available
            version = "unknown"
            if hasattr(module, version_attr):
                version = getattr(module, version_attr)
            elif hasattr(module, "VERSION"):
                version = getattr(module, "VERSION")
            
            # Check minimum version if specified
            if min_version and version != "unknown":
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    return "WARNING", f"Version {version} (>={min_version} recommended)"
            
            return "OK", f"Version {version}"
            
        except ImportError:
            return "ERROR", "Not installed"
    
    def check_carla(self) -> Tuple[str, str]:
        """Validate CARLA installation and version."""
        try:
            import carla
            version = carla.__version__ if hasattr(carla, '__version__') else "unknown"
            
            if version == "0.9.14":
                return "OK", f"Version {version}"
            elif version != "unknown":
                return "WARNING", f"Version {version} (0.9.14 required)"
            else:
                return "WARNING", "Version unknown (0.9.14 required)"
                
        except ImportError:
            return "ERROR", "Not installed (pip install carla==0.9.14)"
    
    def check_sumo(self) -> Tuple[str, str]:
        """Validate SUMO installation and environment."""
        sumo_home = os.environ.get('SUMO_HOME')
        
        if not sumo_home:
            return "ERROR", "SUMO_HOME not set"
        
        if not os.path.exists(sumo_home):
            return "ERROR", f"SUMO_HOME points to non-existent path: {sumo_home}"
        
        # Check SUMO binary
        sumo_binary = os.path.join(sumo_home, 'bin', 'sumo')
        if platform.system() == 'Windows':
            sumo_binary += '.exe'
        
        if not os.path.exists(sumo_binary):
            return "WARNING", f"SUMO binary not found at {sumo_binary}"
        
        # Try to get SUMO version
        try:
            result = subprocess.run([sumo_binary, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return "OK", version_line
            else:
                return "WARNING", "SUMO found but version check failed"
        except Exception:
            return "WARNING", f"SUMO_HOME set to {sumo_home}"
    
    def check_cuda(self) -> Tuple[str, str]:
        """Validate CUDA installation and GPU availability."""
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return "WARNING", "nvidia-smi not found (GPU acceleration disabled)"
        except (subprocess.SubprocessError, FileNotFoundError):
            return "WARNING", "nvidia-smi not found (GPU acceleration disabled)"
        
        # Check CUDA via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                return "OK", f"CUDA {cuda_version}, {gpu_count}x {gpu_name}"
            else:
                return "WARNING", "CUDA available but no GPU detected"
        except ImportError:
            # Try CuPy as alternative
            try:
                import cupy
                cuda_version = cupy.cuda.runtime.runtimeGetVersion()
                major = cuda_version // 1000
                minor = (cuda_version % 1000) // 10
                return "OK", f"CUDA {major}.{minor} via CuPy"
            except ImportError:
                return "WARNING", "PyTorch/CuPy not installed (GPU validation skipped)"
    
    def check_matlab(self) -> Tuple[str, str]:
        """Validate MATLAB Engine for Python."""
        try:
            import matlab.engine
            
            # Try to start MATLAB engine
            if self.verbose:
                print("  Starting MATLAB engine for validation...")
            
            try:
                eng = matlab.engine.start_matlab()
                version = eng.version()
                eng.quit()
                return "OK", f"MATLAB {version}"
            except Exception as e:
                return "WARNING", "matlab.engine installed but MATLAB not accessible"
                
        except ImportError:
            return "ERROR", "Not installed (see MATLAB docs for installation)"
    
    def check_core_packages(self):
        """Check all core Python packages."""
        packages = [
            ("numpy", None, "__version__", "1.24.0"),
            ("scipy", None, "__version__", "1.10.0"),
            ("pandas", None, "__version__", "2.0.0"),
            ("matplotlib", None, "__version__", "3.7.0"),
            ("yaml", "yaml", "__version__", None),
            ("h5py", None, "__version__", "3.9.0"),
            ("tqdm", None, "__version__", "4.66.0"),
        ]
        
        for package_info in packages:
            if len(package_info) == 4:
                package, import_name, version_attr, min_version = package_info
            else:
                package = package_info
                import_name = None
                version_attr = "__version__"
                min_version = None
            
            status, message = self.check_package(package, import_name, version_attr, min_version)
            self.print_status(f"Python: {package}", status, message)
    
    def check_optional_packages(self):
        """Check optional but recommended packages."""
        packages = [
            ("psutil", None, "__version__", None),
            ("nvidia-ml-py", "nvidia_ml_py", None, None),
            ("py3nvml", None, "__version__", None),
            ("joblib", None, "__version__", None),
            ("plotly", None, "__version__", None),
            ("pytest", None, "__version__", None),
            ("memory_profiler", None, "__version__", None),
        ]
        
        for package_info in packages:
            if len(package_info) == 4:
                package, import_name, version_attr, min_version = package_info
            else:
                package = package_info
                import_name = None
                version_attr = "__version__"
                min_version = None
            
            status, message = self.check_package(package, import_name, version_attr, min_version)
            self.print_status(f"Optional: {package}", status, message)
    
    def check_directory_structure(self):
        """Validate expected directory structure."""
        directories = [
            "config",
            "scripts",
            "simulation",
            "results",
            "models",
            "scenarios/carla",
            "matlab/validation",
        ]
        
        for directory in directories:
            path = Path(directory)
            if path.exists():
                self.print_status(f"Directory: {directory}", "OK", "")
            else:
                self.print_status(f"Directory: {directory}", "WARNING", "Not found (will be created)")
    
    def check_config_files(self):
        """Check for required configuration files."""
        config_files = [
            ("config/simulation.yaml", "Main configuration"),
            ("scripts/run_monte_carlo.py", "Monte Carlo runner"),
            ("runner.py", "Main entry point"),
        ]
        
        for filepath, description in config_files:
            path = Path(filepath)
            if path.exists():
                self.print_status(f"Config: {filepath}", "OK", f"{description}")
            else:
                self.print_status(f"Config: {filepath}", "ERROR", f"Missing - {description}")
    
    def run_validation(self):
        """Run complete environment validation."""
        print(f"{BOLD}APACC Environment Validation{ENDC}")
        print(f"DOI: https://doi.org/10.5281/zenodo.8475")
        print(f"Checking system requirements...\n")
        
        # Core Requirements
        self.print_header("Core Requirements")
        status, message = self.check_python_version()
        self.print_status("Python Version", status, message)
        
        status, message = self.check_carla()
        self.print_status("CARLA", status, message)
        
        status, message = self.check_sumo()
        self.print_status("SUMO", status, message)
        
        status, message = self.check_matlab()
        self.print_status("MATLAB Engine", status, message)
        
        status, message = self.check_cuda()
        self.print_status("CUDA/GPU", status, message)
        
        # Python Packages
        self.print_header("Python Packages")
        self.check_core_packages()
        
        # Optional Packages
        self.print_header("Optional Packages")
        self.check_optional_packages()
        
        # Directory Structure
        self.print_header("Directory Structure")
        self.check_directory_structure()
        
        # Configuration Files
        self.print_header("Configuration Files")
        self.check_config_files()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary."""
        self.print_header("Validation Summary")
        
        if not self.has_errors and not self.has_warnings:
            print(f"{GREEN}{BOLD}✓ All checks passed!{ENDC}")
            print("Your environment is ready for APACC validation.")
        elif not self.has_errors:
            print(f"{YELLOW}{BOLD}⚠ Validation completed with warnings{ENDC}")
            print("Core requirements are met, but some optional components are missing.")
            print("You can proceed with validation, but some features may be limited.")
        else:
            print(f"{RED}{BOLD}✗ Validation failed{ENDC}")
            print("Critical requirements are not met. Please fix errors before proceeding.")
        
        print(f"\nFor detailed setup instructions, see the README.md file.")
    
    def print_fix_instructions(self):
        """Print instructions for fixing common issues."""
        self.print_header("Common Fixes")
        
        fixes = {
            "Python 3.10": "Use pyenv or conda to install Python 3.10:\n  conda create -n apacc python=3.10",
            "CARLA": "pip install carla==0.9.14\n  or download from https://github.com/carla-simulator/carla/releases",
            "SUMO": "1. Download SUMO from https://sumo.dlr.de/\n  2. Set SUMO_HOME environment variable",
            "MATLAB Engine": "cd $MATLAB_ROOT/extern/engines/python\n  python setup.py install",
            "CUDA": "Install NVIDIA drivers and CUDA toolkit from https://developer.nvidia.com/cuda-downloads",
        }
        
        for component, fix in fixes.items():
            print(f"{BOLD}{component}:{ENDC}")
            print(f"  {fix}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate APACC environment and dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--fix-instructions', action='store_true',
                      help='Show instructions for fixing common issues')
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator(verbose=args.verbose)
    validator.run_validation()
    
    if args.fix_instructions:
        validator.print_fix_instructions()
    
    # Exit with error code if validation failed
    sys.exit(1 if validator.has_errors else 0)


if __name__ == "__main__":
    main()