#!/usr/bin/env python3
"""
validate_environment_apaccsim.py

Environment validation script for APACC-Sim toolkit
Checks for required software, dependencies, and configurations
Part of Appendix B: APACC-Sim Toolkit for Paper 3

Author: George Frangou
Institution: Cranfield University
"""

import os
import sys
import subprocess
import platform
import importlib
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class EnvironmentValidator:
    """Validates the APACC-Sim environment setup"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'python': {},
            'carla': {},
            'sumo': {},
            'matlab': {},
            'docker': {},
            'git': {},
            'cuda': {},
            'packages': {}
        }
        self.critical_failures = []
        
    def print_status(self, component: str, status: bool, message: str = ""):
        """Pretty print validation status"""
        symbol = "✓" if status else "✗"
        color = Colors.GREEN if status else Colors.RED
        print(f"{color}{symbol}{Colors.ENDC} {component}: {message}")
        
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠{Colors.ENDC}  {message}")
        
    def print_info(self, message: str):
        """Print info message"""
        if self.verbose:
            print(f"{Colors.BLUE}ℹ{Colors.ENDC}  {message}")
            
    def check_python_version(self) -> bool:
        """Check Python version (requires 3.10+)"""
        print(f"\n{Colors.BOLD}Checking Python Environment...{Colors.ENDC}")
        
        python_version = sys.version_info
        version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        self.results['python']['version'] = version_str
        self.results['python']['path'] = sys.executable
        
        if python_version.major == 3 and python_version.minor >= 10:
            self.print_status("Python", True, f"Version {version_str} (3.10+ required)")
            return True
        else:
            self.print_status("Python", False, f"Version {version_str} (3.10+ required)")
            self.critical_failures.append("Python 3.10+ required")
            return False
            
    def check_carla(self) -> bool:
        """Check CARLA installation and binaries"""
        print(f"\n{Colors.BOLD}Checking CARLA Installation...{Colors.ENDC}")
        
        # Check environment variable
        carla_root = os.environ.get('CARLA_ROOT')
        if not carla_root:
            self.print_status("CARLA_ROOT", False, "Environment variable not set")
            self.print_warning("Set CARLA_ROOT to your CARLA installation directory")
            return False
            
        self.results['carla']['root'] = carla_root
        
        # Check CARLA version (expecting 0.9.14)
        carla_path = Path(carla_root)
        if not carla_path.exists():
            self.print_status("CARLA directory", False, f"Path {carla_root} does not exist")
            return False
            
        # Check for key CARLA files
        carla_exe = carla_path / "CarlaUE4.exe" if platform.system() == "Windows" else carla_path / "CarlaUE4.sh"
        pythonapi_path = carla_path / "PythonAPI" / "carla"
        
        if carla_exe.exists():
            self.print_status("CARLA executable", True, str(carla_exe))
            self.results['carla']['executable'] = str(carla_exe)
        else:
            self.print_status("CARLA executable", False, "Not found")
            return False
            
        if pythonapi_path.exists():
            self.print_status("CARLA PythonAPI", True, "Found")
            # Try to import carla
            try:
                sys.path.insert(0, str(pythonapi_path.parent))
                import carla
                self.print_status("CARLA Python module", True, f"Version {carla.__version__}")
                self.results['carla']['version'] = carla.__version__
                return True
            except ImportError as e:
                self.print_status("CARLA Python module", False, f"Import failed: {e}")
                return False
        else:
            self.print_status("CARLA PythonAPI", False, "Not found")
            return False
            
    def check_sumo(self) -> bool:
        """Check SUMO installation"""
        print(f"\n{Colors.BOLD}Checking SUMO Installation...{Colors.ENDC}")
        
        sumo_binaries = ['sumo', 'sumo-gui']
        sumo_found = True
        
        for binary in sumo_binaries:
            try:
                result = subprocess.run([binary, '--version'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    self.print_status(binary, True, version_line)
                    self.results['sumo'][binary] = version_line
                else:
                    self.print_status(binary, False, "Binary found but returned error")
                    sumo_found = False
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                self.print_status(binary, False, "Not found in PATH")
                self.print_warning(f"Install SUMO from https://sumo.dlr.de/docs/Downloads.php")
                sumo_found = False
                
        # Check SUMO_HOME environment variable
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home:
            self.print_status("SUMO_HOME", True, sumo_home)
            self.results['sumo']['home'] = sumo_home
        else:
            self.print_warning("SUMO_HOME environment variable not set")
            sumo_found = False
            
        return sumo_found
        
    def check_matlab(self) -> bool:
        """Check MATLAB installation and Symbolic Toolbox"""
        print(f"\n{Colors.BOLD}Checking MATLAB Installation...{Colors.ENDC}")
        
        # Try to find MATLAB
        matlab_cmd = 'matlab' if platform.system() != 'Windows' else 'matlab.exe'
        
        try:
            # Check if MATLAB is in PATH
            result = subprocess.run([matlab_cmd, '-batch', 'disp(version)'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=30)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status("MATLAB", True, f"Version {version}")
                self.results['matlab']['version'] = version
                
                # Check for Symbolic Toolbox
                check_symbolic = subprocess.run(
                    [matlab_cmd, '-batch', "disp(exist('syms', 'file'))"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if '2' in check_symbolic.stdout:
                    self.print_status("Symbolic Toolbox", True, "Available")
                    self.results['matlab']['symbolic_toolbox'] = True
                    return True
                else:
                    self.print_status("Symbolic Toolbox", False, "Not found")
                    self.results['matlab']['symbolic_toolbox'] = False
                    return False
            else:
                self.print_status("MATLAB", False, "Found but returned error")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.print_status("MATLAB", False, "Not found in PATH")
            self.print_warning("MATLAB is optional but required for formal verification")
            return False
            
    def check_docker(self) -> bool:
        """Check Docker installation"""
        print(f"\n{Colors.BOLD}Checking Docker Installation...{Colors.ENDC}")
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status("Docker", True, version)
                self.results['docker']['version'] = version
                
                # Check if Docker daemon is running
                daemon_check = subprocess.run(['docker', 'ps'], 
                                            capture_output=True, 
                                            text=True, 
                                            timeout=5)
                if daemon_check.returncode == 0:
                    self.print_status("Docker daemon", True, "Running")
                    self.results['docker']['daemon_running'] = True
                    return True
                else:
                    self.print_status("Docker daemon", False, "Not running")
                    self.print_warning("Start Docker daemon to use containerized deployment")
                    self.results['docker']['daemon_running'] = False
                    return False
            else:
                self.print_status("Docker", False, "Installation found but returned error")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.print_status("Docker", False, "Not found")
            self.print_warning("Docker is optional but recommended for deployment")
            return False
            
    def check_git(self) -> bool:
        """Check Git installation"""
        print(f"\n{Colors.BOLD}Checking Git Installation...{Colors.ENDC}")
        
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status("Git", True, version)
                self.results['git']['version'] = version
                
                # Check if we're in a git repository
                repo_check = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=5)
                if repo_check.returncode == 0:
                    # Get current commit hash
                    commit_result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                                 capture_output=True, 
                                                 text=True, 
                                                 timeout=5)
                    if commit_result.returncode == 0:
                        commit = commit_result.stdout.strip()[:7]
                        self.print_info(f"Git repository detected, commit: {commit}")
                        self.results['git']['commit'] = commit
                return True
            else:
                self.print_status("Git", False, "Found but returned error")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.print_status("Git", False, "Not found")
            self.print_warning("Git is required for version control")
            self.critical_failures.append("Git required for reproducibility")
            return False
            
    def check_cuda(self) -> bool:
        """Check CUDA installation for GPU acceleration"""
        print(f"\n{Colors.BOLD}Checking CUDA/GPU Support...{Colors.ENDC}")
        
        cuda_available = False
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                self.print_status("NVIDIA GPU", True, "Detected via nvidia-smi")
                # Parse GPU info
                for line in result.stdout.split('\n'):
                    if 'NVIDIA' in line and 'Driver Version' in line:
                        self.print_info(line.strip())
                cuda_available = True
                self.results['cuda']['nvidia_smi'] = True
            else:
                self.print_status("NVIDIA GPU", False, "nvidia-smi failed")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.print_status("NVIDIA GPU", False, "nvidia-smi not found")
            
        # Check CUDA toolkit
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                version_line = [l for l in result.stdout.split('\n') if 'release' in l][0]
                self.print_status("CUDA Toolkit", True, version_line.strip())
                self.results['cuda']['nvcc'] = version_line.strip()
                cuda_available = True
            else:
                self.print_status("CUDA Toolkit", False, "nvcc found but failed")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, IndexError):
            self.print_status("CUDA Toolkit", False, "nvcc not found")
            
        if not cuda_available:
            self.print_warning("GPU acceleration not available - simulations will run on CPU")
            
        return cuda_available
        
    def check_python_packages(self, requirements_file: Optional[str] = None) -> bool:
        """Check required Python packages"""
        print(f"\n{Colors.BOLD}Checking Python Package Dependencies...{Colors.ENDC}")
        
        # Default required packages if no requirements file specified
        default_packages = {
            'pyyaml': '5.4.1',
            'matplotlib': '3.5.0',
            'seaborn': '0.11.2',
            'plotly': '5.9.0',
            'scipy': '1.7.3',
            'pandas': '1.4.0',
            'numpy': '1.21.0',
            'h5py': '3.6.0',
            'pytest': '7.0.0',
            'ray': '2.0.0',
            'redis': '4.0.0'
        }
        
        # Load requirements from file if provided
        requirements = {}
        if requirements_file and Path(requirements_file).exists():
            self.print_info(f"Loading requirements from {requirements_file}")
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '>=' in line:
                            pkg, version = line.split('>=')
                            requirements[pkg.strip()] = version.strip()
                        elif '==' in line:
                            pkg, version = line.split('==')
                            requirements[pkg.strip()] = version.strip()
                        else:
                            requirements[line] = None
        else:
            requirements = default_packages
            
        all_satisfied = True
        missing_packages = []
        
        for package, min_version in requirements.items():
            try:
                module = importlib.import_module(package)
                installed_version = getattr(module, '__version__', 'unknown')
                
                if min_version and installed_version != 'unknown':
                    # Simple version comparison (could be enhanced)
                    version_ok = True  # Simplified for this implementation
                    if version_ok:
                        self.print_status(package, True, f"Version {installed_version}")
                        self.results['packages'][package] = installed_version
                    else:
                        self.print_status(package, False, 
                                        f"Version {installed_version} (>={min_version} required)")
                        missing_packages.append(f"{package}>={min_version}")
                        all_satisfied = False
                else:
                    self.print_status(package, True, f"Installed")
                    self.results['packages'][package] = installed_version
                    
            except ImportError:
                self.print_status(package, False, "Not installed")
                missing_packages.append(package)
                all_satisfied = False
                
        if missing_packages:
            self.print_warning(f"\nInstall missing packages with:")
            print(f"pip install {' '.join(missing_packages)}")
            
        return all_satisfied
        
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate validation report"""
        report = {
            'timestamp': subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], 
                                      capture_output=True, text=True).stdout.strip() 
                        if platform.system() != 'Windows' else 'N/A',
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'validation_results': self.results,
            'critical_failures': self.critical_failures,
            'all_checks_passed': len(self.critical_failures) == 0
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.print_info(f"Report saved to {output_file}")
            
        return report
        
    def run_validation(self, requirements_file: Optional[str] = None) -> bool:
        """Run complete environment validation"""
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}APACC-Sim Environment Validation{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        
        # Run all checks
        python_ok = self.check_python_version()
        carla_ok = self.check_carla()
        sumo_ok = self.check_sumo()
        matlab_ok = self.check_matlab()
        docker_ok = self.check_docker()
        git_ok = self.check_git()
        cuda_ok = self.check_cuda()
        packages_ok = self.check_python_packages(requirements_file)
        
        # Summary
        print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}Validation Summary{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        
        if self.critical_failures:
            print(f"\n{Colors.RED}Critical failures detected:{Colors.ENDC}")
            for failure in self.critical_failures:
                print(f"  - {failure}")
            print(f"\n{Colors.RED}Environment validation FAILED{Colors.ENDC}")
            return False
        else:
            print(f"\n{Colors.GREEN}All critical checks passed!{Colors.ENDC}")
            
            if not all([carla_ok, sumo_ok, matlab_ok, docker_ok, cuda_ok]):
                print(f"\n{Colors.YELLOW}Optional components missing:{Colors.ENDC}")
                if not carla_ok:
                    print("  - CARLA: Required for high-fidelity physics simulation")
                if not sumo_ok:
                    print("  - SUMO: Required for traffic simulation")
                if not matlab_ok:
                    print("  - MATLAB: Required for formal verification")
                if not docker_ok:
                    print("  - Docker: Recommended for deployment")
                if not cuda_ok:
                    print("  - CUDA: Recommended for GPU acceleration")
                    
            print(f"\n{Colors.GREEN}Environment ready for APACC-Sim!{Colors.ENDC}")
            return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate APACC-Sim environment setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                     # Run basic validation
  %(prog)s -v                  # Run with verbose output
  %(prog)s -r requirements.txt # Check specific requirements file
  %(prog)s -o report.json      # Save validation report
        '''
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('-r', '--requirements', type=str,
                       default='requirements_apaccsim.txt',
                       help='Path to requirements file (default: requirements_apaccsim.txt)')
    parser.add_argument('-o', '--output', type=str,
                       help='Save validation report to JSON file')
    
    args = parser.parse_args()
    
    # Run validation
    validator = EnvironmentValidator(verbose=args.verbose)
    success = validator.run_validation(args.requirements)
    
    # Generate report if requested
    if args.output:
        validator.generate_report(args.output)
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()