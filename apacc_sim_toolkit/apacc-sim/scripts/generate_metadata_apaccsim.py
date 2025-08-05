#!/usr/bin/env python3
"""
generate_metadata_apaccsim.py

Collects and generates comprehensive metadata for APACC-Sim validation runs
Ensures reproducibility and traceability for certification

Author: George Frangou
Institution: Cranfield University
"""

import os
import sys
import json
import platform
import subprocess
import datetime
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import socket
import psutil
import importlib
import warnings

# Optional imports for enhanced metadata
try:
    import cpuinfo
    CPU_INFO_AVAILABLE = True
except ImportError:
    CPU_INFO_AVAILABLE = False
    warnings.warn("py-cpuinfo not available. CPU details will be limited.")

try:
    import GPUtil
    GPU_INFO_AVAILABLE = True
except ImportError:
    GPU_INFO_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU details will be limited.")


class MetadataCollector:
    """Collects comprehensive system and experiment metadata"""
    
    def __init__(self, config_file: Optional[str] = None, 
                 experiment_name: Optional[str] = None):
        self.config_file = config_file
        self.experiment_name = experiment_name or "apacc_sim_validation"
        self.metadata = {
            'collection_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'experiment_name': self.experiment_name,
            'metadata_version': '1.0'
        }
        
    def collect_all(self) -> Dict[str, Any]:
        """Collect all metadata categories"""
        print("Collecting APACC-Sim metadata...")
        
        self.metadata['host'] = self._collect_host_info()
        self.metadata['system'] = self._collect_system_info()
        self.metadata['hardware'] = self._collect_hardware_info()
        self.metadata['software'] = self._collect_software_info()
        self.metadata['python'] = self._collect_python_info()
        self.metadata['git'] = self._collect_git_info()
        self.metadata['configuration'] = self._collect_config_info()
        self.metadata['environment'] = self._collect_environment_info()
        self.metadata['checksums'] = self._collect_checksums()
        
        return self.metadata
        
    def _collect_host_info(self) -> Dict[str, Any]:
        """Collect host machine information"""
        print("  Collecting host information...")
        
        host_info = {
            'hostname': socket.gethostname(),
            'fqdn': socket.getfqdn(),
            'user': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
            'working_directory': os.getcwd(),
            'home_directory': str(Path.home())
        }
        
        # Try to get IP addresses
        try:
            host_info['ip_addresses'] = []
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        host_info['ip_addresses'].append({
                            'interface': interface,
                            'address': addr.address
                        })
        except Exception as e:
            host_info['ip_addresses'] = f"Error: {e}"
            
        return host_info
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect operating system information"""
        print("  Collecting system information...")
        
        system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0]
        }
        
        # Linux-specific information
        if platform.system() == 'Linux':
            try:
                with open('/etc/os-release', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('PRETTY_NAME='):
                            system_info['distribution'] = line.split('=')[1].strip().strip('"')
                            break
            except:
                system_info['distribution'] = 'Unknown'
                
            # Kernel information
            try:
                result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
                if result.returncode == 0:
                    system_info['kernel'] = result.stdout.strip()
            except:
                pass
                
        return system_info
        
    def _collect_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware specifications"""
        print("  Collecting hardware information...")
        
        hardware_info = {}
        
        # CPU Information
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else None
        }
        
        # Enhanced CPU info if available
        if CPU_INFO_AVAILABLE:
            try:
                detailed_cpu = cpuinfo.get_cpu_info()
                cpu_info.update({
                    'brand': detailed_cpu.get('brand_raw', 'Unknown'),
                    'vendor': detailed_cpu.get('vendor_id_raw', 'Unknown'),
                    'family': detailed_cpu.get('family', 'Unknown'),
                    'model': detailed_cpu.get('model', 'Unknown'),
                    'flags': detailed_cpu.get('flags', [])[:10]  # First 10 flags only
                })
            except:
                pass
                
        hardware_info['cpu'] = cpu_info
        
        # Memory Information
        memory = psutil.virtual_memory()
        hardware_info['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent
        }
        
        # Swap Information
        swap = psutil.swap_memory()
        hardware_info['swap'] = {
            'total_gb': round(swap.total / (1024**3), 2),
            'used_gb': round(swap.used / (1024**3), 2),
            'percent_used': swap.percent
        }
        
        # Disk Information
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'free_gb': round(usage.free / (1024**3), 2),
                    'percent_used': usage.percent
                })
            except PermissionError:
                continue
        hardware_info['disks'] = disk_info
        
        # GPU Information
        gpu_info = []
        
        # Try nvidia-smi first
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_info.append({
                            'name': parts[0],
                            'memory_mb': parts[1],
                            'driver': parts[2]
                        })
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Use GPUtil if available
        if GPU_INFO_AVAILABLE and not gpu_info:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'temperature': gpu.temperature,
                        'load': gpu.load
                    })
            except:
                pass
                
        hardware_info['gpus'] = gpu_info if gpu_info else 'No GPUs detected'
        
        # Network interfaces
        network_info = []
        for interface, addrs in psutil.net_if_addrs().items():
            interface_info = {'name': interface, 'addresses': []}
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    interface_info['addresses'].append({
                        'type': 'IPv4',
                        'address': addr.address,
                        'netmask': addr.netmask
                    })
                elif addr.family == socket.AF_INET6:
                    interface_info['addresses'].append({
                        'type': 'IPv6',
                        'address': addr.address
                    })
            if interface_info['addresses']:
                network_info.append(interface_info)
        hardware_info['network_interfaces'] = network_info
        
        return hardware_info
        
    def _collect_software_info(self) -> Dict[str, Any]:
        """Collect software environment information"""
        print("  Collecting software information...")
        
        software_info = {}
        
        # Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                software_info['docker'] = result.stdout.strip()
                
                # Check if running in container
                if Path('/.dockerenv').exists():
                    software_info['in_docker_container'] = True
                    
                    # Get container info if possible
                    try:
                        hostname = socket.gethostname()
                        result = subprocess.run(['docker', 'inspect', hostname],
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            container_info = json.loads(result.stdout)[0]
                            software_info['container_id'] = container_info['Id'][:12]
                            software_info['container_image'] = container_info['Config']['Image']
                    except:
                        pass
                else:
                    software_info['in_docker_container'] = False
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            software_info['docker'] = 'Not installed'
            
        # Git
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                software_info['git'] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            software_info['git'] = 'Not installed'
            
        # CUDA
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        software_info['cuda'] = line.strip()
                        break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            software_info['cuda'] = 'Not installed'
            
        # CARLA
        carla_root = os.environ.get('CARLA_ROOT')
        if carla_root and Path(carla_root).exists():
            software_info['carla_root'] = carla_root
            version_file = Path(carla_root) / 'VERSION'
            if version_file.exists():
                software_info['carla_version'] = version_file.read_text().strip()
        else:
            software_info['carla'] = 'Not configured'
            
        # SUMO
        try:
            result = subprocess.run(['sumo', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                software_info['sumo'] = result.stdout.split('\n')[0]
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            software_info['sumo'] = 'Not installed'
            
        # MATLAB
        try:
            result = subprocess.run(['matlab', '-batch', 'disp(version)'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                software_info['matlab'] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            software_info['matlab'] = 'Not installed'
            
        return software_info
        
    def _collect_python_info(self) -> Dict[str, Any]:
        """Collect Python environment information"""
        print("  Collecting Python information...")
        
        python_info = {
            'version': sys.version,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro,
                'releaselevel': sys.version_info.releaselevel
            },
            'executable': sys.executable,
            'prefix': sys.prefix,
            'path': sys.path[:5]  # First 5 paths only
        }
        
        # Virtual environment detection
        python_info['in_virtualenv'] = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if python_info['in_virtualenv']:
            python_info['virtualenv_path'] = sys.prefix
            
        # Conda environment detection
        if 'CONDA_DEFAULT_ENV' in os.environ:
            python_info['conda_env'] = os.environ['CONDA_DEFAULT_ENV']
            python_info['conda_prefix'] = os.environ.get('CONDA_PREFIX', 'Unknown')
            
        # Installed packages
        packages = {}
        important_packages = [
            'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
            'pyyaml', 'h5py', 'ray', 'redis', 'pytest', 'carla', 'sumolib',
            'torch', 'tensorflow', 'scikit-learn'
        ]
        
        for pkg in important_packages:
            try:
                module = importlib.import_module(pkg)
                version = getattr(module, '__version__', 'Unknown')
                packages[pkg] = version
            except ImportError:
                packages[pkg] = 'Not installed'
                
        python_info['key_packages'] = packages
        
        # Get full package list if pip is available
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                all_packages = json.loads(result.stdout)
                python_info['total_packages'] = len(all_packages)
                python_info['all_packages'] = {pkg['name']: pkg['version'] 
                                             for pkg in all_packages[:50]}  # First 50 only
        except:
            python_info['total_packages'] = 'Unknown'
            
        return python_info
        
    def _collect_git_info(self) -> Dict[str, Any]:
        """Collect Git repository information"""
        print("  Collecting Git information...")
        
        git_info = {}
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                git_info['in_repository'] = False
                return git_info
                
            git_info['in_repository'] = True
            
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
                git_info['commit_hash_short'] = result.stdout.strip()[:7]
                
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
                
            # Get remote URL
            result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
                
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['has_uncommitted_changes'] = len(result.stdout.strip()) > 0
                git_info['uncommitted_files'] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                
            # Get last commit info
            result = subprocess.run(['git', 'log', '-1', '--format=%H|%an|%ae|%ai|%s'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split('|')
                if len(parts) >= 5:
                    git_info['last_commit'] = {
                        'hash': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    }
                    
            # Get tags at current commit
            result = subprocess.run(['git', 'tag', '--points-at', 'HEAD'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                git_info['tags'] = result.stdout.strip().split('\n')
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            git_info['error'] = 'Failed to collect git information'
            
        return git_info
        
    def _collect_config_info(self) -> Dict[str, Any]:
        """Collect configuration file information"""
        print("  Collecting configuration information...")
        
        config_info = {}
        
        if self.config_file and Path(self.config_file).exists():
            config_path = Path(self.config_file)
            config_info['path'] = str(config_path.absolute())
            config_info['size_bytes'] = config_path.stat().st_size
            config_info['modified'] = datetime.datetime.fromtimestamp(
                config_path.stat().st_mtime
            ).isoformat()
            
            # Calculate checksum
            with open(config_path, 'rb') as f:
                config_info['sha256'] = hashlib.sha256(f.read()).hexdigest()
                
            # Try to load and include key parameters
            try:
                if config_path.suffix in ['.yaml', '.yml']:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        
                    # Extract key configuration parameters
                    config_info['parameters'] = {
                        'simulation_count': config_data.get('simulation', {}).get('num_scenarios', 'Unknown'),
                        'controllers': config_data.get('controllers', {}).keys() if 'controllers' in config_data else [],
                        'environments': config_data.get('environments', []),
                        'random_seed': config_data.get('random_seed', 'Not set')
                    }
                elif config_path.suffix == '.json':
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    config_info['parameters'] = {
                        'simulation_count': config_data.get('num_scenarios', 'Unknown'),
                        'controllers': list(config_data.get('controllers', {}).keys())
                    }
            except Exception as e:
                config_info['load_error'] = str(e)
        else:
            config_info['status'] = 'No configuration file specified'
            
        return config_info
        
    def _collect_environment_info(self) -> Dict[str, str]:
        """Collect environment variables"""
        print("  Collecting environment variables...")
        
        # Important environment variables to capture
        important_vars = [
            'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
            'CARLA_ROOT', 'SUMO_HOME', 'ROS_DISTRO', 'ROS_VERSION',
            'DISPLAY', 'USER', 'HOME', 'SHELL', 'TERM',
            'APACCSIM_CONFIG', 'APACCSIM_DATA', 'APACCSIM_RESULTS'
        ]
        
        env_info = {}
        for var in important_vars:
            value = os.environ.get(var)
            if value:
                # Truncate very long values like PATH
                if var in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH'] and len(value) > 500:
                    env_info[var] = value[:500] + '...(truncated)'
                else:
                    env_info[var] = value
                    
        # Count total environment variables
        env_info['_total_env_vars'] = len(os.environ)
        
        return env_info
        
    def _collect_checksums(self) -> Dict[str, str]:
        """Calculate checksums of important files"""
        print("  Calculating checksums...")
        
        checksums = {}
        
        # Files to checksum
        important_files = [
            'runner_apaccsim.py',
            'requirements_apaccsim.txt',
            'simulation_apaccsim.yaml',
            self.config_file
        ]
        
        for file_path in important_files:
            if file_path and Path(file_path).exists():
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        checksums[file_path] = {
                            'sha256': hashlib.sha256(content).hexdigest(),
                            'md5': hashlib.md5(content).hexdigest(),
                            'size_bytes': len(content)
                        }
                except Exception as e:
                    checksums[file_path] = f'Error: {e}'
                    
        return checksums
        
    def save_metadata(self, output_file: str, format: str = 'json'):
        """Save metadata to file"""
        output_path = Path(output_file)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            print(f"\nMetadata saved to {output_path} (JSON)")
            
        elif format == 'markdown':
            self._save_markdown(output_path)
            print(f"\nMetadata saved to {output_path} (Markdown)")
            
        elif format == 'both':
            # Save JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            print(f"\nMetadata saved to {json_path} (JSON)")
            
            # Save Markdown
            md_path = output_path.with_suffix('.md')
            self._save_markdown(md_path)
            print(f"Metadata saved to {md_path} (Markdown)")
            
    def _save_markdown(self, output_path: Path):
        """Save metadata as formatted Markdown"""
        lines = []
        lines.append("# APACC-Sim Validation Metadata\n")
        lines.append(f"**Generated:** {self.metadata['collection_timestamp']}  ")
        lines.append(f"**Experiment:** {self.metadata['experiment_name']}  ")
        lines.append(f"**Metadata Version:** {self.metadata['metadata_version']}\n")
        
        # Host Information
        lines.append("## Host Information\n")
        host = self.metadata.get('host', {})
        lines.append(f"- **Hostname:** {host.get('hostname', 'Unknown')}  ")
        lines.append(f"- **User:** {host.get('user', 'Unknown')}  ")
        lines.append(f"- **Working Directory:** `{host.get('working_directory', 'Unknown')}`  \n")
        
        # System Information
        lines.append("## System Information\n")
        system = self.metadata.get('system', {})
        lines.append(f"- **Platform:** {system.get('platform', 'Unknown')}  ")
        lines.append(f"- **Architecture:** {system.get('architecture', 'Unknown')}  ")
        if 'distribution' in system:
            lines.append(f"- **Distribution:** {system['distribution']}  ")
        lines.append("")
        
        # Hardware Information
        lines.append("## Hardware Specifications\n")
        hardware = self.metadata.get('hardware', {})
        
        # CPU
        cpu = hardware.get('cpu', {})
        lines.append("### CPU\n")
        lines.append(f"- **Model:** {cpu.get('brand', 'Unknown')}  ")
        lines.append(f"- **Cores:** {cpu.get('physical_cores', 'Unknown')} physical, "
                    f"{cpu.get('logical_cores', 'Unknown')} logical  ")
        lines.append(f"- **Frequency:** {cpu.get('frequency_current', 'Unknown')} MHz  \n")
        
        # Memory
        memory = hardware.get('memory', {})
        lines.append("### Memory\n")
        lines.append(f"- **Total:** {memory.get('total_gb', 'Unknown')} GB  ")
        lines.append(f"- **Available:** {memory.get('available_gb', 'Unknown')} GB  ")
        lines.append(f"- **Used:** {memory.get('percent_used', 'Unknown')}%  \n")
        
        # GPU
        gpus = hardware.get('gpus', [])
        if isinstance(gpus, list) and gpus:
            lines.append("### GPU\n")
            for i, gpu in enumerate(gpus):
                lines.append(f"**GPU {i}:**  ")
                lines.append(f"- Name: {gpu.get('name', 'Unknown')}  ")
                if 'memory_mb' in gpu:
                    lines.append(f"- Memory: {gpu['memory_mb']}  ")
                if 'driver' in gpu:
                    lines.append(f"- Driver: {gpu['driver']}  ")
                lines.append("")
                
        # Software versions
        lines.append("## Software Environment\n")
        software = self.metadata.get('software', {})
        for key, value in software.items():
            if value and value != 'Not installed':
                lines.append(f"- **{key.title()}:** {value}  ")
        lines.append("")
        
        # Python environment
        lines.append("## Python Environment\n")
        python = self.metadata.get('python', {})
        lines.append(f"- **Version:** {python.get('version', 'Unknown').split()[0]}  ")
        lines.append(f"- **Executable:** `{python.get('executable', 'Unknown')}`  ")
        if python.get('in_virtualenv'):
            lines.append(f"- **Virtual Environment:** `{python.get('virtualenv_path', 'Yes')}`  ")
        if 'conda_env' in python:
            lines.append(f"- **Conda Environment:** {python['conda_env']}  ")
        lines.append("")
        
        # Key packages
        lines.append("### Key Python Packages\n")
        packages = python.get('key_packages', {})
        lines.append("| Package | Version |")
        lines.append("|---------|---------|")
        for pkg, version in packages.items():
            if version != 'Not installed':
                lines.append(f"| {pkg} | {version} |")
        lines.append("")
        
        # Git information
        git = self.metadata.get('git', {})
        if git.get('in_repository'):
            lines.append("## Git Repository\n")
            lines.append(f"- **Branch:** {git.get('branch', 'Unknown')}  ")
            lines.append(f"- **Commit:** `{git.get('commit_hash_short', 'Unknown')}`  ")
            lines.append(f"- **Uncommitted Changes:** {'Yes' if git.get('has_uncommitted_changes') else 'No'}  ")
            if 'remote_url' in git:
                lines.append(f"- **Remote:** {git['remote_url']}  ")
            lines.append("")
            
        # Configuration
        config = self.metadata.get('configuration', {})
        if 'path' in config:
            lines.append("## Configuration\n")
            lines.append(f"- **File:** `{config['path']}`  ")
            lines.append(f"- **SHA256:** `{config.get('sha256', 'Unknown')}`  ")
            if 'parameters' in config:
                params = config['parameters']
                lines.append(f"- **Scenarios:** {params.get('simulation_count', 'Unknown')}  ")
                if 'controllers' in params:
                    lines.append(f"- **Controllers:** {', '.join(params['controllers'])}  ")
            lines.append("")
            
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
            
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on metadata"""
        # Create a unique ID from timestamp and config hash
        timestamp = self.metadata['collection_timestamp']
        
        # Include configuration hash if available
        config_hash = 'no_config'
        if 'configuration' in self.metadata and 'sha256' in self.metadata['configuration']:
            config_hash = self.metadata['configuration']['sha256'][:8]
            
        # Include git commit if available
        git_hash = 'no_git'
        if 'git' in self.metadata and 'commit_hash_short' in self.metadata['git']:
            git_hash = self.metadata['git']['commit_hash_short']
            
        # Generate experiment ID
        exp_id = f"apacc_sim_{timestamp.split('T')[0]}_{git_hash}_{config_hash}"
        exp_id = exp_id.replace(':', '').replace('-', '')
        
        return exp_id


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive metadata for APACC-Sim validation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                           # Generate metadata with defaults
  %(prog)s -c simulation.yaml        # Include configuration file
  %(prog)s -o metadata.json          # Save to specific file
  %(prog)s -f markdown               # Generate Markdown format
  %(prog)s -n "Production Run 42"    # Name the experiment
        '''
    )
    
    parser.add_argument('-c', '--config', type=str,
                       help='Configuration file to include in metadata')
    parser.add_argument('-o', '--output', type=str,
                       default='metadata_apaccsim.json',
                       help='Output file path (default: metadata_apaccsim.json)')
    parser.add_argument('-f', '--format', choices=['json', 'markdown', 'both'],
                       default='both',
                       help='Output format (default: both)')
    parser.add_argument('-n', '--name', type=str,
                       help='Experiment name for identification')
    parser.add_argument('--minimal', action='store_true',
                       help='Collect minimal metadata only')
    
    args = parser.parse_args()
    
    # Create collector
    collector = MetadataCollector(
        config_file=args.config,
        experiment_name=args.name
    )
    
    # Collect metadata
    if args.minimal:
        # Minimal collection (faster)
        collector.metadata['host'] = collector._collect_host_info()
        collector.metadata['system'] = collector._collect_system_info()
        collector.metadata['python'] = collector._collect_python_info()
        collector.metadata['git'] = collector._collect_git_info()
    else:
        # Full collection
        collector.collect_all()
        
    # Generate experiment ID
    exp_id = collector.generate_experiment_id()
    collector.metadata['experiment_id'] = exp_id
    print(f"\nExperiment ID: {exp_id}")
    
    # Save metadata
    collector.save_metadata(args.output, args.format)
    
    # Print summary
    print("\nMetadata Summary:")
    print(f"  Platform: {collector.metadata.get('system', {}).get('platform', 'Unknown')}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Git Commit: {collector.metadata.get('git', {}).get('commit_hash_short', 'Unknown')}")
    print(f"  Timestamp: {collector.metadata['collection_timestamp']}")


if __name__ == '__main__':
    main()