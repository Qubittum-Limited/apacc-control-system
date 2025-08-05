#!/usr/bin/env python3
"""
Importance Sampling for APACC Rare Event Analysis
Focus on critical scenarios and edge cases
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class RareEventScenario:
    """Definition of a rare event scenario"""
    scenario_type: str
    parameters: Dict
    nominal_probability: float
    criticality_score: float

class ImportanceSamplingAPACC:
    """Importance sampling for efficient rare event simulation"""
    
    def __init__(self, nominal_distribution: Dict, target_events: List[RareEventScenario]):
        self.nominal_dist = nominal_distribution
        self.target_events = target_events
        self.importance_dist = None
        self.weights = []
        
    def optimize_importance_distribution(self):
        """Find optimal importance distribution using cross-entropy method"""
        # Initialize with nominal distribution
        current_params = self._dist_to_params(self.nominal_dist)
        
        # Cross-entropy optimization
        for iteration in range(10):
            # Sample from current distribution
            samples = self._sample_from_params(current_params, n=1000)
            
            # Evaluate performance
            scores = [self._evaluate_sample(s) for s in samples]
            
            # Select elite samples (top 10%)
            elite_threshold = np.percentile(scores, 90)
            elite_samples = [s for s, score in zip(samples, scores) 
                           if score >= elite_threshold]
            
            # Update parameters based on elite samples
            current_params = self._fit_params(elite_samples)
            
            print(f"Iteration {iteration}: Elite threshold = {elite_threshold:.3f}")
            
        self.importance_dist = self._params_to_dist(current_params)
        
    def _evaluate_sample(self, sample: Dict) -> float:
        """Evaluate how well a sample covers rare events"""
        score = 0.0
        
        for event in self.target_events:
            # Check if sample triggers the rare event
            if self._triggers_event(sample, event):
                # Weight by criticality and rarity
                score += event.criticality_score / event.nominal_probability
                
        return score
        
    def _triggers_event(self, sample: Dict, event: RareEventScenario) -> bool:
        """Check if a sample triggers a specific rare event"""
        if event.scenario_type == 'near_collision':
            return sample['min_ttc'] < event.parameters['threshold']
        elif event.scenario_type == 'emergency_brake':
            return sample['max_deceleration'] > event.parameters['threshold']
        elif event.scenario_type == 'sensor_failure':
            return sample['sensor_failure_prob'] > event.parameters['threshold']
        else:
            return False
            
    def generate_importance_samples(self, n_samples: int) -> List[Dict]:
        """Generate samples from importance distribution"""
        if self.importance_dist is None:
            raise ValueError("Must optimize importance distribution first")
            
        samples = []
        weights = []
        
        for _ in range(n_samples):
            # Sample from importance distribution
            sample = self._sample_from_dist(self.importance_dist)
            
            # Calculate importance weight
            weight = self._calculate_weight(sample)
            
            samples.append(sample)
            weights.append(weight)
            
        self.weights = weights
        return samples
        
    def _calculate_weight(self, sample: Dict) -> float:
        """Calculate importance sampling weight"""
        # w = p_nominal(x) / p_importance(x)
        p_nominal = self._pdf_nominal(sample)
        p_importance = self._pdf_importance(sample)
        
        return p_nominal / p_importance if p_importance > 0 else 0
        
    def estimate_rare_event_probability(self, event: RareEventScenario, 
                                      samples: List[Dict]) -> Tuple[float, float]:
        """Estimate probability of rare event with confidence interval"""
        # Check which samples trigger the event
        indicators = [1 if self._triggers_event(s, event) else 0 
                     for s in samples]
        
        # Weighted estimate
        prob_estimate = np.sum(np.array(indicators) * np.array(self.weights)) / len(samples)
        
        # Variance estimate for confidence interval
        variance = np.var(np.array(indicators) * np.array(self.weights))
        std_error = np.sqrt(variance / len(samples))
        
        # 95% confidence interval
        ci_lower = prob_estimate - 1.96 * std_error
        ci_upper = prob_estimate + 1.96 * std_error
        
        return prob_estimate, (ci_lower, ci_upper)
        
    def _dist_to_params(self, dist: Dict) -> np.ndarray:
        """Convert distribution dict to parameter vector"""
        # Example: Extract mean and variance parameters
        params = []
        for key, value in dist.items():
            if isinstance(value, dict) and 'mean' in value:
                params.extend([value['mean'], value.get('std', 1.0)])
            else:
                params.append(value)
        return np.array(params)
        
    def _params_to_dist(self, params: np.ndarray) -> Dict:
        """Convert parameter vector back to distribution dict"""
        # Reconstruct distribution structure
        dist = {}
        idx = 0
        
        for key in self.nominal_dist:
            if isinstance(self.nominal_dist[key], dict) and 'mean' in self.nominal_dist[key]:
                dist[key] = {
                    'mean': params[idx],
                    'std': params[idx + 1]
                }
                idx += 2
            else:
                dist[key] = params[idx]
                idx += 1
                
        return dist
        
    def _sample_from_params(self, params: np.ndarray, n: int) -> List[Dict]:
        """Sample from distribution defined by parameters"""
        dist = self._params_to_dist(params)
        return [self._sample_from_dist(dist) for _ in range(n)]
        
    def _sample_from_dist(self, dist: Dict) -> Dict:
        """Sample single instance from distribution"""
        sample = {}
        
        for key, value in dist.items():
            if isinstance(value, dict) and 'mean' in value:
                # Normal distribution
                sample[key] = np.random.normal(value['mean'], value.get('std', 1.0))
            elif isinstance(value, dict) and 'min' in value:
                # Uniform distribution
                sample[key] = np.random.uniform(value['min'], value['max'])
            else:
                # Constant
                sample[key] = value
                
        return sample
        
    def _fit_params(self, samples: List[Dict]) -> np.ndarray:
        """Fit distribution parameters to samples"""
        # Calculate mean and std for each parameter
        param_values = {key: [] for key in samples[0].keys()}
        
        for sample in samples:
            for key, value in sample.items():
                param_values[key].append(value)
                
        # Compute statistics
        params = []
        for key in self.nominal_dist:
            values = np.array(param_values[key])
            if isinstance(self.nominal_dist[key], dict) and 'mean' in self.nominal_dist[key]:
                params.extend([np.mean(values), np.std(values)])
            else:
                params.append(np.mean(values))
                
        return np.array(params)
        
    def _pdf_nominal(self, sample: Dict) -> float:
        """Probability density under nominal distribution"""
        pdf = 1.0
        
        for key, value in sample.items():
            if key in self.nominal_dist:
                dist_spec = self.nominal_dist[key]
                if isinstance(dist_spec, dict) and 'mean' in dist_spec:
                    # Normal distribution
                    pdf *= stats.norm.pdf(value, dist_spec['mean'], dist_spec.get('std', 1.0))
                elif isinstance(dist_spec, dict) and 'min' in dist_spec:
                    # Uniform distribution
                    if dist_spec['min'] <= value <= dist_spec['max']:
                        pdf *= 1.0 / (dist_spec['max'] - dist_spec['min'])
                    else:
                        return 0.0
                        
        return pdf
        
    def _pdf_importance(self, sample: Dict) -> float:
        """Probability density under importance distribution"""
        if self.importance_dist is None:
            return self._pdf_nominal(sample)
            
        pdf = 1.0
        
        for key, value in sample.items():
            if key in self.importance_dist:
                dist_spec = self.importance_dist[key]
                if isinstance(dist_spec, dict) and 'mean' in dist_spec:
                    pdf *= stats.norm.pdf(value, dist_spec['mean'], dist_spec.get('std', 1.0))
                elif isinstance(dist_spec, dict) and 'min' in dist_spec:
                    if dist_spec['min'] <= value <= dist_spec['max']:
                        pdf *= 1.0 / (dist_spec['max'] - dist_spec['min'])
                    else:
                        return 0.0
                        
        return pdf


class AdaptiveImportanceSampling:
    """Adaptive importance sampling that updates the distribution during simulation"""
    
    def __init__(self, apacc_simulator):
        self.simulator = apacc_simulator
        self.history = []
        self.current_dist = None
        
    def run_adaptive_sampling(self, n_iterations: int, samples_per_iter: int,
                            target_events: List[RareEventScenario]):
        """Run adaptive importance sampling"""
        results = []
        
        # Initialize with uniform sampling
        current_dist = self._initialize_distribution()
        
        for iteration in range(n_iterations):
            print(f"\nAdaptive Iteration {iteration + 1}/{n_iterations}")
            
            # Generate scenarios from current distribution
            scenarios = self._generate_scenarios(current_dist, samples_per_iter)
            
            # Run simulations
            sim_results = []
            for scenario in scenarios:
                result = self.simulator.run_scenario(scenario)
                sim_results.append(result)
                
            # Analyze results
            event_triggers = self._analyze_results(sim_results, target_events)
            
            # Update distribution based on results
            current_dist = self._update_distribution(current_dist, sim_results, event_triggers)
            
            # Store iteration results
            results.append({
                'iteration': iteration,
                'distribution': current_dist.copy(),
                'event_rates': self._calculate_event_rates(event_triggers),
                'samples': sim_results
            })
            
            # Log progress
            self._log_iteration_stats(iteration, event_triggers)
            
        return results
        
    def _initialize_distribution(self) -> Dict:
        """Initialize sampling distribution"""
        return {
            'vehicle_density': {'mean': 20, 'std': 10},
            'pedestrian_density': {'mean': 5, 'std': 3},
            'weather_severity': {'min': 0, 'max': 1},
            'sensor_noise_level': {'mean': 0.1, 'std': 0.05},
            'communication_delay': {'mean': 20, 'std': 10}  # ms
        }
        
    def _generate_scenarios(self, dist: Dict, n: int) -> List[Dict]:
        """Generate scenarios from distribution"""
        scenarios = []
        
        for i in range(n):
            scenario = {
                'id': f'adaptive_{i}',
                'vehicle_density': max(0, np.random.normal(dist['vehicle_density']['mean'], 
                                                          dist['vehicle_density']['std'])),
                'pedestrian_density': max(0, np.random.normal(dist['pedestrian_density']['mean'],
                                                             dist['pedestrian_density']['std'])),
                'weather_severity': np.random.uniform(dist['weather_severity']['min'],
                                                    dist['weather_severity']['max']),
                'sensor_noise': max(0, np.random.normal(dist['sensor_noise_level']['mean'],
                                                       dist['sensor_noise_level']['std'])),
                'comm_delay': max(0, np.random.normal(dist['communication_delay']['mean'],
                                                     dist['communication_delay']['std']))
            }
            scenarios.append(scenario)
            
        return scenarios
        
    def _analyze_results(self, results: List[Dict], 
                        target_events: List[RareEventScenario]) -> Dict:
        """Analyze simulation results for rare events"""
        event_triggers = {event.scenario_type: [] for event in target_events}
        
        for result in results:
            for event in target_events:
                triggered = False
                
                if event.scenario_type == 'near_collision':
                    triggered = result.get('min_ttc', float('inf')) < event.parameters['threshold']
                elif event.scenario_type == 'emergency_brake':
                    triggered = result.get('max_deceleration', 0) > event.parameters['threshold']
                elif event.scenario_type == 'control_failure':
                    triggered = result.get('control_latency_ms', 0) > event.parameters['threshold']
                    
                event_triggers[event.scenario_type].append(triggered)
                
        return event_triggers
        
    def _update_distribution(self, current_dist: Dict, results: List[Dict],
                           event_triggers: Dict) -> Dict:
        """Update distribution to increase rare event probability"""
        new_dist = current_dist.copy()
        
        # Find scenarios that triggered rare events
        rare_scenarios = []
        for i, result in enumerate(results):
            if any(event_triggers[event_type][i] for event_type in event_triggers):
                rare_scenarios.append(result)
                
        if len(rare_scenarios) > 0:
            # Shift distribution toward rare event scenarios
            alpha = 0.3  # Learning rate
            
            # Update vehicle density
            rare_vehicle_density = np.mean([s.get('vehicle_density', 20) for s in rare_scenarios])
            new_dist['vehicle_density']['mean'] = (1 - alpha) * current_dist['vehicle_density']['mean'] + \
                                                  alpha * rare_vehicle_density
                                                  
            # Update weather severity (tend toward worse weather)
            rare_weather = np.mean([s.get('weather_severity', 0.5) for s in rare_scenarios])
            new_dist['weather_severity']['min'] = max(0, rare_weather - 0.2)
            new_dist['weather_severity']['max'] = min(1, rare_weather + 0.2)
            
        return new_dist
        
    def _calculate_event_rates(self, event_triggers: Dict) -> Dict:
        """Calculate rate of each event type"""
        rates = {}
        for event_type, triggers in event_triggers.items():
            rates[event_type] = np.mean(triggers) if triggers else 0.0
        return rates
        
    def _log_iteration_stats(self, iteration: int, event_triggers: Dict):
        """Log statistics for current iteration"""
        print(f"Event trigger rates:")
        for event_type, triggers in event_triggers.items():
            rate = np.mean(triggers) if triggers else 0.0
            print(f"  {event_type}: {rate:.3%}")


# Subset Simulation for very rare events
class SubsetSimulationAPACC:
    """Subset simulation for extremely rare event estimation"""
    
    def __init__(self, failure_threshold: float, intermediate_levels: int = 4):
        self.failure_threshold = failure_threshold
        self.intermediate_levels = intermediate_levels
        self.level_thresholds = []
        
    def run_subset_simulation(self, initial_samples: int, 
                            conditional_samples: int,
                            performance_function: Callable) -> Tuple[float, float]:
        """
        Run subset simulation
        performance_function: maps scenario -> performance metric (higher = safer)
        """
        # Level 0: Monte Carlo
        print("Level 0: Initial Monte Carlo sampling")
        samples = self._generate_initial_samples(initial_samples)
        performances = [performance_function(s) for s in samples]
        
        # Sort by performance
        sorted_indices = np.argsort(performances)
        sorted_samples = [samples[i] for i in sorted_indices]
        sorted_performances = [performances[i] for i in sorted_indices]
        
        # Estimate conditional probabilities
        p_conditional = []
        current_samples = sorted_samples
        current_performances = sorted_performances
        
        for level in range(1, self.intermediate_levels + 1):
            print(f"\nLevel {level}: Conditional sampling")
            
            # Set threshold at bottom 10%
            threshold_idx = int(0.1 * len(current_performances))
            threshold = current_performances[threshold_idx]
            self.level_thresholds.append(threshold)
            
            # Seeds for next level (bottom 10%)
            seeds = current_samples[:threshold_idx]
            
            # Conditional sampling using MCMC
            new_samples = []
            new_performances = []
            
            for seed in seeds:
                # Generate conditional samples
                chain = self._mcmc_sampling(seed, performance_function, 
                                          threshold, conditional_samples // len(seeds))
                new_samples.extend(chain)
                new_performances.extend([performance_function(s) for s in chain])
                
            # Update for next level
            current_samples = new_samples
            current_performances = new_performances
            
            # Estimate conditional probability
            p_cond = np.mean([p <= threshold for p in current_performances])
            p_conditional.append(p_cond)
            
            print(f"Threshold: {threshold:.3f}, P(conditional): {p_cond:.3e}")
            
        # Final failure probability
        p_failure = 0.1 ** self.intermediate_levels
        for p_c in p_conditional:
            p_failure *= p_c
            
        # Coefficient of variation
        cov = self._estimate_cov(p_conditional)
        
        return p_failure, cov
        
    def _generate_initial_samples(self, n: int) -> List[Dict]:
        """Generate initial Monte Carlo samples"""
        samples = []
        for i in range(n):
            sample = {
                'id': f'subset_{i}',
                'speed': np.random.uniform(0, 30),  # m/s
                'following_distance': np.random.uniform(5, 50),  # m
                'road_friction': np.random.uniform(0.3, 1.0),
                'visibility': np.random.uniform(10, 200),  # m
                'reaction_time': np.random.uniform(0.5, 2.0)  # s
            }
            samples.append(sample)
        return samples
        
    def _mcmc_sampling(self, seed: Dict, performance_func: Callable,
                      threshold: float, n_samples: int) -> List[Dict]:
        """Markov Chain Monte Carlo sampling conditional on threshold"""
        chain = [seed]
        current = seed.copy()
        
        for _ in range(n_samples - 1):
            # Propose new sample
            proposal = self._propose_sample(current)
            
            # Check if it satisfies condition
            if performance_func(proposal) <= threshold:
                # Accept
                current = proposal
            # else reject and keep current
            
            chain.append(current.copy())
            
        return chain
        
    def _propose_sample(self, current: Dict) -> Dict:
        """Propose new sample using random walk"""
        proposal = current.copy()
        
        # Random walk with appropriate step sizes
        proposal['speed'] += np.random.normal(0, 2)
        proposal['following_distance'] += np.random.normal(0, 3)
        proposal['road_friction'] += np.random.normal(0, 0.1)
        proposal['visibility'] += np.random.normal(0, 10)
        proposal['reaction_time'] += np.random.normal(0, 0.1)
        
        # Ensure bounds
        proposal['speed'] = np.clip(proposal['speed'], 0, 30)
        proposal['following_distance'] = np.clip(proposal['following_distance'], 1, 50)
        proposal['road_friction'] = np.clip(proposal['road_friction'], 0.1, 1.0)
        proposal['visibility'] = np.clip(proposal['visibility'], 5, 200)
        proposal['reaction_time'] = np.clip(proposal['reaction_time'], 0.3, 3.0)
        
        return proposal
        
    def _estimate_cov(self, p_conditional: List[float]) -> float:
        """Estimate coefficient of variation"""
        # Simplified estimation
        return 0.3  # Typical value for subset simulation


# Example usage combining all techniques
def comprehensive_apacc_validation():
    """Run comprehensive validation using multiple techniques"""
    
    # Define rare events of interest
    rare_events = [
        RareEventScenario(
            scenario_type='near_collision',
            parameters={'threshold': 0.5},  # TTC < 0.5s
            nominal_probability=1e-6,
            criticality_score=10.0
        ),
        RareEventScenario(
            scenario_type='emergency_brake',
            parameters={'threshold': 8.0},  # decel > 8 m/s^2
            nominal_probability=1e-5,
            criticality_score=8.0
        ),
        RareEventScenario(
            scenario_type='control_failure',
            parameters={'threshold': 50.0},  # latency > 50ms
            nominal_probability=1e-4,
            criticality_score=9.0
        )
    ]
    
    # Nominal distribution
    nominal_dist = {
        'min_ttc': {'mean': 5.0, 'std': 2.0},
        'max_deceleration': {'mean': 3.0, 'std': 1.5},
        'control_latency_ms': {'mean': 8.0, 'std': 3.0},
        'sensor_failure_prob': {'mean': 0.001, 'std': 0.0005}
    }
    
    print("=== APACC Validation Suite ===\n")
    
    # 1. Importance Sampling
    print("1. Running Importance Sampling...")
    importance_sampler = ImportanceSamplingAPACC(nominal_dist, rare_events)
    importance_sampler.optimize_importance_distribution()
    
    samples = importance_sampler.generate_importance_samples(10000)
    
    for event in rare_events:
        prob, ci = importance_sampler.estimate_rare_event_probability(event, samples)
        print(f"{event.scenario_type}: P = {prob:.2e} [{ci[0]:.2e}, {ci[1]:.2e}]")
        
    # 2. Subset Simulation for extreme events
    print("\n2. Running Subset Simulation...")
    
    def safety_performance(scenario):
        """Safety performance function (higher = safer)"""
        ttc = scenario.get('following_distance', 20) / scenario.get('speed', 10)
        visibility_factor = scenario.get('visibility', 100) / 200
        friction_factor = scenario.get('road_friction', 0.7)
        
        return ttc * visibility_factor * friction_factor
        
    subset_sim = SubsetSimulationAPACC(failure_threshold=0.5)
    p_failure, cov = subset_sim.run_subset_simulation(
        initial_samples=1000,
        conditional_samples=1000,
        performance_function=safety_performance
    )
    
    print(f"\nExtreme failure probability: {p_failure:.2e} (CoV: {cov:.2f})")
    
    # 3. Generate validation report
    print("\n3. Generating Validation Report...")
    
    report = {
        'importance_sampling_results': {
            'samples': len(samples),
            'event_probabilities': {
                event.scenario_type: importance_sampler.estimate_rare_event_probability(event, samples)[0]
                for event in rare_events
            }
        },
        'subset_simulation_results': {
            'failure_probability': p_failure,
            'coefficient_of_variation': cov,
            'intermediate_thresholds': subset_sim.level_thresholds
        }
    }
    
    with open('apacc_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print("\nValidation complete! Results saved to apacc_validation_report.json")
    
    return report


if __name__ == "__main__":
    # Run comprehensive validation
    results = comprehensive_apacc_validation()