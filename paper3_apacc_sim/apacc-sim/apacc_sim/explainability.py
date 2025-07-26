"""
Explainability and Decision Tracking Module

Captures and analyzes decision-making processes for
certification and debugging of autonomous controllers.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DecisionTrace:
    """Single decision point with full context"""
    timestamp: float
    decision_id: str
    controller_state: Dict[str, Any]
    sensor_inputs: Dict[str, Any]
    rules_activated: List[Dict[str, float]]
    control_output: Dict[str, float]
    confidence: float
    reasoning: str
    

@dataclass
class RuleActivation:
    """Symbolic rule activation record"""
    rule_id: str
    rule_description: str
    activation_strength: float
    conditions_met: List[str]
    contribution_to_output: Dict[str, float]
    

class ExplainabilityTracker:
    """
    Tracks and analyzes controller decision-making
    
    Provides native explainability for certification
    and debugging of control decisions.
    """
    
    def __init__(self):
        """Initialize explainability tracker"""
        self.decision_traces: List[DecisionTrace] = []
        self.rule_statistics = defaultdict(lambda: {
            'activation_count': 0,
            'total_strength': 0.0,
            'control_contributions': defaultdict(float)
        })
        self.decision_patterns = []
        
    def record_decision(self, trace: DecisionTrace):
        """
        Record a control decision with full context
        
        Args:
            trace: Complete decision trace
        """
        self.decision_traces.append(trace)
        
        # Update rule statistics
        for rule in trace.rules_activated:
            rule_id = rule.get('rule_id', 'unknown')
            self.rule_statistics[rule_id]['activation_count'] += 1
            self.rule_statistics[rule_id]['total_strength'] += rule.get('strength', 0.0)
            
            # Track control contributions
            for control_dim, value in rule.get('contribution', {}).items():
                self.rule_statistics[rule_id]['control_contributions'][control_dim] += value
                
    def analyze_decision_patterns(self, window_size: int = 100) -> List[Dict[str, Any]]:
        """
        Analyze patterns in decision-making
        
        Args:
            window_size: Number of recent decisions to analyze
            
        Returns:
            List of identified patterns
        """
        if len(self.decision_traces) < window_size:
            window_size = len(self.decision_traces)
            
        recent_decisions = self.decision_traces[-window_size:]
        patterns = []
        
        # Pattern 1: Repeated rule activations
        rule_sequences = []
        for i in range(len(recent_decisions) - 1):
            curr_rules = {r['rule_id'] for r in recent_decisions[i].rules_activated}
            next_rules = {r['rule_id'] for r in recent_decisions[i+1].rules_activated}
            rule_sequences.append((curr_rules, next_rules))
            
        # Find common sequences
        sequence_counts = defaultdict(int)
        for seq in rule_sequences:
            sequence_counts[str(seq)] += 1
            
        frequent_sequences = [
            {'sequence': seq, 'count': count}
            for seq, count in sequence_counts.items()
            if count > window_size * 0.1  # >10% occurrence
        ]
        
        if frequent_sequences:
            patterns.append({
                'type': 'frequent_rule_sequences',
                'patterns': frequent_sequences
            })
            
        # Pattern 2: Confidence drops
        confidence_values = [d.confidence for d in recent_decisions]
        confidence_drops = []
        
        for i in range(1, len(confidence_values)):
            if confidence_values[i] < confidence_values[i-1] * 0.8:  # 20% drop
                confidence_drops.append({
                    'index': i,
                    'from': confidence_values[i-1],
                    'to': confidence_values[i]
                })
                
        if confidence_drops:
            patterns.append({
                'type': 'confidence_drops',
                'occurrences': len(confidence_drops),
                'average_drop': np.mean([d['from'] - d['to'] for d in confidence_drops])
            })
            
        # Pattern 3: Control oscillations
        control_history = defaultdict(list)
        for decision in recent_decisions:
            for key, value in decision.control_output.items():
                control_history[key].append(value)
                
        oscillations = {}
        for control_dim, values in control_history.items():
            if len(values) > 2:
                # Check for sign changes
                sign_changes = sum(
                    1 for i in range(1, len(values))
                    if np.sign(values[i]) != np.sign(values[i-1])
                )
                
                if sign_changes > len(values) * 0.3:  # >30% sign changes
                    oscillations[control_dim] = {
                        'sign_changes': sign_changes,
                        'frequency': sign_changes / len(values)
                    }
                    
        if oscillations:
            patterns.append({
                'type': 'control_oscillations',
                'dimensions': oscillations
            })
            
        self.decision_patterns = patterns
        return patterns
        
    def generate_decision_explanation(self, decision_id: str) -> str:
        """
        Generate human-readable explanation for a specific decision
        
        Args:
            decision_id: Unique identifier of decision
            
        Returns:
            Natural language explanation
        """
        # Find the decision
        decision = None
        for trace in self.decision_traces:
            if trace.decision_id == decision_id:
                decision = trace
                break
                
        if not decision:
            return f"Decision {decision_id} not found"
            
        # Build explanation
        explanation_parts = [
            f"Decision {decision_id} at time {decision.timestamp:.3f}s:",
            f"Confidence: {decision.confidence:.1%}"
        ]
        
        # Add reasoning if provided
        if decision.reasoning:
            explanation_parts.append(f"Reasoning: {decision.reasoning}")
            
        # Explain rules
        if decision.rules_activated:
            explanation_parts.append("\nActivated Rules:")
            for rule in sorted(decision.rules_activated, 
                             key=lambda r: r.get('strength', 0), 
                             reverse=True):
                rule_desc = rule.get('description', rule.get('rule_id', 'Unknown'))
                strength = rule.get('strength', 0.0)
                explanation_parts.append(f"  - {rule_desc} (strength: {strength:.2f})")
                
        # Explain control output
        explanation_parts.append("\nControl Output:")
        for control_dim, value in decision.control_output.items():
            explanation_parts.append(f"  - {control_dim}: {value:.3f}")
            
        return "\n".join(explanation_parts)
        
    def compute_explainability_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics for explainability quality
        
        Returns:
            Dictionary of explainability metrics
        """
        if not self.decision_traces:
            return {}
            
        metrics = {
            'total_decisions': len(self.decision_traces),
            'decisions_with_reasoning': sum(
                1 for d in self.decision_traces if d.reasoning
            ),
            'average_rules_per_decision': np.mean([
                len(d.rules_activated) for d in self.decision_traces
            ]),
            'average_confidence': np.mean([
                d.confidence for d in self.decision_traces
            ]),
            'unique_rules_used': len(self.rule_statistics),
            'rule_coverage': self._compute_rule_coverage(),
            'decision_diversity': self._compute_decision_diversity()
        }
        
        # Add pattern analysis
        patterns = self.analyze_decision_patterns()
        metrics['identified_patterns'] = len(patterns)
        metrics['has_oscillations'] = any(
            p['type'] == 'control_oscillations' for p in patterns
        )
        
        return metrics
        
    def _compute_rule_coverage(self) -> float:
        """Compute percentage of rules that have been activated"""
        # This would need to know total available rules
        # For now, estimate based on activation distribution
        activations = [stats['activation_count'] 
                      for stats in self.rule_statistics.values()]
        
        if not activations:
            return 0.0
            
        # Use Gini coefficient as proxy for coverage
        activations = sorted(activations)
        n = len(activations)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * activations)) / (n * np.sum(activations)) - (n + 1) / n
        
        # Convert to coverage score (1 - gini gives more uniform = better coverage)
        return 1 - gini
        
    def _compute_decision_diversity(self) -> float:
        """Compute diversity of control decisions"""
        if len(self.decision_traces) < 2:
            return 0.0
            
        # Compute variance in control outputs
        control_vectors = []
        for trace in self.decision_traces:
            vector = [trace.control_output.get(dim, 0.0) 
                     for dim in ['steering', 'throttle', 'brake']]
            control_vectors.append(vector)
            
        control_array = np.array(control_vectors)
        
        # Normalize by standard deviation
        diversity = np.mean(np.std(control_array, axis=0))
        
        return float(diversity)
        
    def export_decision_log(self, output_path: Path, format: str = 'json'):
        """
        Export complete decision log
        
        Args:
            output_path: Path to save log
            format: Output format (json, csv, parquet)
        """
        if format == 'json':
            self._export_json_log(output_path)
        elif format == 'csv':
            self._export_csv_log(output_path)
        elif format == 'parquet':
            self._export_parquet_log(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _export_json_log(self, output_path: Path):
        """Export decision log as JSON"""
        log_data = {
            'metadata': {
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'total_decisions': len(self.decision_traces)
            },
            'decisions': [asdict(trace) for trace in self.decision_traces],
            'rule_statistics': dict(self.rule_statistics),
            'patterns': self.decision_patterns,
            'metrics': self.compute_explainability_metrics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
            
        logger.info(f"Decision log exported to {output_path}")
        
    def _export_csv_log(self, output_path: Path):
        """Export decision log as CSV (simplified)"""
        rows = []
        
        for trace in self.decision_traces:
            row = {
                'timestamp': trace.timestamp,
                'decision_id': trace.decision_id,
                'confidence': trace.confidence,
                'num_rules': len(trace.rules_activated),
                'steering': trace.control_output.get('steering', 0.0),
                'throttle': trace.control_output.get('throttle', 0.0),
                'brake': trace.control_output.get('brake', 0.0),
                'reasoning': trace.reasoning
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Decision log (CSV) exported to {output_path}")
        
    def _export_parquet_log(self, output_path: Path):
        """Export decision log as Parquet"""
        # Flatten nested structures for Parquet
        rows = []
        
        for trace in self.decision_traces:
            base_row = {
                'timestamp': trace.timestamp,
                'decision_id': trace.decision_id,
                'confidence': trace.confidence,
                'reasoning': trace.reasoning
            }
            
            # Add control outputs
            for key, value in trace.control_output.items():
                base_row[f'control_{key}'] = value
                
            # Add top 3 rules
            top_rules = sorted(trace.rules_activated, 
                             key=lambda r: r.get('strength', 0), 
                             reverse=True)[:3]
            
            for i, rule in enumerate(top_rules):
                base_row[f'rule_{i}_id'] = rule.get('rule_id', '')
                base_row[f'rule_{i}_strength'] = rule.get('strength', 0.0)
                
            rows.append(base_row)
            
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, compression='snappy')
        
        logger.info(f"Decision log (Parquet) exported to {output_path}")
        
    def visualize_rule_activation_heatmap(self, output_path: Path):
        """
        Generate rule activation heatmap visualization
        
        Args:
            output_path: Path to save visualization
        """
        # This would generate a heatmap showing rule activation patterns over time
        # Implementation would use matplotlib or similar
        logger.info("Visualization not implemented in base version")
        
    def save_traces(self, output_path: Path):
        """Save decision traces for later analysis"""
        self.export_decision_log(output_path, format='json')