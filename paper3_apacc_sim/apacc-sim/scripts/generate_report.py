#!/usr/bin/env python3
"""
Generate Report Script

Creates comprehensive validation reports from
APACC-Sim results in various formats.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from apacc_sim.metrics import MetricsCollector
from apacc_sim.explainability import ExplainabilityTracker


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from file based on extension"""
    path = Path(results_path)
    
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_latex_report(results: pd.DataFrame, output_path: Path):
    """Generate LaTeX report for academic publication"""
    
    # Compute statistics
    total_scenarios = len(results)
    simulators = results['simulator'].unique() if 'simulator' in results else ['unknown']
    
    collision_rate = results['collision'].mean() * 100 if 'collision' in results else 0
    avg_latency = results['avg_control_latency'].mean() if 'avg_control_latency' in results else 0
    p99_latency = results['p99_control_latency'].mean() if 'p99_control_latency' in results else 0
    
    # Group by simulator
    by_simulator = results.groupby('simulator').agg({
        'collision': ['count', 'sum', 'mean'],
        'avg_control_latency': 'mean'
    }).round(3)
    
    latex_template = r"""
\documentclass{article}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{graphicx}

\title{APACC-Sim Validation Report}
\date{\today}

\begin{document}
\maketitle

\section{Executive Summary}

The validation campaign evaluated the controller across \num{%(total_scenarios)d} scenarios 
using %(num_simulators)d simulation paradigms. The overall collision rate was 
\SI{%(collision_rate).2f}{\percent} with an average control latency of 
\SI{%(avg_latency).2f}{\milli\second}.

\section{Detailed Results}

\subsection{Performance by Simulator}

\begin{table}[h]
\centering
\caption{Controller performance across simulation environments}
\label{tab:simulator_results}
\begin{tabular}{lrrrr}
\toprule
Simulator & Scenarios & Collisions & Collision Rate & Avg Latency (ms) \\
\midrule
%(simulator_rows)s
\bottomrule
\end{tabular}
\end{table}

\subsection{Safety Metrics}

\begin{itemize}
    \item Total collisions: %(total_collisions)d
    \item Collision rate: \SI{%(collision_rate).2f}{\percent}
    \item Confidence interval (95%%): [%(ci_lower).3f, %(ci_upper).3f]
\end{itemize}

\subsection{Performance Metrics}

\begin{itemize}
    \item Average control latency: \SI{%(avg_latency).2f}{\milli\second}
    \item P99 control latency: \SI{%(p99_latency).2f}{\milli\second}
    \item Real-time compliance: %(rt_compliance).1f%%
\end{itemize}

\section{Certification Compliance}

The validation results demonstrate compliance with:
\begin{itemize}
    \item ISO 26262 ASIL-D: %(iso26262_status)s
    \item ISO 21448 SOTIF: %(sotif_status)s
\end{itemize}

\end{document}
"""
    
    # Build simulator rows for table
    simulator_rows = []
    for sim in by_simulator.index:
        row_data = by_simulator.loc[sim]
        scenarios = int(row_data[('collision', 'count')])
        collisions = int(row_data[('collision', 'sum')])
        collision_rate = row_data[('collision', 'mean')] * 100
        latency = row_data[('avg_control_latency', 'mean')]
        
        simulator_rows.append(
            f"{sim.capitalize()} & {scenarios} & {collisions} & "
            f"{collision_rate:.2f} & {latency:.2f} \\\\"
        )
    
    # Calculate confidence interval
    n = len(results)
    p = collision_rate / 100
    z = 1.96  # 95% confidence
    margin = z * ((p * (1 - p)) / n) ** 0.5
    ci_lower = max(0, p - margin) * 100
    ci_upper = min(1, p + margin) * 100
    
    # Fill template
    latex_content = latex_template % {
        'total_scenarios': total_scenarios,
        'num_simulators': len(simulators),
        'collision_rate': collision_rate,
        'avg_latency': avg_latency,
        'p99_latency': p99_latency,
        'simulator_rows': '\n'.join(simulator_rows),
        'total_collisions': results['collision'].sum() if 'collision' in results else 0,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rt_compliance': (results['avg_control_latency'] < 10).mean() * 100 if 'avg_control_latency' in results else 0,
        'iso26262_status': 'PASS' if collision_rate < 0.01 else 'REQUIRES REVIEW',
        'sotif_status': 'PASS' if collision_rate < 0.1 else 'REQUIRES REVIEW'
    }
    
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    logging.info(f"LaTeX report saved to {output_path}")


def generate_markdown_report(results: pd.DataFrame, output_path: Path):
    """Generate Markdown report for documentation"""
    
    metrics_collector = MetricsCollector()
    metrics_dict = [{
        'collision': row.get('collision', False),
        'time_to_collision': row.get('time_to_collision', float('inf')),
        'control_latency_ms': row.get('avg_control_latency', 0)
    } for _, row in results.iterrows()]
    
    aggregated = metrics_collector.aggregate_metrics(metrics_dict)
    
    md_content = f"""# APACC-Sim Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total Scenarios**: {len(results):,}
- **Simulation Environments**: {', '.join(results['simulator'].unique() if 'simulator' in results else ['N/A'])}
- **Overall Safety Score**: {aggregated.get('safety_score', 0):.1f}/100

## Safety Performance

| Metric | Value |
|--------|-------|
| Collision Rate | {aggregated.get('collision_rate', 0):.2f}% |
| Total Collisions | {aggregated.get('collision_count', 0)} |
| Near Misses | {aggregated.get('total_near_misses', 0)} |
| Average TTC | {aggregated.get('avg_ttc', float('inf')):.2f}s |
| Minimum TTC | {aggregated.get('min_ttc', float('inf')):.2f}s |

## Control Performance

| Metric | Value |
|--------|-------|
| Average Latency | {aggregated.get('avg_control_latency', 0):.2f} ms |
| P99 Latency | {aggregated.get('p99_control_latency', 0):.2f} ms |
| Real-time Compliance | {(results['avg_control_latency'] < 10).mean() * 100 if 'avg_control_latency' in results else 0:.1f}% |

## Results by Simulator

"""
    
    # Add per-simulator results
    if 'simulator' in results:
        for simulator in results['simulator'].unique():
            sim_data = results[results['simulator'] == simulator]
            md_content += f"""
### {simulator.upper()}

- Scenarios: {len(sim_data)}
- Collision Rate: {sim_data['collision'].mean() * 100:.2f}%
- Avg Latency: {sim_data['avg_control_latency'].mean():.2f} ms
"""
    
    md_content += """
## Certification Assessment

### ISO 26262 (Functional Safety)
- **Status**: {}
- **Collision Rate**: {:.2e} (Target: < 1e-6)

### ISO 21448 (SOTIF)
- **Status**: {}
- **Edge Cases**: {} identified

---
*Report generated by APACC-Sim v1.0.0*
""".format(
        'PASS' if aggregated.get('collision_rate', 100) < 0.01 else 'REVIEW REQUIRED',
        aggregated.get('collision_rate', 0) / 100,
        'PASS' if aggregated.get('collision_rate', 100) < 0.1 else 'REVIEW REQUIRED',
        'Multiple' if aggregated.get('total_near_misses', 0) > 10 else 'Few'
    )
    
    with open(output_path, 'w') as f:
        f.write(md_content)
    
    logging.info(f"Markdown report saved to {output_path}")


def generate_plots(results: pd.DataFrame, output_dir: Path):
    """Generate visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Collision rate by simulator
        if 'simulator' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            collision_rates = results.groupby('simulator')['collision'].mean() * 100
            collision_rates.plot(kind='bar', ax=ax)
            ax.set_ylabel('Collision Rate (%)')
            ax.set_title('Collision Rate by Simulator')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'collision_by_simulator.png', dpi=300)
            plt.close()
        
        # Plot 2: Control latency distribution
        if 'avg_control_latency' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            results['avg_control_latency'].hist(bins=50, ax=ax)
            ax.axvline(10, color='r', linestyle='--', label='10ms deadline')
            ax.set_xlabel('Control Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Control Latency Distribution')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'latency_distribution.png', dpi=300)
            plt.close()
        
        # Plot 3: Time-to-collision distribution
        if 'time_to_collision' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            ttc_finite = results[results['time_to_collision'] < float('inf')]['time_to_collision']
            if len(ttc_finite) > 0:
                ttc_finite.hist(bins=50, ax=ax)
                ax.axvline(2.0, color='r', linestyle='--', label='2s safety threshold')
                ax.set_xlabel('Time to Collision (s)')
                ax.set_ylabel('Frequency')
                ax.set_title('Time-to-Collision Distribution')
                ax.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'ttc_distribution.png', dpi=300)
                plt.close()
        
        logging.info(f"Plots saved to {output_dir}")
        
    except ImportError:
        logging.warning("Matplotlib not available - skipping plots")


def main():
    """Main report generation"""
    parser = argparse.ArgumentParser(
        description="Generate validation reports from APACC-Sim results"
    )
    
    parser.add_argument(
        'results',
        help='Path to results file (parquet/csv/json)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['html', 'latex', 'markdown', 'all'],
        default='markdown',
        help='Report format to generate'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load results
    logging.info(f"Loading results from {args.results}")
    try:
        results = load_results(args.results)
        logging.info(f"Loaded {len(results)} scenarios")
    except Exception as e:
        logging.error(f"Failed to load results: {e}")
        return 1
    
    # Determine output path
    if args.output:
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
    else:
        output_base = Path(args.results).stem + "_report"
        output_dir = Path(args.results).parent
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate reports
    try:
        if args.format in ['latex', 'all']:
            generate_latex_report(results, output_dir / f"{output_base}.tex")
        
        if args.format in ['markdown', 'all']:
            generate_markdown_report(results, output_dir / f"{output_base}.md")
        
        if args.format in ['html', 'all']:
            # Convert markdown to HTML using the MetricsCollector export
            metrics_collector = MetricsCollector()
            metrics_collector.metrics_history = results.to_dict('records')
            metrics_collector.export_metrics_report(
                str(output_dir / f"{output_base}.html"),
                format='html'
            )
        
        if args.plots:
            plots_dir = output_dir / f"{output_base}_plots"
            generate_plots(results, plots_dir)
        
        logging.info(f"Report generation complete. Output in: {output_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Report generation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())