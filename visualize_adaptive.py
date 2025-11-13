"""
Visualization Functions for Adaptive Experimental Design
Creates figures showing how adaptive design converges to correct model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")


def plot_belief_evolution(trajectory_df, true_model=None, title="Belief Evolution"):
    """
    Plot how beliefs about each model evolve over trials
    
    Args:
        trajectory_df: DataFrame from designer.get_belief_trajectory()
        true_model: name of true model to highlight
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get belief columns
    belief_cols = [col for col in trajectory_df.columns if col.startswith('belief_')]
    
    for col in belief_cols:
        model_name = col.replace('belief_', '')
        
        # Highlight true model
        if true_model and model_name == true_model:
            ax.plot(trajectory_df['trial'], trajectory_df[col], 
                   linewidth=3, label=model_name, color='green')
        else:
            ax.plot(trajectory_df['trial'], trajectory_df[col], 
                   linewidth=2, label=model_name, alpha=0.7)
    
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, 
              label='Chance (uniform)')
    ax.set_xlabel('Trial', fontsize=13, fontweight='bold')
    ax.set_ylabel('Belief (Probability)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_entropy_reduction(trajectory_df, title="Uncertainty Reduction"):
    """
    Plot how uncertainty (entropy) decreases over trials
    
    Args:
        trajectory_df: DataFrame from designer.get_belief_trajectory()
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(trajectory_df['trial'], trajectory_df['entropy'], 
           'o-', linewidth=2, markersize=6, color='steelblue')
    
    # Maximum entropy (uniform over 3 models)
    max_entropy = np.log2(3)
    ax.axhline(y=max_entropy, color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Max uncertainty: {max_entropy:.2f}')
    ax.axhline(y=0, color='green', linestyle='--', 
              linewidth=2, alpha=0.7, label='No uncertainty: 0')
    
    ax.set_xlabel('Trial', fontsize=13, fontweight='bold')
    ax.set_ylabel('Entropy (bits)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_stimulus_selection(trajectory_df, title="Adaptive Stimulus Selection"):
    """
    Show which stimuli were selected over time
    
    Args:
        trajectory_df: DataFrame from designer.get_belief_trajectory()
        title: plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Stimulus sequence
    ax1 = axes[0]
    ax1.scatter(trajectory_df['trial'], trajectory_df['stimulus'], 
               c=trajectory_df['entropy'], cmap='viridis', 
               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_ylabel('Stimulus', fontsize=12, fontweight='bold')
    ax1.set_title('Stimuli Chosen by Adaptive Design', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(int(trajectory_df['stimulus'].max()) + 1))
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Stimulus frequency histogram
    ax2 = axes[1]
    stimulus_counts = trajectory_df['stimulus'].value_counts().sort_index()
    ax2.bar(stimulus_counts.index, stimulus_counts.values, 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Stimulus', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Stimulus Selection Frequency', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_design_comparison(comparison_df, title="Design Strategy Comparison"):
    """
    Compare different experimental designs
    
    Args:
        comparison_df: DataFrame with columns: design, trial, belief_correct, entropy
        title: plot title
    """
    # Filter to only designs with belief data
    designs_with_beliefs = comparison_df[comparison_df['belief_correct'].notna()]['design'].unique()
    
    if len(designs_with_beliefs) == 0:
        print("Warning: No designs with belief tracking found")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Belief in correct model over time
    ax1 = axes[0]
    for design in designs_with_beliefs:
        design_data = comparison_df[comparison_df['design'] == design]
        
        # Calculate mean and SEM
        grouped = design_data.groupby('trial')['belief_correct']
        means = grouped.mean()
        sems = grouped.sem()
        
        ax1.plot(means.index, means, 'o-', label=design, 
                linewidth=2, markersize=5)
        ax1.fill_between(means.index, means - sems, means + sems, alpha=0.2)
    
    ax1.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, 
               label='Chance')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, 
               label='High confidence')
    ax1.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Belief in Correct Model', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Entropy over time
    ax2 = axes[1]
    for design in designs_with_beliefs:
        design_data = comparison_df[comparison_df['design'] == design]
        
        grouped = design_data.groupby('trial')['entropy']
        means = grouped.mean()
        sems = grouped.sem()
        
        ax2.plot(means.index, means, 'o-', label=design, 
                linewidth=2, markersize=5)
        ax2.fill_between(means.index, means - sems, means + sems, alpha=0.2)
    
    ax2.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax2.set_title('Uncertainty Reduction', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_efficiency_comparison(comparison_df, threshold=0.8):
    """
    Compare how many trials each design needs to reach threshold
    
    Args:
        comparison_df: DataFrame with design comparison results
        threshold: belief threshold for "convergence"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trials_to_threshold = []
    designs = []
    
    for design in comparison_df['design'].unique():
        design_data = comparison_df[comparison_df['design'] == design]
        
        # Find first trial where belief exceeds threshold
        above_threshold = design_data[design_data['belief_correct'] >= threshold]
        
        if len(above_threshold) > 0:
            trials_needed = above_threshold['trial'].min()
        else:
            trials_needed = design_data['trial'].max()  # Didn't converge
        
        trials_to_threshold.append(trials_needed)
        designs.append(design)
    
    # Bar plot
    colors = ['green' if t < comparison_df['trial'].max() else 'red' 
             for t in trials_to_threshold]
    bars = ax.bar(designs, trials_to_threshold, color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Trials to Reach Threshold', fontsize=12, fontweight='bold')
    ax.set_title(f'Design Efficiency (threshold = {threshold:.0%} belief)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars, trials_to_threshold):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(value)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_comprehensive_summary(result, comparison_df=None):
    """
    Create comprehensive summary figure with multiple panels
    
    Args:
        result: dict from simulate_adaptive_experiment()
        comparison_df: optional DataFrame for design comparison
    """
    if comparison_df is not None:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    trajectory = result['belief_trajectory']
    
    # 1. Belief evolution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    belief_cols = [col for col in trajectory.columns if col.startswith('belief_')]
    for col in belief_cols:
        model_name = col.replace('belief_', '')
        if model_name == result['true_model']:
            ax1.plot(trajectory['trial'], trajectory[col], 
                    linewidth=3, label=model_name, color='green')
        else:
            ax1.plot(trajectory['trial'], trajectory[col], 
                    linewidth=2, label=model_name, alpha=0.7)
    ax1.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Belief')
    ax1.set_title('Belief Evolution', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Entropy (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(trajectory['trial'], trajectory['entropy'], 
            'o-', linewidth=2, color='steelblue')
    ax2.axhline(y=np.log2(3), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title('Uncertainty Reduction', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Stimulus selection (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(trajectory['trial'], trajectory['stimulus'], 
               c=trajectory['entropy'], cmap='viridis', s=80, alpha=0.7)
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Stimulus')
    ax3.set_title('Adaptive Stimulus Selection', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Stimulus frequency (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    stimulus_counts = trajectory['stimulus'].value_counts().sort_index()
    ax4.bar(stimulus_counts.index, stimulus_counts.values, 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Stimulus')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Stimulus Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Design comparison (if provided)
    if comparison_df is not None:
        ax5 = fig.add_subplot(gs[2, :])
        for design in comparison_df['design'].unique():
            design_data = comparison_df[comparison_df['design'] == design]
            grouped = design_data.groupby('trial')['belief_correct']
            means = grouped.mean()
            sems = grouped.sem()
            ax5.plot(means.index, means, 'o-', label=design, linewidth=2)
            ax5.fill_between(means.index, means - sems, means + sems, alpha=0.2)
        ax5.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Trial')
        ax5.set_ylabel('Belief in Correct Model')
        ax5.set_title('Design Strategy Comparison', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    fig.suptitle('Adaptive Experimental Design Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig


# Demo
def demo_visualizations():
    """Demonstrate adaptive design visualizations"""
    from adaptive_design import (
        AdaptiveDesigner, simulate_adaptive_experiment, efficiency_analysis,
        RandomDesigner, BalancedDesigner
    )
    from models import (
        RLModel_BroadGeneralization, RLModel_ImpairedSafety,
        BayesianModel_UncertaintyAverse, DEFAULT_PARAMS
    )
    
    print("Running adaptive experiment...")
    
    # Set up
    models = [RLModel_BroadGeneralization, RLModel_ImpairedSafety, 
             BayesianModel_UncertaintyAverse]
    
    model_params = {
        'RLModel_BroadGeneralization': DEFAULT_PARAMS['ocd']['RL_broad'],
        'RLModel_ImpairedSafety': DEFAULT_PARAMS['ocd']['RL_impaired'],
        'BayesianModel_UncertaintyAverse': DEFAULT_PARAMS['ocd']['Bayesian']
    }
    
    designer = AdaptiveDesigner(
        models=models,
        model_params=model_params,
        stimulus_space=list(range(10))
    )
    
    # Run experiment
    result = simulate_adaptive_experiment(
        designer=designer,
        true_model_class=RLModel_BroadGeneralization,
        true_params=DEFAULT_PARAMS['ocd']['RL_broad'],
        n_trials=50
    )
    
    print("Creating visualizations...")
    
    # Individual plots
    fig1 = plot_belief_evolution(result['belief_trajectory'], 
                                 true_model=result['true_model'])
    fig1.savefig('belief_evolution.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_entropy_reduction(result['belief_trajectory'])
    fig2.savefig('entropy_reduction.png', dpi=150, bbox_inches='tight')
    
    fig3 = plot_stimulus_selection(result['belief_trajectory'])
    fig3.savefig('stimulus_selection.png', dpi=150, bbox_inches='tight')
    
    # Compare designs
    print("\nComparing design strategies...")
    designs = {
        'Adaptive': AdaptiveDesigner,
        'Random': RandomDesigner,
        'Balanced': BalancedDesigner
    }
    
    comparison = efficiency_analysis(
        designs=designs,
        true_model=RLModel_BroadGeneralization,
        true_params=DEFAULT_PARAMS['ocd']['RL_broad'],
        max_trials=50
    )
    
    fig4 = plot_design_comparison(comparison)
    fig4.savefig('design_comparison.png', dpi=150, bbox_inches='tight')
    
    fig5 = plot_efficiency_comparison(comparison, threshold=0.8)
    fig5.savefig('efficiency_comparison.png', dpi=150, bbox_inches='tight')
    
    # Comprehensive summary
    fig6 = create_comprehensive_summary(result, comparison)
    fig6.savefig('adaptive_summary.png', dpi=150, bbox_inches='tight')
    
    print("\nFigures saved!")
    plt.show()


if __name__ == "__main__":
    demo_visualizations()