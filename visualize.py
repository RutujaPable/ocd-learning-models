"""
Visualization Functions
Create publication-ready figures for analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("colorblind")

def plot_learning_curves(data, group_col='group', trial_col='trial', 
                         error_col='error', title="Learning Curves"):
    """
    Plot learning curves comparing groups
    
    Args:
        data: DataFrame with columns for group, trial, and error/prediction
        group_col: column name for group identity
        trial_col: column name for trial number
        error_col: column name for metric to plot
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and SEM for each group by trial
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        
        means = group_data.groupby(trial_col)[error_col].mean()
        sems = group_data.groupby(trial_col)[error_col].sem()
        
        trials = means.index
        
        ax.plot(trials, means, label=group, linewidth=2)
        ax.fill_between(trials, means - sems, means + sems, alpha=0.2)
    
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel(error_col.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_generalization_gradient(model, stimulus_range=None, title="Generalization Gradient"):
    """
    Plot how predictions vary across stimulus space
    Shows generalization from learned stimuli
    
    Args:
        model: trained model instance
        stimulus_range: range of stimuli to test (default: 0 to n_stimuli)
        title: plot title
    """
    if stimulus_range is None:
        stimulus_range = range(len(model.values) if hasattr(model, 'values') 
                              else len(model.means))
    
    predictions = [model.predict(s) for s in stimulus_range]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stimulus_range, predictions, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Stimulus', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_group_comparison(comparison_dict, metric='error', title="Group Comparison"):
    """
    Bar plot comparing two groups on a metric
    
    Args:
        comparison_dict: output from simulator.compare_populations()
        metric: what was compared
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    groups = [comparison_dict['group1_name'], comparison_dict['group2_name']]
    means = [comparison_dict['group1_mean'], comparison_dict['group2_mean']]
    stds = [comparison_dict['group1_std'], comparison_dict['group2_std']]
    
    x_pos = np.arange(len(groups))
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel(f'Mean {metric.capitalize()}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add effect size annotation
    effect_size = comparison_dict['effect_size']
    ax.text(0.5, max(means) + max(stds) * 1.2, 
            f"Effect Size (Cohen's d): {effect_size:.3f}",
            ha='center', fontsize=11, style='italic')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return fig


def plot_parameter_distributions(population, title="Parameter Distributions"):
    """
    Plot distributions of parameters in a population
    
    Args:
        population: Population object
        title: plot title
    """
    stats = population.summary_stats()
    n_params = len([k for k in stats.keys() if k != 'n_stimuli'])
    
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    for idx, (param, param_stats) in enumerate(stats.items()):
        if param == 'n_stimuli':
            continue
            
        ax = axes[idx]
        
        # Get individual values
        values = [s['params'][param] for s in population.subjects]
        
        # Histogram
        ax.hist(values, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Mark mean
        mean = param_stats['mean']
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
                  label=f"Mean: {mean:.3f}")
        
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f"{param} Distribution", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"{title} ({population.group_name})", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_prediction_errors_by_stimulus(data, title="Prediction Errors by Stimulus"):
    """
    Violin plot of prediction errors for each stimulus
    
    Args:
        data: DataFrame from simulate_experiment()
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.violinplot(data=data, x='stimulus', y='error', hue='group', 
                   split=True, ax=ax, inner='quartile')
    
    ax.set_xlabel('Stimulus', fontsize=12)
    ax.set_ylabel('Prediction Error', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, title='Group')
    
    plt.tight_layout()
    return fig


def plot_learning_trajectory(subject_history, title="Individual Learning Trajectory"):
    """
    Plot detailed learning trajectory for a single subject
    
    Args:
        subject_history: model.get_history() from a trained model
        title: plot title
    """
    if not subject_history:
        print("No history available. Model needs to be trained first.")
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    trials = range(len(subject_history))
    predictions = [h['prediction'] for h in subject_history]
    outcomes = [h['outcome'] for h in subject_history]
    errors = [abs(h['prediction'] - h['outcome']) for h in subject_history]
    
    # Top: Predictions vs Outcomes
    ax1 = axes[0]
    ax1.plot(trials, predictions, 'o-', label='Predictions', alpha=0.7, markersize=4)
    ax1.plot(trials, outcomes, 's-', label='Outcomes', alpha=0.7, markersize=4)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Bottom: Errors over time
    ax2 = axes[1]
    ax2.plot(trials, errors, 'o-', color='red', alpha=0.7, markersize=4)
    ax2.set_xlabel('Trial', fontsize=12)
    ax2.set_ylabel('Prediction Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_multiple_subjects(population, trial_sequence, n_subjects=5):
    """
    Plot learning trajectories for multiple subjects
    
    Args:
        population: Population object
        trial_sequence: experiment trials
        n_subjects: how many subjects to plot
    """
    from simulator import simulate_experiment
    
    fig, axes = plt.subplots(n_subjects, 1, figsize=(12, 3*n_subjects), sharex=True)
    if n_subjects == 1:
        axes = [axes]
    
    # Simulate for each subject
    for idx in range(min(n_subjects, len(population.subjects))):
        subject = population.subjects[idx]
        model = subject['model']
        
        # Reset and run through trials
        model.reset()
        predictions = []
        outcomes = []
        
        for stimulus, outcome in trial_sequence:
            pred = model.predict(stimulus)
            predictions.append(pred)
            outcomes.append(outcome)
            model.learn(stimulus, outcome)
        
        # Plot
        ax = axes[idx]
        trials = range(len(trial_sequence))
        ax.plot(trials, predictions, 'o-', label='Predictions', 
               alpha=0.6, markersize=3)
        ax.plot(trials, outcomes, 's-', label='Outcomes', 
               alpha=0.6, markersize=3)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f"Subject {subject['id']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    axes[-1].set_xlabel('Trial', fontsize=12)
    fig.suptitle(f"Individual Learning Trajectories ({population.group_name})", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_summary_figure(control_pop, ocd_pop, trial_sequence):
    """
    Create comprehensive summary figure with multiple panels
    
    Args:
        control_pop: Control Population
        ocd_pop: OCD Population  
        trial_sequence: experiment trials
    """
    from simulator import simulate_experiment, compare_populations
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Run simulations
    control_data = simulate_experiment(control_pop, trial_sequence)
    control_data['group'] = 'Control'
    ocd_data = simulate_experiment(ocd_pop, trial_sequence)
    ocd_data['group'] = 'OCD'
    combined_data = pd.concat([control_data, ocd_data])
    
    # 1. Learning curves (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    for group in ['Control', 'OCD']:
        group_data = combined_data[combined_data['group'] == group]
        means = group_data.groupby('trial')['error'].mean()
        sems = group_data.groupby('trial')['error'].sem()
        ax1.plot(means.index, means, label=group, linewidth=2)
        ax1.fill_between(means.index, means-sems, means+sems, alpha=0.2)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Prediction Error')
    ax1.set_title('Learning Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Group comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    comparison = compare_populations(control_pop, ocd_pop, trial_sequence)
    groups = ['Control', 'OCD']
    means = [comparison['group1_mean'], comparison['group2_mean']]
    stds = [comparison['group1_std'], comparison['group2_std']]
    ax2.bar(groups, means, yerr=stds, capsize=10, alpha=0.7, 
           color=['steelblue', 'coral'])
    ax2.set_ylabel('Mean Error')
    ax2.set_title('Group Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Prediction errors by stimulus (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    for group, color in zip(['Control', 'OCD'], ['steelblue', 'coral']):
        group_data = combined_data[combined_data['group'] == group]
        stim_errors = group_data.groupby('stimulus')['error'].mean()
        ax3.plot(stim_errors.index, stim_errors, 'o-', label=group, 
                color=color, linewidth=2, markersize=8)
    ax3.set_xlabel('Stimulus')
    ax3.set_ylabel('Mean Prediction Error')
    ax3.set_title('Errors by Stimulus', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter distributions (bottom row)
    stats_control = control_pop.summary_stats()
    stats_ocd = ocd_pop.summary_stats()
    
    param_idx = 0
    for param in stats_control.keys():
        if param == 'n_stimuli' or param_idx >= 3:
            continue
            
        ax = fig.add_subplot(gs[2, param_idx])
        
        control_vals = [s['params'][param] for s in control_pop.subjects]
        ocd_vals = [s['params'][param] for s in ocd_pop.subjects]
        
        ax.hist([control_vals, ocd_vals], bins=15, label=['Control', 'OCD'],
               alpha=0.7, color=['steelblue', 'coral'])
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.set_title(f'{param} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        param_idx += 1
    
    fig.suptitle('Simulation Summary', fontsize=16, fontweight='bold', y=0.995)
    
    return fig


# Quick demo
def demo_visualizations():
    """Demonstrate visualization functions"""
    from simulator import create_standard_populations, simulate_experiment
    from experiments import create_standard_experiment
    
    print("Creating populations and running experiment...")
    control, ocd = create_standard_populations('RL_broad', n_subjects=30)
    trials = create_standard_experiment('discrimination', n_trials=100)
    
    print("Generating visualizations...")
    
    # Individual plots
    control_data = simulate_experiment(control, trials)
    control_data['group'] = 'Control'
    ocd_data = simulate_experiment(ocd, trials)
    ocd_data['group'] = 'OCD'
    combined = pd.concat([control_data, ocd_data])
    
    fig1 = plot_learning_curves(combined)
    fig1.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_parameter_distributions(control)
    fig2.savefig('control_params.png', dpi=150, bbox_inches='tight')
    
    fig3 = create_summary_figure(control, ocd, trials)
    fig3.savefig('summary.png', dpi=150, bbox_inches='tight')
    
    print("Figures saved!")
    plt.show()


if __name__ == "__main__":
    demo_visualizations()