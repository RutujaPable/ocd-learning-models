"""
Parameter Space Exploration
Systematically varies parameters to see how model predictions change
Identifies where theories make different predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import (
    RLModel_BroadGeneralization,
    RLModel_ImpairedSafety,
    BayesianModel_UncertaintyAverse
)
from experiments import create_standard_experiment
from simulator import simulate_experiment, Population


def explore_2d_parameter_space(model_class, param1_name, param1_range, 
                                param2_name, param2_range, 
                                fixed_params, trial_sequence, metric='final_error'):
    """
    Explore a 2D slice of parameter space
    
    Args:
        model_class: which model to test
        param1_name, param2_name: parameters to vary
        param1_range, param2_range: arrays of values to test
        fixed_params: dict of other parameters to hold constant
        trial_sequence: experiment trials
        metric: what to measure ('final_error', 'mean_error', 'learning_rate')
        
    Returns:
        2D array of metric values
    """
    results = np.zeros((len(param1_range), len(param2_range)))
    
    for i, param1_val in enumerate(param1_range):
        for j, param2_val in enumerate(param2_range):
            # Set parameters
            params = fixed_params.copy()
            params[param1_name] = param1_val
            params[param2_name] = param2_val
            params['n_stimuli'] = 10
            
            # Create model
            model = model_class(params)
            
            # Run through experiment
            errors = []
            for stimulus, outcome in trial_sequence:
                pred = model.predict(stimulus)
                errors.append(abs(pred - outcome))
                model.learn(stimulus, outcome)
            
            # Calculate metric
            if metric == 'final_error':
                results[i, j] = np.mean(errors[-20:])  # Last 20 trials
            elif metric == 'mean_error':
                results[i, j] = np.mean(errors)
            elif metric == 'learning_rate':
                # How much error decreases from first to last quarter
                first_quarter = np.mean(errors[:len(errors)//4])
                last_quarter = np.mean(errors[-len(errors)//4:])
                results[i, j] = first_quarter - last_quarter
            
    return results


def plot_parameter_heatmap(param1_range, param2_range, results, 
                           param1_name, param2_name, metric_name,
                           title=None):
    """
    Create heatmap of parameter space exploration
    
    Args:
        param1_range, param2_range: parameter values tested
        results: 2D array of metric values
        param1_name, param2_name: parameter names
        metric_name: what was measured
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(results, aspect='auto', origin='lower', cmap='viridis')
    
    # Set ticks
    ax.set_xticks(np.arange(len(param2_range))[::max(1, len(param2_range)//10)])
    ax.set_xticklabels([f'{x:.2f}' for x in param2_range[::max(1, len(param2_range)//10)]])
    ax.set_yticks(np.arange(len(param1_range))[::max(1, len(param1_range)//10)])
    ax.set_yticklabels([f'{y:.2f}' for y in param1_range[::max(1, len(param1_range)//10)]])
    
    ax.set_xlabel(param2_name, fontsize=13, fontweight='bold')
    ax.set_ylabel(param1_name, fontsize=13, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title(f'{metric_name} across {param1_name} and {param2_name}',
                    fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def compare_models_across_parameters(param_name, param_range, trial_sequence,
                                     control_params, ocd_params, 
                                     model_classes):
    """
    Compare how different models respond to parameter changes
    
    Args:
        param_name: which parameter to vary
        param_range: values to test
        trial_sequence: experiment
        control_params: base parameters for control
        ocd_params: base parameters for OCD
        model_classes: list of model classes to compare
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_class in model_classes:
        for param_val in param_range:
            # Control condition
            control_p = control_params[model_class.__name__].copy()
            control_p[param_name] = param_val
            control_p['n_stimuli'] = 10
            
            control_model = model_class(control_p)
            control_errors = []
            for stim, out in trial_sequence:
                pred = control_model.predict(stim)
                control_errors.append(abs(pred - out))
                control_model.learn(stim, out)
            
            # OCD condition
            ocd_p = ocd_params[model_class.__name__].copy()
            ocd_p[param_name] = param_val
            ocd_p['n_stimuli'] = 10
            
            ocd_model = model_class(ocd_p)
            ocd_errors = []
            for stim, out in trial_sequence:
                pred = ocd_model.predict(stim)
                ocd_errors.append(abs(pred - out))
                ocd_model.learn(stim, out)
            
            # Record
            results.append({
                'model': model_class.__name__,
                param_name: param_val,
                'control_error': np.mean(control_errors),
                'ocd_error': np.mean(ocd_errors),
                'difference': np.mean(ocd_errors) - np.mean(control_errors)
            })
    
    return pd.DataFrame(results)


def identify_divergence_points(model1_class, model2_class, 
                               param1_name, param1_range,
                               param2_name, param2_range,
                               fixed_params1, fixed_params2,
                               trial_sequence):
    """
    Find where two models make most different predictions
    
    Args:
        model1_class, model2_class: models to compare
        param1_name, param1_range: first parameter to vary
        param2_name, param2_range: second parameter to vary
        fixed_params1, fixed_params2: other parameters for each model
        trial_sequence: experiment
        
    Returns:
        2D array of prediction divergence
    """
    divergence = np.zeros((len(param1_range), len(param2_range)))
    
    for i, param1_val in enumerate(param1_range):
        for j, param2_val in enumerate(param2_range):
            # Model 1
            params1 = fixed_params1.copy()
            params1[param1_name] = param1_val
            params1[param2_name] = param2_val
            params1['n_stimuli'] = 10
            
            model1 = model1_class(params1)
            predictions1 = []
            
            for stim, out in trial_sequence:
                predictions1.append(model1.predict(stim))
                model1.learn(stim, out)
            
            # Model 2
            params2 = fixed_params2.copy()
            params2[param1_name] = param1_val
            params2[param2_name] = param2_val
            params2['n_stimuli'] = 10
            
            model2 = model2_class(params2)
            predictions2 = []
            
            for stim, out in trial_sequence:
                predictions2.append(model2.predict(stim))
                model2.learn(stim, out)
            
            # Calculate divergence (mean absolute difference)
            divergence[i, j] = np.mean(np.abs(np.array(predictions1) - np.array(predictions2)))
    
    return divergence


def plot_model_comparison(comparison_df, param_name):
    """
    Plot how different models respond to parameter changes
    
    Args:
        comparison_df: output from compare_models_across_parameters
        param_name: which parameter was varied
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Absolute errors
    ax1 = axes[0]
    for model in comparison_df['model'].unique():
        model_data = comparison_df[comparison_df['model'] == model]
        ax1.plot(model_data[param_name], model_data['control_error'], 
                'o-', label=f'{model} (Control)', linewidth=2, markersize=6)
        ax1.plot(model_data[param_name], model_data['ocd_error'],
                's--', label=f'{model} (OCD)', linewidth=2, markersize=6)
    
    ax1.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Prediction Error', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance vs Parameter', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Group differences
    ax2 = axes[1]
    for model in comparison_df['model'].unique():
        model_data = comparison_df[comparison_df['model'] == model]
        ax2.plot(model_data[param_name], model_data['difference'],
                'o-', label=model, linewidth=2, markersize=7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax2.set_ylabel('OCD - Control Error', fontsize=12, fontweight='bold')
    ax2.set_title('Group Difference vs Parameter', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def comprehensive_exploration():
    """Run comprehensive parameter space exploration"""
    print("=" * 60)
    print("PARAMETER SPACE EXPLORATION")
    print("=" * 60)
    
    # Generate experiment
    trial_sequence = create_standard_experiment('discrimination', n_trials=100)
    
    # 1. Explore alpha x sigma for RL Broad Generalization
    print("\n1. Exploring alpha x sigma for RL Broad Generalization...")
    
    alphas = np.linspace(0.1, 0.9, 20)
    sigmas = np.linspace(0.5, 3.5, 20)
    
    results_rl = explore_2d_parameter_space(
        model_class=RLModel_BroadGeneralization,
        param1_name='alpha',
        param1_range=alphas,
        param2_name='sigma',
        param2_range=sigmas,
        fixed_params={'noise': 0.1},
        trial_sequence=trial_sequence,
        metric='final_error'
    )
    
    fig1 = plot_parameter_heatmap(
        alphas, sigmas, results_rl,
        'Learning Rate (alpha)', 'Generalization Width (sigma)',
        'Final Error',
        title='RL Broad Generalization: Parameter Space'
    )
    fig1.savefig('param_space_alpha_sigma.png', dpi=150, bbox_inches='tight')
    print("   Saved: param_space_alpha_sigma.png")
    
    # 2. Compare models as sigma varies
    print("\n2. Comparing models across generalization width (sigma)...")
    
    sigmas_test = np.linspace(0.5, 3.5, 15)
    
    control_params = {
        'RLModel_BroadGeneralization': {'alpha': 0.3, 'noise': 0.1},
        'RLModel_ImpairedSafety': {'alpha_pos': 0.3, 'alpha_neg': 0.3, 'noise': 0.1},
        'BayesianModel_UncertaintyAverse': {'uncertainty_weight': 0.1, 
                                            'prior_mean': 0.5, 'prior_variance': 0.25,
                                            'noise': 0.1}
    }
    
    ocd_params = {
        'RLModel_BroadGeneralization': {'alpha': 0.3, 'noise': 0.1},
        'RLModel_ImpairedSafety': {'alpha_pos': 0.15, 'alpha_neg': 0.35, 'noise': 0.1},
        'BayesianModel_UncertaintyAverse': {'uncertainty_weight': 0.4,
                                            'prior_mean': 0.5, 'prior_variance': 0.25,
                                            'noise': 0.1}
    }
    
    comparison = compare_models_across_parameters(
        param_name='sigma',
        param_range=sigmas_test,
        trial_sequence=trial_sequence,
        control_params=control_params,
        ocd_params=ocd_params,
        model_classes=[RLModel_BroadGeneralization, RLModel_ImpairedSafety, 
                      BayesianModel_UncertaintyAverse]
    )
    
    fig2 = plot_model_comparison(comparison, 'sigma')
    fig2.savefig('model_comparison_sigma.png', dpi=150, bbox_inches='tight')
    print("   Saved: model_comparison_sigma.png")
    
    # 3. Identify divergence between theories
    print("\n3. Finding where theories diverge most...")
    
    divergence = identify_divergence_points(
        model1_class=RLModel_BroadGeneralization,
        model2_class=RLModel_ImpairedSafety,
        param1_name='alpha',
        param1_range=np.linspace(0.1, 0.9, 15),
        param2_name='sigma',
        param2_range=np.linspace(0.5, 3.5, 15),
        fixed_params1={'noise': 0.1},
        fixed_params2={'alpha_pos': 0.3, 'alpha_neg': 0.3, 'noise': 0.1},
        trial_sequence=trial_sequence
    )
    
    fig3 = plot_parameter_heatmap(
        np.linspace(0.1, 0.9, 15),
        np.linspace(0.5, 3.5, 15),
        divergence,
        'Learning Rate (alpha)',
        'Generalization Width (sigma)',
        'Prediction Divergence',
        title='Theory Divergence: RL Broad vs RL Impaired'
    )
    fig3.savefig('theory_divergence.png', dpi=150, bbox_inches='tight')
    print("   Saved: theory_divergence.png")
    
    # Save comparison data
    comparison.to_csv('model_comparison_results.csv', index=False)
    print("\n   Saved: model_comparison_results.csv")
    
    # 4. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nOptimal parameters (lowest final error):")
    min_idx = np.unravel_index(np.argmin(results_rl), results_rl.shape)
    print(f"  alpha: {alphas[min_idx[0]]:.3f}")
    print(f"  sigma: {sigmas[min_idx[1]]:.3f}")
    print(f"  Error: {results_rl[min_idx]:.4f}")
    
    print("\nGroup differences across sigma:")
    for model in comparison['model'].unique():
        model_data = comparison[comparison['model'] == model]
        max_diff_idx = model_data['difference'].abs().idxmax()
        max_diff = model_data.loc[max_diff_idx]
        print(f"\n  {model}:")
        print(f"    Max difference at sigma={max_diff['sigma']:.2f}")
        print(f"    OCD-Control difference: {max_diff['difference']:.4f}")
    
    print("\nMaximum theory divergence:")
    max_div_idx = np.unravel_index(np.argmax(divergence), divergence.shape)
    print(f"  alpha: {np.linspace(0.1, 0.9, 15)[max_div_idx[0]]:.3f}")
    print(f"  sigma: {np.linspace(0.5, 3.5, 15)[max_div_idx[1]]:.3f}")
    print(f"  Divergence: {divergence[max_div_idx]:.4f}")
    
    plt.show()
    
    return results_rl, comparison, divergence


if __name__ == "__main__":
    results = comprehensive_exploration()
    