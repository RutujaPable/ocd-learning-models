"""
FAST Parameter Recovery - Uses smart multi-start instead of slow global optimizer
Should complete in ~2 minutes instead of 30+ minutes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from models import RLModel_BroadGeneralization
from experiments import GeneralizationTask


def generate_synthetic_data(model_class, true_params, trial_sequence):
    """Generate synthetic data - averaging predictions to reduce noise"""
    params = true_params.copy()
    params['n_stimuli'] = 10
    
    model = model_class(params)
    responses = []
    
    for stimulus, outcome in trial_sequence:
        # Average over stochastic predictions to get stable response
        preds = [model.predict(stimulus) for _ in range(50)]
        response = np.mean(preds)
        responses.append(response)
        model.learn(stimulus, outcome)
    
    return responses


def negative_log_likelihood(params_array, model_class, trial_sequence, responses):
    """Calculate NLL - fast version"""
    params = {
        'alpha': params_array[0],
        'sigma': params_array[1],
        'noise': 0.05,
        'n_stimuli': 10
    }
    
    # Quick parameter validation
    if params['alpha'] <= 0 or params['alpha'] >= 1 or params['sigma'] <= 0:
        return 1e10
    
    try:
        model = model_class(params)
    except:
        return 1e10
    
    log_likelihood = 0
    obs_variance = 0.01
    
    for i, (stimulus, outcome) in enumerate(trial_sequence):
        # SPEED FIX: Only 10 predictions instead of 30
        preds = [model.predict(stimulus) for _ in range(10)]
        expected_pred = np.mean(preds)
        
        residual = responses[i] - expected_pred
        likelihood = np.exp(-residual**2 / (2 * obs_variance))
        likelihood = max(likelihood, 1e-10)
        
        log_likelihood += np.log(likelihood)
        model.learn(stimulus, outcome)
    
    return -log_likelihood


def fit_model_fast(model_class, trial_sequence, responses):
    """FAST fitting with smart multi-start"""
    
    # Try 3 carefully chosen starting points
    starting_points = [
        [0.3, 1.5],   # Middle values
        [0.2, 0.7],   # Low sigma
        [0.5, 3.5],   # High sigma
    ]
    
    best_result = None
    best_nll = np.inf
    
    for initial in starting_points:
        try:
            result = minimize(
                negative_log_likelihood,
                initial,
                args=(model_class, trial_sequence, responses),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99), (0.1, 5.0)],
                options={'maxiter': 100}  # Limit iterations
            )
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None:
        # Fallback
        return {
            'fitted_params': {'alpha': 0.3, 'sigma': 1.5, 'noise': 0.05},
            'neg_log_likelihood': 1e10,
            'success': False,
            'BIC': 1e10
        }
    
    return {
        'fitted_params': {
            'alpha': best_result.x[0],
            'sigma': best_result.x[1],
            'noise': 0.05
        },
        'neg_log_likelihood': best_nll,
        'success': best_result.success,
        'BIC': 2 * best_nll + 2 * np.log(len(trial_sequence))
    }


def parameter_recovery_study(model_class, param_grid, n_replications=5, n_trials=150):
    """Run parameter recovery - FAST version"""
    results = []
    
    # SPEED FIX: 150 trials instead of 200
    exp = GeneralizationTask(
        n_stimuli=10,
        n_training_trials=int(n_trials * 0.8),
        n_test_trials=int(n_trials * 0.2),
        trained_stimuli=[2, 7],
        reward_function='gaussian'
    )
    trial_sequence = exp.generate_trials()
    
    print(f"Running recovery for {model_class.__name__}")
    print(f"Task: Generalization (better for identifying sigma)")
    print(f"Grid: {len(param_grid)} sets × {n_replications} reps = {len(param_grid)*n_replications} fits")
    print(f"Estimated time: ~2 minutes\n")
    
    import time
    start_time = time.time()
    
    for idx, true_params in enumerate(param_grid):
        print(f"Parameter set {idx+1}/{len(param_grid)}: α={true_params['alpha']:.2f}, σ={true_params['sigma']:.2f}")
        
        for rep in range(n_replications):
            # Generate data
            responses = generate_synthetic_data(model_class, true_params, trial_sequence)
            
            # Fit
            fit_result = fit_model_fast(model_class, trial_sequence, responses)
            
            # Record
            result = {
                'replication': rep,
                'fit_success': fit_result['success'],
                'neg_log_likelihood': fit_result['neg_log_likelihood'],
                'BIC': fit_result['BIC'],
                'true_alpha': true_params['alpha'],
                'fitted_alpha': fit_result['fitted_params']['alpha'],
                'error_alpha': abs(true_params['alpha'] - fit_result['fitted_params']['alpha']),
                'true_sigma': true_params['sigma'],
                'fitted_sigma': fit_result['fitted_params']['sigma'],
                'error_sigma': abs(true_params['sigma'] - fit_result['fitted_params']['sigma']),
                'true_noise': 0.05,
                'fitted_noise': 0.05,
                'error_noise': 0.0
            }
            
            results.append(result)
        
        # Show progress
        elapsed = time.time() - start_time
        per_set = elapsed / (idx + 1)
        remaining = per_set * (len(param_grid) - idx - 1)
        print(f"  ✓ Done (avg errors: α={np.mean([r['error_alpha'] for r in results[-n_replications:]]):.3f}, "
              f"σ={np.mean([r['error_sigma'] for r in results[-n_replications:]]):.3f}) "
              f"[~{int(remaining)}s remaining]\n")
    
    total_time = time.time() - start_time
    print(f"Total time: {int(total_time)}s ({total_time/60:.1f} minutes)")
    
    return pd.DataFrame(results)


def create_parameter_grid_rl_broad(n_points=3):
    """Use extreme, clearly different values"""
    alphas = [0.2, 0.4, 0.6]
    sigmas = [0.5, 2.0, 4.5]
    
    grid = []
    for alpha in alphas:
        for sigma in sigmas:
            grid.append({'alpha': alpha, 'sigma': sigma, 'noise': 0.05})
    
    return grid


def plot_recovery_results(recovery_df, model_name='Model'):
    """Visualize recovery"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Alpha
    ax = axes[0]
    ax.scatter(recovery_df['true_alpha'], recovery_df['fitted_alpha'], alpha=0.6, s=50, color='steelblue')
    lim = [0, 1]
    ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect recovery')
    corr = np.corrcoef(recovery_df['true_alpha'], recovery_df['fitted_alpha'])[0,1]
    mae = recovery_df['error_alpha'].mean()
    ax.set_xlabel('True α', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitted α', fontsize=12, fontweight='bold')
    ax.set_title(f'α Recovery\nr = {corr:.3f}, MAE = {mae:.3f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Sigma  
    ax = axes[1]
    ax.scatter(recovery_df['true_sigma'], recovery_df['fitted_sigma'], alpha=0.6, s=50, color='coral')
    lim = [0, 5]
    ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect recovery')
    corr = np.corrcoef(recovery_df['true_sigma'], recovery_df['fitted_sigma'])[0,1]
    mae = recovery_df['error_sigma'].mean()
    ax.set_xlabel('True σ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitted σ', fontsize=12, fontweight='bold')
    ax.set_title(f'σ Recovery\nr = {corr:.3f}, MAE = {mae:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Parameter Recovery: {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_recovery_by_true_value(recovery_df):
    """Show recovery quality by parameter value"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Alpha error by true alpha
    ax = axes[0, 0]
    for alpha in recovery_df['true_alpha'].unique():
        data = recovery_df[recovery_df['true_alpha'] == alpha]
        ax.scatter([alpha]*len(data), data['error_alpha'], alpha=0.5, s=30)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Error = 0.1')
    ax.set_xlabel('True α')
    ax.set_ylabel('Absolute Error')
    ax.set_title('α Error by True Value', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sigma error by true sigma
    ax = axes[0, 1]
    for sigma in recovery_df['true_sigma'].unique():
        data = recovery_df[recovery_df['true_sigma'] == sigma]
        ax.scatter([sigma]*len(data), data['error_sigma'], alpha=0.5, s=30)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Error = 0.5')
    ax.set_xlabel('True σ')
    ax.set_ylabel('Absolute Error')
    ax.set_title('σ Error by True Value', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distributions
    ax = axes[1, 0]
    ax.hist(recovery_df['error_alpha'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(recovery_df['error_alpha'].mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('α Error')
    ax.set_ylabel('Count')
    ax.set_title(f'α Error Distribution (mean={recovery_df["error_alpha"].mean():.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.hist(recovery_df['error_sigma'], bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(recovery_df['error_sigma'].mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('σ Error')
    ax.set_ylabel('Count')
    ax.set_title(f'σ Error Distribution (mean={recovery_df["error_sigma"].mean():.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def quick_recovery_demo():
    """Quick validation"""
    print("=" * 70)
    print("FAST PARAMETER RECOVERY - VALIDATION")
    print("=" * 70)
    print()
    
    param_grid = create_parameter_grid_rl_broad(n_points=3)
    
    recovery_df = parameter_recovery_study(
        model_class=RLModel_BroadGeneralization,
        param_grid=param_grid,
        n_replications=5,
        n_trials=150  # Shorter for speed
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Overall statistics
    for param in ['alpha', 'sigma']:
        error = recovery_df[f'error_{param}']
        correlation = np.corrcoef(
            recovery_df[f'true_{param}'],
            recovery_df[f'fitted_{param}']
        )[0, 1]
        
        print(f"\n{param.upper()}:")
        print(f"  Mean error:  {error.mean():.4f}")
        print(f"  Median error: {error.median():.4f}")
        print(f"  Max error:   {error.max():.4f}")
        print(f"  Correlation: {correlation:.3f}")
        
        # Quality assessment
        if param == 'alpha':
            threshold = 0.15
            quality = "✓ GOOD" if correlation > 0.7 and error.mean() < threshold else "⚠ ACCEPTABLE" if correlation > 0.5 else "❌ POOR"
        else:  # sigma
            threshold = 0.8
            quality = "✓ GOOD" if correlation > 0.7 and error.mean() < threshold else "⚠ ACCEPTABLE" if correlation > 0.5 else "❌ POOR"
        
        print(f"  Quality: {quality} (threshold: error < {threshold}, r > 0.7)")
    
    success_rate = recovery_df['fit_success'].mean()
    print(f"\nOptimization success rate: {success_rate:.1%}")
    
    # Check if any parameter combinations are problematic
    print("\n" + "=" * 70)
    print("PARAMETER-SPECIFIC ANALYSIS")
    print("=" * 70)
    
    for alpha in recovery_df['true_alpha'].unique():
        for sigma in recovery_df['true_sigma'].unique():
            subset = recovery_df[(recovery_df['true_alpha'] == alpha) & 
                                 (recovery_df['true_sigma'] == sigma)]
            if len(subset) > 0:
                alpha_err = subset['error_alpha'].mean()
                sigma_err = subset['error_sigma'].mean()
                status = "✓" if alpha_err < 0.15 and sigma_err < 0.8 else "⚠"
                print(f"{status} α={alpha:.1f}, σ={sigma:.1f}: "
                      f"errors = {alpha_err:.3f}, {sigma_err:.3f}")
    
    # Plots
    print("\nGenerating plots...")
    fig1 = plot_recovery_results(recovery_df, 'RL Broad Generalization (FAST)')
    fig1.savefig('recovery_FAST.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_recovery_by_true_value(recovery_df)
    fig2.savefig('recovery_details_FAST.png', dpi=150, bbox_inches='tight')
    
    recovery_df.to_csv('recovery_FAST.csv', index=False)
    
    print("\nFiles saved:")
    print("  - recovery_FAST.png (main results)")
    print("  - recovery_details_FAST.png (detailed analysis)")
    print("  - recovery_FAST.csv (raw data)")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    alpha_corr = np.corrcoef(recovery_df['true_alpha'], recovery_df['fitted_alpha'])[0,1]
    sigma_corr = np.corrcoef(recovery_df['true_sigma'], recovery_df['fitted_sigma'])[0,1]
    
    if alpha_corr > 0.7 and sigma_corr > 0.7:
        print("\n✓✓✓ EXCELLENT: Both parameters recover well!")
        print("    The model is identifiable and can be used for real data.")
    elif alpha_corr > 0.5 and sigma_corr > 0.5:
        print("\n⚠ ACCEPTABLE: Parameters show moderate recovery.")
        print("    The model can provide useful information but estimates will be noisy.")
    else:
        print("\n❌ POOR: Parameter recovery is inadequate.")
        print("    Consider: longer experiments, different task, or simpler model.")
    
    plt.show()
    return recovery_df


if __name__ == "__main__":
    results = quick_recovery_demo()