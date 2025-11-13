"""
Population Simulation Framework
Generates virtual participants with individual variability
"""

import numpy as np
import pandas as pd
from models import (
    RLModel_BroadGeneralization,
    RLModel_ImpairedSafety,
    BayesianModel_UncertaintyAverse,
    DEFAULT_PARAMS
)

class Population:
    """Represents a group of virtual participants"""
    
    def __init__(self, model_class, base_params, n_subjects=50, 
                 variability=0.1, group_name="Population"):
        """
        Create a population of virtual participants
        
        Args:
            model_class: class from models.py (e.g., RLModel_BroadGeneralization)
            base_params: dict of base parameters for this population
            n_subjects: number of virtual participants
            variability: how much parameters vary across individuals (std dev as proportion of mean)
            group_name: identifier (e.g., "OCD", "Control")
        """
        self.model_class = model_class
        self.base_params = base_params
        self.n_subjects = n_subjects
        self.variability = variability
        self.group_name = group_name
        self.subjects = []
        
        # Generate subjects
        self._generate_subjects()
        
    def _generate_subjects(self):
        """Create individual participants with parameter variability"""
        for i in range(self.n_subjects):
            # Add individual variability to parameters
            subject_params = self._add_variability(self.base_params)
            
            # Create model instance for this subject
            subject = {
                'id': f"{self.group_name}_{i:03d}",
                'model': self.model_class(subject_params),
                'params': subject_params
            }
            
            self.subjects.append(subject)
            
    def _add_variability(self, params):
        """
        Add individual differences to base parameters
        Uses truncated normal to keep parameters in valid ranges
        """
        varied_params = params.copy()
        
        for key, value in params.items():
            if key == 'n_stimuli':
                # Don't vary this
                continue
                
            # Add noise proportional to parameter value
            noise = np.random.normal(0, self.variability * abs(value))
            new_value = value + noise
            
            # Constrain to valid ranges
            if key in ['alpha', 'alpha_pos', 'alpha_neg', 'noise']:
                # Must be positive, typically < 1
                new_value = np.clip(new_value, 0.01, 0.99)
            elif key in ['sigma', 'uncertainty_weight']:
                # Must be positive
                new_value = np.clip(new_value, 0.1, 10.0)
            elif key in ['prior_variance']:
                new_value = np.clip(new_value, 0.01, 1.0)
                
            varied_params[key] = new_value
            
        return varied_params
        
    def reset_all(self):
        """Reset learning history for all subjects"""
        for subject in self.subjects:
            subject['model'].reset()
            
    def get_subject(self, idx):
        """Get a specific subject by index"""
        return self.subjects[idx]
        
    def get_all_subjects(self):
        """Return all subjects"""
        return self.subjects
        
    def summary_stats(self):
        """Get summary statistics of population parameters"""
        stats = {}
        
        # Collect all parameter values
        param_names = self.base_params.keys()
        
        for param in param_names:
            if param == 'n_stimuli':
                continue
                
            values = [s['params'][param] for s in self.subjects]
            stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return stats


def simulate_experiment(population, trial_sequence):
    """
    Run an experiment on all subjects in a population
    
    Args:
        population: Population object
        trial_sequence: list of (stimulus, outcome) tuples
        
    Returns:
        DataFrame with columns: subject_id, trial, stimulus, outcome, prediction, error
    """
    results = []
    
    # Reset all subjects
    population.reset_all()
    
    # Run each subject through the experiment
    for subject in population.get_all_subjects():
        model = subject['model']
        subject_id = subject['id']
        
        for trial_num, (stimulus, outcome) in enumerate(trial_sequence):
            # Get prediction before learning
            prediction = model.predict(stimulus)
            
            # Subject learns from outcome
            model.learn(stimulus, outcome)
            
            # Record data
            results.append({
                'subject_id': subject_id,
                'trial': trial_num,
                'stimulus': stimulus,
                'outcome': outcome,
                'prediction': prediction,
                'error': abs(prediction - outcome)
            })
            
    return pd.DataFrame(results)


def compare_populations(pop1, pop2, trial_sequence, metric='error'):
    """
    Compare two populations on the same experiment
    
    Args:
        pop1, pop2: Population objects
        trial_sequence: list of trials
        metric: what to compare ('error', 'prediction', etc.)
        
    Returns:
        dict with comparison statistics
    """
    # Run experiment on both populations
    data1 = simulate_experiment(pop1, trial_sequence)
    data2 = simulate_experiment(pop2, trial_sequence)
    
    # Calculate group-level statistics
    group1_metric = data1.groupby('subject_id')[metric].mean()
    group2_metric = data2.groupby('subject_id')[metric].mean()
    
    comparison = {
        'group1_name': pop1.group_name,
        'group2_name': pop2.group_name,
        'group1_mean': group1_metric.mean(),
        'group1_std': group1_metric.std(),
        'group2_mean': group2_metric.mean(),
        'group2_std': group2_metric.std(),
        'difference': group2_metric.mean() - group1_metric.mean(),
        'effect_size': (group2_metric.mean() - group1_metric.mean()) / 
                       np.sqrt((group1_metric.std()**2 + group2_metric.std()**2) / 2)
    }
    
    return comparison


def create_standard_populations(model_type='RL_broad', n_subjects=50, variability=0.15):
    """
    Convenience function to create OCD and Control populations
    
    Args:
        model_type: 'RL_broad', 'RL_impaired', or 'Bayesian'
        n_subjects: number of virtual participants per group
        variability: individual differences
        
    Returns:
        tuple of (control_population, ocd_population)
    """
    # Map model type to class
    model_map = {
        'RL_broad': RLModel_BroadGeneralization,
        'RL_impaired': RLModel_ImpairedSafety,
        'Bayesian': BayesianModel_UncertaintyAverse
    }
    
    model_class = model_map[model_type]
    
    # Get default parameters
    control_params = DEFAULT_PARAMS['control'][model_type]
    ocd_params = DEFAULT_PARAMS['ocd'][model_type]
    
    # Create populations
    control_pop = Population(
        model_class=model_class,
        base_params=control_params,
        n_subjects=n_subjects,
        variability=variability,
        group_name="Control"
    )
    
    ocd_pop = Population(
        model_class=model_class,
        base_params=ocd_params,
        n_subjects=n_subjects,
        variability=variability,
        group_name="OCD"
    )
    
    return control_pop, ocd_pop


def batch_simulate(populations, trial_sequence, n_replications=10):
    """
    Run multiple replications of an experiment
    
    Args:
        populations: list of Population objects
        trial_sequence: experiment trials
        n_replications: how many times to run
        
    Returns:
        DataFrame with all data
    """
    all_data = []
    
    for rep in range(n_replications):
        for pop in populations:
            data = simulate_experiment(pop, trial_sequence)
            data['replication'] = rep
            data['group'] = pop.group_name
            all_data.append(data)
            
    return pd.concat(all_data, ignore_index=True)


# Example usage helper
def quick_demo():
    """Quick demonstration of the simulation framework"""
    print("Creating populations...")
    control, ocd = create_standard_populations('RL_broad', n_subjects=30)
    
    print(f"\nControl population parameters:")
    for param, stats in control.summary_stats().items():
        print(f"  {param}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
    print(f"\nOCD population parameters:")
    for param, stats in ocd.summary_stats().items():
        print(f"  {param}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Simple trial sequence
    print("\nGenerating trial sequence...")
    trials = [(i % 10, 1.0 if i % 10 < 5 else 0.0) for i in range(50)]
    
    print("Running simulation...")
    comparison = compare_populations(control, ocd, trials, metric='error')
    
    print("\nResults:")
    print(f"  Control: {comparison['group1_mean']:.3f} ± {comparison['group1_std']:.3f}")
    print(f"  OCD: {comparison['group2_mean']:.3f} ± {comparison['group2_std']:.3f}")
    print(f"  Difference: {comparison['difference']:.3f}")
    print(f"  Effect size: {comparison['effect_size']:.3f}")
    
    return control, ocd, trials


if __name__ == "__main__":
    quick_demo()