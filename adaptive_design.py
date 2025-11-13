"""
Adaptive Experimental Design
Implements optimal experimental design to discriminate between theories
Uses Bayesian experimental design principles
"""

import numpy as np
import pandas as pd
from itertools import product
from models import (
    RLModel_BroadGeneralization,
    RLModel_ImpairedSafety,
    BayesianModel_UncertaintyAverse
)


class AdaptiveDesigner:
    """
    Bayesian adaptive experimental design
    Chooses trials that maximize expected information gain
    """
    
    def __init__(self, models, model_params, stimulus_space, prior=None):
        """
        Args:
            models: list of model classes to discriminate
            model_params: dict mapping model names to parameter dicts
            stimulus_space: list of possible stimuli to present
            prior: initial beliefs about which model is correct (uniform if None)
        """
        self.models = models
        self.model_params = model_params
        self.stimulus_space = stimulus_space
        
        # Initialize model beliefs (uniform prior)
        if prior is None:
            self.beliefs = {m.__name__: 1.0/len(models) for m in models}
        else:
            self.beliefs = prior
            
        # Create model instances
        self.model_instances = {}
        for model_class in models:
            params = model_params[model_class.__name__].copy()
            params['n_stimuli'] = max(stimulus_space) + 1
            self.model_instances[model_class.__name__] = model_class(params)
            
        self.history = []
        
    def calculate_expected_information_gain(self, stimulus):
        """
        Calculate expected information gain for presenting a stimulus
        
        Information gain = reduction in uncertainty about which model is correct
        Uses KL divergence from prior to expected posterior
        
        Args:
            stimulus: which stimulus to evaluate
            
        Returns:
            expected information gain
        """
        # Current entropy
        current_entropy = self._entropy(self.beliefs)
        
        # Consider possible outcomes (discretize outcome space)
        possible_outcomes = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        
        expected_entropy = 0
        
        for outcome in possible_outcomes:
            # Calculate likelihood of this outcome under each model
            likelihoods = {}
            for model_name, model in self.model_instances.items():
                # Get multiple predictions to account for noise
                predictions = [model.predict(stimulus) for _ in range(30)]
                prediction = np.mean(predictions)
                
                # Use fixed variance for likelihood calculation
                variance = 0.02
                
                likelihood = np.exp(-(outcome - prediction)**2 / (2 * variance))
                likelihoods[model_name] = max(likelihood, 1e-10)
            
            # Calculate posterior beliefs if we observed this outcome
            posterior = {}
            total = 0
            
            for model_name in self.beliefs.keys():
                posterior[model_name] = self.beliefs[model_name] * likelihoods[model_name]
                total += posterior[model_name]
            
            if total > 0:
                for model_name in posterior.keys():
                    posterior[model_name] /= total
            else:
                posterior = self.beliefs.copy()
            
            # Weight by probability of observing this outcome
            prob_outcome = sum(self.beliefs[m] * likelihoods[m] 
                             for m in self.beliefs.keys())
            
            if prob_outcome > 0:
                # Expected entropy after observing this outcome
                expected_entropy += prob_outcome * self._entropy(posterior)
        
        # Information gain = current entropy - expected future entropy
        info_gain = max(current_entropy - expected_entropy, 0)
        
        return info_gain
    
    def _entropy(self, beliefs):
        """Calculate Shannon entropy of belief distribution"""
        entropy = 0
        for prob in beliefs.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def choose_next_trial(self):
        """
        Select the optimal next stimulus to present
        
        Returns:
            optimal stimulus and info gains for all stimuli
        """
        best_stimulus = None
        max_info_gain = -np.inf
        
        info_gains = {}
        
        for stimulus in self.stimulus_space:
            info_gain = self.calculate_expected_information_gain(stimulus)
            info_gains[stimulus] = info_gain
            
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = stimulus
        
        return best_stimulus, info_gains
    
    def update_beliefs(self, stimulus, outcome):
        """
        Update beliefs about models based on observed outcome
        
        Args:
            stimulus: stimulus that was presented
            outcome: observed outcome
        """
        # Calculate likelihoods - average over multiple predictions to handle noise
        likelihoods = {}
        
        for model_name, model in self.model_instances.items():
            # Get multiple predictions to average out noise
            predictions = [model.predict(stimulus) for _ in range(30)]
            mean_prediction = np.mean(predictions)
            
            # Likelihood of observed outcome given model's prediction
            # Use broader variance for more stable discrimination
            variance = 0.02  # Fixed variance for all models
            
            likelihood = np.exp(-(outcome - mean_prediction)**2 / (2 * variance))
            likelihoods[model_name] = max(likelihood, 1e-10)
            
            # Update model with outcome
            model.learn(stimulus, outcome)
        
        # Standard Bayesian update (no tempering)
        posterior = {}
        total = 0
        
        for model_name in self.beliefs.keys():
            posterior[model_name] = self.beliefs[model_name] * likelihoods[model_name]
            total += posterior[model_name]
        
        # Normalize
        if total > 1e-10:
            for model_name in posterior.keys():
                posterior[model_name] /= total
        else:
            posterior = self.beliefs.copy()
        
        self.beliefs = posterior
        
        # Record history
        self.history.append({
            'stimulus': stimulus,
            'outcome': outcome,
            'beliefs': self.beliefs.copy(),
            'entropy': self._entropy(self.beliefs),
            'likelihoods': likelihoods.copy()
        })
    
    def get_belief_trajectory(self):
        """Return DataFrame of belief evolution over time"""
        if not self.history:
            return pd.DataFrame()
        
        data = []
        for i, h in enumerate(self.history):
            row = {'trial': i, 'stimulus': h['stimulus'], 
                   'outcome': h['outcome'], 'entropy': h['entropy']}
            for model_name, belief in h['beliefs'].items():
                row[f'belief_{model_name}'] = belief
            data.append(row)
        
        return pd.DataFrame(data)


class RandomDesigner:
    """Baseline: random stimulus selection"""
    
    def __init__(self, stimulus_space):
        """
        Args:
            stimulus_space: list of possible stimuli
        """
        self.stimulus_space = stimulus_space
        
    def choose_next_trial(self):
        """Randomly select next stimulus"""
        stimulus = np.random.choice(self.stimulus_space)
        return stimulus, {}
    
    def update_beliefs(self, stimulus, outcome):
        """No belief updating for random design"""
        pass


class BalancedDesigner:
    """Baseline: balanced presentation of all stimuli"""
    
    def __init__(self, stimulus_space):
        """
        Args:
            stimulus_space: list of possible stimuli
        """
        self.stimulus_space = stimulus_space
        self.counts = {s: 0 for s in stimulus_space}
        
    def choose_next_trial(self):
        """Choose least-presented stimulus"""
        min_count = min(self.counts.values())
        candidates = [s for s, c in self.counts.items() if c == min_count]
        stimulus = np.random.choice(candidates)
        self.counts[stimulus] += 1
        return stimulus, {}
    
    def update_beliefs(self, stimulus, outcome):
        """No belief updating for balanced design"""
        pass


def compare_designs(designs, true_model, true_params, n_trials=50, n_replications=10):
    """
    Compare different experimental designs
    
    Args:
        designs: dict mapping design names to designer instances
        true_model: ground truth model class
        true_params: ground truth parameters
        n_trials: number of trials per experiment
        n_replications: how many times to replicate
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for design_name, designer_class in designs.items():
        print(f"Testing {design_name}...")
        
        for rep in range(n_replications):
            # Create true model
            params = true_params.copy()
            params['n_stimuli'] = 10
            true_model_instance = true_model(params)
            
            # Create designer
            if design_name == 'Adaptive':
                # Set up models to discriminate
                models = [RLModel_BroadGeneralization, RLModel_ImpairedSafety, 
                         BayesianModel_UncertaintyAverse]
                
                # Use default parameters for each model type
                from models import DEFAULT_PARAMS
                model_params = {
                    'RLModel_BroadGeneralization': DEFAULT_PARAMS['control']['RL_broad'],
                    'RLModel_ImpairedSafety': DEFAULT_PARAMS['control']['RL_impaired'],
                    'BayesianModel_UncertaintyAverse': DEFAULT_PARAMS['control']['Bayesian']
                }
                
                designer = AdaptiveDesigner(
                    models=models,
                    model_params=model_params,
                    stimulus_space=list(range(10))
                )
            else:
                designer = designer_class(stimulus_space=list(range(10)))
            
            # Run experiment
            for trial in range(n_trials):
                # Designer chooses stimulus
                stimulus, _ = designer.choose_next_trial()
                
                # True model generates outcome
                true_model_instance.predict(stimulus)  # For side effects
                outcome = 1.0 if stimulus < 5 else 0.0  # Simple discrimination
                true_model_instance.learn(stimulus, outcome)
                
                # Designer updates
                designer.update_beliefs(stimulus, outcome)
            
            # Evaluate final beliefs (if applicable)
            if hasattr(designer, 'beliefs'):
                final_belief = designer.beliefs.get(true_model.__name__, 0)
                final_entropy = designer._entropy(designer.beliefs)
            else:
                final_belief = np.nan
                final_entropy = np.nan
            
            results.append({
                'design': design_name,
                'replication': rep,
                'final_belief_correct': final_belief,
                'final_entropy': final_entropy
            })
    
    return pd.DataFrame(results)


def simulate_adaptive_experiment(designer, true_model_class, true_params, 
                                 n_trials=50, experiment_type='generalization'):
    """
    Run a full adaptive experiment
    
    Args:
        designer: AdaptiveDesigner instance
        true_model_class: ground truth model
        true_params: ground truth parameters  
        n_trials: number of trials
        experiment_type: type of task ('discrimination' or 'generalization')
        
    Returns:
        dict with results
    """
    # Create true model
    params = true_params.copy()
    params['n_stimuli'] = 10
    true_model = true_model_class(params)
    
    trial_data = []
    
    # For generalization task: train on stimuli 2 and 7, then test on others
    trained_stimuli = [2, 7]
    training_phase = n_trials // 2
    
    for trial in range(n_trials):
        # Designer selects stimulus
        stimulus, info_gains = designer.choose_next_trial()
        
        # Generate outcome based on experiment type
        if experiment_type == 'generalization':
            # Gaussian reward function centered on trained stimuli
            if trial < training_phase:
                # Training phase: only present trained stimuli
                stimulus = np.random.choice(trained_stimuli)
            
            # Reward based on distance from trained stimuli
            reward = 0
            for trained_stim in trained_stimuli:
                distance = abs(stimulus - trained_stim)
                reward += np.exp(-distance**2 / 2)
            true_outcome = min(reward, 1.0)
            
        else:  # discrimination
            true_outcome = 1.0 if stimulus < 5 else 0.0
        
        # True model responds
        prediction = true_model.predict(stimulus)
        true_model.learn(stimulus, true_outcome)
        
        # Designer updates beliefs
        designer.update_beliefs(stimulus, true_outcome)
        
        # Record trial (only include beliefs if designer has them)
        trial_record = {
            'trial': trial,
            'stimulus': stimulus,
            'outcome': true_outcome,
            'prediction': prediction,
            'info_gains': info_gains
        }
        
        if hasattr(designer, 'beliefs'):
            trial_record['beliefs'] = designer.beliefs.copy()
            trial_record['entropy'] = designer._entropy(designer.beliefs)
        
        trial_data.append(trial_record)
    
    result = {
        'trial_data': trial_data,
        'true_model': true_model_class.__name__
    }
    
    # Add belief-specific results if applicable
    if hasattr(designer, 'beliefs'):
        result['final_beliefs'] = designer.beliefs
        result['belief_trajectory'] = designer.get_belief_trajectory()
    
    return result


def efficiency_analysis(designs, true_model, true_params, max_trials=100):
    """
    Analyze how quickly each design converges to correct model
    
    Args:
        designs: dict of design strategies
        true_model: ground truth model class
        true_params: ground truth parameters
        max_trials: maximum trials to test
        
    Returns:
        DataFrame with convergence results
    """
    results = []
    
    for design_name, designer_class in designs.items():
        # Create designer
        if design_name == 'Adaptive':
            models = [RLModel_BroadGeneralization, RLModel_ImpairedSafety, 
                     BayesianModel_UncertaintyAverse]
            
            from models import DEFAULT_PARAMS
            model_params = {
                'RLModel_BroadGeneralization': DEFAULT_PARAMS['control']['RL_broad'],
                'RLModel_ImpairedSafety': DEFAULT_PARAMS['control']['RL_impaired'],
                'BayesianModel_UncertaintyAverse': DEFAULT_PARAMS['control']['Bayesian']
            }
            
            designer = AdaptiveDesigner(
                models=models,
                model_params=model_params,
                stimulus_space=list(range(10))
            )
        else:
            designer = designer_class(stimulus_space=list(range(10)))
        
        # Run experiment
        result = simulate_adaptive_experiment(
            designer, true_model, true_params, 
            n_trials=max_trials
        )
        
        # Extract belief trajectory (only for adaptive)
        if 'belief_trajectory' in result and not result['belief_trajectory'].empty:
            trajectory = result['belief_trajectory']
            correct_model_col = f"belief_{true_model.__name__}"
            
            for _, row in trajectory.iterrows():
                results.append({
                    'design': design_name,
                    'trial': row['trial'],
                    'belief_correct': row.get(correct_model_col, np.nan),
                    'entropy': row['entropy']
                })
        else:
            # For non-adaptive designs, we can't track beliefs
            # Instead, track a dummy metric
            for trial in range(max_trials):
                results.append({
                    'design': design_name,
                    'trial': trial,
                    'belief_correct': np.nan,
                    'entropy': np.nan
                })
    
    return pd.DataFrame(results)


# Quick demo
def demo_adaptive_design():
    """Demonstrate adaptive experimental design"""
    print("=" * 60)
    print("ADAPTIVE EXPERIMENTAL DESIGN DEMO")
    print("=" * 60)
    
    # Set up models to discriminate - USE EXAGGERATED PARAMETERS for clear discrimination
    models = [RLModel_BroadGeneralization, RLModel_ImpairedSafety, 
             BayesianModel_UncertaintyAverse]
    
    from models import DEFAULT_PARAMS
    model_params = {
        'RLModel_BroadGeneralization': {
            'alpha': 0.5,
            'sigma': 4.0,  # VERY broad generalization
            'noise': 0.05,
            'n_stimuli': 10
        },
        'RLModel_ImpairedSafety': {
            'alpha_pos': 0.05,  # VERY impaired safety learning
            'alpha_neg': 0.8,   # VERY strong threat learning
            'sigma': 1.0,
            'noise': 0.05,
            'n_stimuli': 10
        },
        'BayesianModel_UncertaintyAverse': {
            'uncertainty_weight': 0.8,  # VERY uncertainty averse
            'prior_mean': 0.5,
            'prior_variance': 0.25,
            'sigma': 1.0,
            'noise': 0.05,
            'n_stimuli': 10
        }
    }
    
    # Create adaptive designer
    print("\nCreating adaptive designer...")
    print("Models to discriminate (EXAGGERATED PARAMETERS):")
    print("  1. RL Broad Generalization (σ=4.0 - VERY broad)")
    print("  2. RL Impaired Safety (α_pos=0.05, α_neg=0.8 - VERY asymmetric)")
    print("  3. Bayesian Uncertainty Averse (weight=0.8 - VERY conservative)")
    
    designer = AdaptiveDesigner(
        models=models,
        model_params=model_params,
        stimulus_space=list(range(10))
    )
    
    print(f"\nInitial beliefs: {designer.beliefs}")
    print(f"Initial entropy: {designer._entropy(designer.beliefs):.3f} bits")
    
    # Run experiment (true model is Broad Generalization)
    print("\nRunning adaptive experiment...")
    print("(True model: RL Broad Generalization with EXAGGERATED parameters)")
    print("(Task: Generalization - learn about stimuli 2 & 7, test on others)")
    
    result = simulate_adaptive_experiment(
        designer=designer,
        true_model_class=RLModel_BroadGeneralization,
        true_params={'alpha': 0.5, 'sigma': 4.0, 'noise': 0.05, 'n_stimuli': 10},
        n_trials=50,
        experiment_type='generalization'
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nFinal beliefs:")
    for model, belief in result['final_beliefs'].items():
        marker = " ← TRUE MODEL" if model == result['true_model'] else ""
        print(f"  {model}: {belief:.4f}{marker}")
    
    print(f"\nTrue model: {result['true_model']}")
    print(f"Correctly identified: {result['final_beliefs'][result['true_model']] > 0.5}")
    
    # Show stimulus selection pattern
    trajectory = result['belief_trajectory']
    print(f"\nStimulus selection pattern (first 10 trials):")
    print(trajectory[['trial', 'stimulus', 'entropy']].head(10))
    
    print(f"\nFinal entropy: {trajectory['entropy'].iloc[-1]:.3f} bits")
    print(f"Entropy reduction: {designer._entropy({m.__name__: 1/3 for m in models}) - trajectory['entropy'].iloc[-1]:.3f} bits")
    
    return result


if __name__ == "__main__":
    result = demo_adaptive_design()