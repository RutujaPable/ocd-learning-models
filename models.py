"""
Learning and Generalization Models for OCD Research
Implements competing theories about how learning differs in OCD
"""

import numpy as np
from scipy.stats import norm

class LearningModel:
    """Base class for all learning models"""
    
    def __init__(self, params, name="Base Model"):
        """
        Initialize model with parameters
        
        Args:
            params: dict with model-specific parameters
            name: str, model identifier
        """
        self.params = params
        self.name = name
        self.history = []  # stores (stimulus, outcome, prediction) tuples
        
    def reset(self):
        """Clear learning history"""
        self.history = []
        
    def learn(self, stimulus, outcome):
        """Update model based on experience"""
        raise NotImplementedError("Subclasses must implement learn()")
        
    def predict(self, stimulus):
        """Generate prediction for a stimulus"""
        raise NotImplementedError("Subclasses must implement predict()")
        
    def get_history(self):
        """Return learning history"""
        return self.history


class RLModel_BroadGeneralization(LearningModel):
    """
    Reinforcement Learning with Gaussian Generalization
    Theory: OCD involves broader generalization (larger sigma)
    """
    
    def __init__(self, params):
        """
        Args:
            params: dict with keys:
                - alpha: learning rate (0-1)
                - sigma: generalization width (>0)
                - noise: response variability (>0)
                - n_stimuli: number of discrete stimulus representations
        """
        super().__init__(params, name="RL-Broad Generalization")
        
        # Value function: stores learned values for each stimulus
        n_stim = params.get('n_stimuli', 10)
        self.values = np.zeros(n_stim)
        
    def _generalization_kernel(self, stim1, stim2):
        """
        Gaussian similarity between stimuli
        Higher sigma = broader generalization
        """
        distance = abs(stim1 - stim2)
        return np.exp(-distance**2 / (2 * self.params['sigma']**2))
        
    def learn(self, stimulus, outcome):
        """
        Update values using prediction error
        Generalization spreads learning to similar stimuli
        """
        # Get prediction before learning
        prediction = self.predict(stimulus)
        
        # Prediction error
        delta = outcome - prediction
        
        # Update all stimulus values based on generalization
        for i in range(len(self.values)):
            similarity = self._generalization_kernel(stimulus, i)
            self.values[i] += self.params['alpha'] * delta * similarity
            
        # Store history
        self.history.append({
            'stimulus': stimulus,
            'outcome': outcome,
            'prediction': prediction,
            'error': delta
        })
        
    def predict(self, stimulus):
        """
        Predict outcome for stimulus
        Uses generalization from all learned values
        """
        # Weighted average of all values based on similarity
        total_value = 0
        total_weight = 0
        
        for i in range(len(self.values)):
            similarity = self._generalization_kernel(stimulus, i)
            total_value += self.values[i] * similarity
            total_weight += similarity
            
        if total_weight > 0:
            prediction = total_value / total_weight
        else:
            prediction = 0
            
        # Add noise
        noisy_prediction = prediction + np.random.normal(0, self.params['noise'])
        
        return np.clip(noisy_prediction, 0, 1)  # Keep in [0,1] range


class RLModel_ImpairedSafety(LearningModel):
    """
    Reinforcement Learning with Asymmetric Learning
    Theory: OCD shows impaired safety learning (slower for positive outcomes)
    """
    
    def __init__(self, params):
        """
        Args:
            params: dict with keys:
                - alpha_pos: learning rate for positive outcomes
                - alpha_neg: learning rate for negative outcomes
                - sigma: generalization width
                - noise: response variability
                - n_stimuli: number of stimuli
        """
        super().__init__(params, name="RL-Impaired Safety")
        
        n_stim = params.get('n_stimuli', 10)
        self.values = np.zeros(n_stim)
        
    def _generalization_kernel(self, stim1, stim2):
        """Gaussian similarity"""
        distance = abs(stim1 - stim2)
        return np.exp(-distance**2 / (2 * self.params['sigma']**2))
        
    def learn(self, stimulus, outcome):
        """
        Update with outcome-dependent learning rate
        """
        prediction = self.predict(stimulus)
        delta = outcome - prediction
        
        # Use different learning rates based on outcome
        if outcome > prediction:  # Better than expected (safety signal)
            alpha = self.params['alpha_pos']
        else:  # Worse than expected (threat signal)
            alpha = self.params['alpha_neg']
            
        # Update with generalization
        for i in range(len(self.values)):
            similarity = self._generalization_kernel(stimulus, i)
            self.values[i] += alpha * delta * similarity
            
        self.history.append({
            'stimulus': stimulus,
            'outcome': outcome,
            'prediction': prediction,
            'error': delta,
            'alpha_used': alpha
        })
        
    def predict(self, stimulus):
        """Predict with generalization"""
        total_value = 0
        total_weight = 0
        
        for i in range(len(self.values)):
            similarity = self._generalization_kernel(stimulus, i)
            total_value += self.values[i] * similarity
            total_weight += similarity
            
        prediction = total_value / total_weight if total_weight > 0 else 0
        noisy_prediction = prediction + np.random.normal(0, self.params['noise'])
        
        return np.clip(noisy_prediction, 0, 1)


class BayesianModel_UncertaintyAverse(LearningModel):
    """
    Bayesian Learning with Uncertainty Weighting
    Theory: OCD involves heightened uncertainty aversion
    """
    
    def __init__(self, params):
        """
        Args:
            params: dict with keys:
                - uncertainty_weight: how much uncertainty affects decisions (>1 = more averse)
                - prior_mean: initial belief about outcomes
                - prior_variance: initial uncertainty
                - sigma: generalization width
                - noise: response variability
                - n_stimuli: number of stimuli
        """
        super().__init__(params, name="Bayesian-Uncertainty Averse")
        
        n_stim = params.get('n_stimuli', 10)
        # Store mean and variance for each stimulus
        self.means = np.ones(n_stim) * params.get('prior_mean', 0.5)
        self.variances = np.ones(n_stim) * params.get('prior_variance', 0.25)
        self.n_observations = np.zeros(n_stim)
        
    def _generalization_kernel(self, stim1, stim2):
        """Gaussian similarity"""
        distance = abs(stim1 - stim2)
        return np.exp(-distance**2 / (2 * self.params['sigma']**2))
        
    def learn(self, stimulus, outcome):
        """
        Bayesian update of beliefs
        """
        prediction = self.predict(stimulus)
        
        # Update beliefs for this stimulus
        old_var = self.variances[stimulus]
        old_mean = self.means[stimulus]
        
        # Bayesian update (assuming known observation variance)
        obs_var = 0.1  # Fixed observation noise
        new_var = 1 / (1/old_var + 1/obs_var)
        new_mean = new_var * (old_mean/old_var + outcome/obs_var)
        
        # Generalize learning to similar stimuli
        for i in range(len(self.means)):
            similarity = self._generalization_kernel(stimulus, i)
            
            # Update proportional to similarity
            self.means[i] += similarity * (new_mean - old_mean) * 0.5
            self.variances[i] = self.variances[i] * (1 - similarity * 0.3) + new_var * similarity * 0.3
            self.n_observations[i] += similarity
            
        self.history.append({
            'stimulus': stimulus,
            'outcome': outcome,
            'prediction': prediction,
            'uncertainty': np.sqrt(self.variances[stimulus])
        })
        
    def predict(self, stimulus):
        """
        Predict with uncertainty aversion
        High uncertainty -> more conservative (pessimistic) predictions
        """
        # Get belief about this stimulus
        total_mean = 0
        total_weight = 0
        total_uncertainty = 0
        
        for i in range(len(self.means)):
            similarity = self._generalization_kernel(stimulus, i)
            total_mean += self.means[i] * similarity
            total_uncertainty += np.sqrt(self.variances[i]) * similarity
            total_weight += similarity
            
        mean = total_mean / total_weight if total_weight > 0 else 0.5
        uncertainty = total_uncertainty / total_weight if total_weight > 0 else 1.0
        
        # Uncertainty aversion: reduce prediction by uncertainty-weighted penalty
        uncertainty_penalty = self.params['uncertainty_weight'] * uncertainty
        adjusted_prediction = mean - uncertainty_penalty
        
        # Add noise
        noisy_prediction = adjusted_prediction + np.random.normal(0, self.params['noise'])
        
        return np.clip(noisy_prediction, 0, 1)


# Default parameter sets for different populations
DEFAULT_PARAMS = {
    'control': {
        'RL_broad': {
            'alpha': 0.3,
            'sigma': 1.0,
            'noise': 0.1,
            'n_stimuli': 10
        },
        'RL_impaired': {
            'alpha_pos': 0.3,
            'alpha_neg': 0.3,
            'sigma': 1.0,
            'noise': 0.1,
            'n_stimuli': 10
        },
        'Bayesian': {
            'uncertainty_weight': 0.1,
            'prior_mean': 0.5,
            'prior_variance': 0.25,
            'sigma': 1.0,
            'noise': 0.1,
            'n_stimuli': 10
        }
    },
    'ocd': {
        'RL_broad': {
            'alpha': 0.3,
            'sigma': 2.5,  # Broader generalization
            'noise': 0.1,
            'n_stimuli': 10
        },
        'RL_impaired': {
            'alpha_pos': 0.15,  # Impaired safety learning
            'alpha_neg': 0.35,  # Enhanced threat learning
            'sigma': 1.0,
            'noise': 0.1,
            'n_stimuli': 10
        },
        'Bayesian': {
            'uncertainty_weight': 0.4,  # More uncertainty averse
            'prior_mean': 0.5,
            'prior_variance': 0.25,
            'sigma': 1.0,
            'noise': 0.1,
            'n_stimuli': 10
        }
    }
}