"""
Experimental Design and Trial Generation
Creates different types of learning experiments
"""

import numpy as np
import pandas as pd

class ExperimentDesign:
    """Base class for experiment designs"""
    
    def __init__(self, n_stimuli=10, n_trials=100):
        """
        Args:
            n_stimuli: number of distinct stimuli
            n_trials: total number of trials
        """
        self.n_stimuli = n_stimuli
        self.n_trials = n_trials
        
    def generate_trials(self):
        """Generate sequence of (stimulus, outcome) tuples"""
        raise NotImplementedError("Subclasses must implement generate_trials()")


class SimpleDiscriminationTask(ExperimentDesign):
    """
    Simple discrimination: some stimuli are rewarded, others not
    """
    
    def __init__(self, n_stimuli=10, n_trials=100, reward_threshold=5):
        """
        Args:
            reward_threshold: stimuli below this value are rewarded
        """
        super().__init__(n_stimuli, n_trials)
        self.reward_threshold = reward_threshold
        
    def generate_trials(self):
        """
        Generate random sequence where outcome depends on stimulus
        """
        trials = []
        
        for _ in range(self.n_trials):
            # Random stimulus
            stimulus = np.random.randint(0, self.n_stimuli)
            
            # Deterministic outcome based on stimulus
            if stimulus < self.reward_threshold:
                outcome = 1.0  # Reward
            else:
                outcome = 0.0  # No reward
                
            trials.append((stimulus, outcome))
            
        return trials


class GeneralizationTask(ExperimentDesign):
    """
    Generalization task: learn about some stimuli, test on others
    Tests how learning spreads across stimulus space
    """
    
    def __init__(self, n_stimuli=10, n_training_trials=80, n_test_trials=20,
                 trained_stimuli=[2, 7], reward_function='gaussian'):
        """
        Args:
            trained_stimuli: which stimuli receive feedback
            reward_function: how reward varies across stimulus space
        """
        super().__init__(n_stimuli, n_training_trials + n_test_trials)
        self.n_training_trials = n_training_trials
        self.n_test_trials = n_test_trials
        self.trained_stimuli = trained_stimuli
        self.reward_function = reward_function
        
    def _get_outcome(self, stimulus):
        """Determine outcome for a stimulus"""
        if self.reward_function == 'gaussian':
            # Reward peaks at certain stimuli
            outcome = 0
            for trained_stim in self.trained_stimuli:
                distance = abs(stimulus - trained_stim)
                outcome += np.exp(-distance**2 / 2)
            outcome = min(outcome, 1.0)
            
        elif self.reward_function == 'linear':
            # Linear gradient across stimulus space
            outcome = stimulus / self.n_stimuli
            
        else:  # 'step'
            # Binary: trained stimuli are good
            outcome = 1.0 if stimulus in self.trained_stimuli else 0.0
            
        return outcome
        
    def generate_trials(self):
        """
        Generate training trials (feedback) then test trials (no feedback)
        """
        trials = []
        
        # Training phase: only trained stimuli, with feedback
        for _ in range(self.n_training_trials):
            stimulus = np.random.choice(self.trained_stimuli)
            outcome = self._get_outcome(stimulus)
            trials.append((stimulus, outcome))
            
        # Test phase: all stimuli, including novel ones
        # (In real experiment, no feedback. Here we include outcomes for analysis)
        for _ in range(self.n_test_trials):
            stimulus = np.random.randint(0, self.n_stimuli)
            outcome = self._get_outcome(stimulus)
            trials.append((stimulus, outcome))
            
        return trials


class ReversalLearningTask(ExperimentDesign):
    """
    Reversal learning: reward contingencies change mid-experiment
    Tests flexibility
    """
    
    def __init__(self, n_stimuli=10, n_trials=100, reversal_trial=50):
        """
        Args:
            reversal_trial: when contingencies flip
        """
        super().__init__(n_stimuli, n_trials)
        self.reversal_trial = reversal_trial
        
    def generate_trials(self):
        """
        First half: stimuli 0-4 rewarded
        Second half: stimuli 5-9 rewarded
        """
        trials = []
        
        for trial_num in range(self.n_trials):
            stimulus = np.random.randint(0, self.n_stimuli)
            
            # Determine reward based on phase
            if trial_num < self.reversal_trial:
                # Phase 1: low stimuli rewarded
                outcome = 1.0 if stimulus < 5 else 0.0
            else:
                # Phase 2: high stimuli rewarded
                outcome = 1.0 if stimulus >= 5 else 0.0
                
            trials.append((stimulus, outcome))
            
        return trials


class ProbabilisticRewardTask(ExperimentDesign):
    """
    Probabilistic outcomes: each stimulus has a probability of reward
    Tests learning under uncertainty
    """
    
    def __init__(self, n_stimuli=10, n_trials=100, reward_probs=None):
        """
        Args:
            reward_probs: list/array of reward probability for each stimulus
        """
        super().__init__(n_stimuli, n_trials)
        
        if reward_probs is None:
            # Default: linear increase in reward probability
            self.reward_probs = np.linspace(0.1, 0.9, n_stimuli)
        else:
            self.reward_probs = np.array(reward_probs)
            
    def generate_trials(self):
        """
        Stochastic outcomes based on stimulus-specific probabilities
        """
        trials = []
        
        for _ in range(self.n_trials):
            stimulus = np.random.randint(0, self.n_stimuli)
            
            # Probabilistic outcome
            if np.random.rand() < self.reward_probs[stimulus]:
                outcome = 1.0
            else:
                outcome = 0.0
                
            trials.append((stimulus, outcome))
            
        return trials


class GradedOutcomesTask(ExperimentDesign):
    """
    Continuous outcomes: rewards are not binary
    More realistic for many domains
    """
    
    def __init__(self, n_stimuli=10, n_trials=100, outcome_function='sine'):
        """
        Args:
            outcome_function: how outcomes vary ('sine', 'linear', 'quadratic')
        """
        super().__init__(n_stimuli, n_trials)
        self.outcome_function = outcome_function
        
    def _get_outcome(self, stimulus):
        """Calculate graded outcome for stimulus"""
        x = stimulus / self.n_stimuli  # Normalize to [0,1]
        
        if self.outcome_function == 'sine':
            outcome = 0.5 + 0.5 * np.sin(2 * np.pi * x)
        elif self.outcome_function == 'linear':
            outcome = x
        elif self.outcome_function == 'quadratic':
            outcome = x ** 2
        else:  # 'random'
            outcome = np.random.rand()
            
        # Add noise
        outcome += np.random.normal(0, 0.1)
        return np.clip(outcome, 0, 1)
        
    def generate_trials(self):
        """Generate trials with graded outcomes"""
        trials = []
        
        for _ in range(self.n_trials):
            stimulus = np.random.randint(0, self.n_stimuli)
            outcome = self._get_outcome(stimulus)
            trials.append((stimulus, outcome))
            
        return trials


class BlockedDesign(ExperimentDesign):
    """
    Blocked presentation: stimuli presented in blocks rather than random
    Tests sequential effects
    """
    
    def __init__(self, n_stimuli=10, trials_per_block=10, n_blocks=10):
        """
        Args:
            trials_per_block: how many trials of same stimulus
            n_blocks: how many blocks total
        """
        super().__init__(n_stimuli, trials_per_block * n_blocks)
        self.trials_per_block = trials_per_block
        self.n_blocks = n_blocks
        
    def generate_trials(self):
        """
        Present stimuli in blocks
        """
        trials = []
        
        for block in range(self.n_blocks):
            # Pick a stimulus for this block
            stimulus = np.random.randint(0, self.n_stimuli)
            
            # Determine outcome for this stimulus
            outcome = 1.0 if stimulus < self.n_stimuli // 2 else 0.0
            
            # Repeat for block length
            for _ in range(self.trials_per_block):
                trials.append((stimulus, outcome))
                
        return trials


def create_standard_experiment(experiment_type='discrimination', **kwargs):
    """
    Convenience function to create common experiment types
    
    Args:
        experiment_type: 'discrimination', 'generalization', 'reversal', 
                        'probabilistic', 'graded', 'blocked'
        **kwargs: passed to experiment constructor
        
    Returns:
        list of (stimulus, outcome) tuples
    """
    experiment_map = {
        'discrimination': SimpleDiscriminationTask,
        'generalization': GeneralizationTask,
        'reversal': ReversalLearningTask,
        'probabilistic': ProbabilisticRewardTask,
        'graded': GradedOutcomesTask,
        'blocked': BlockedDesign
    }
    
    exp_class = experiment_map.get(experiment_type, SimpleDiscriminationTask)
    experiment = exp_class(**kwargs)
    
    return experiment.generate_trials()


def analyze_trial_sequence(trials):
    """
    Get basic statistics about a trial sequence
    
    Args:
        trials: list of (stimulus, outcome) tuples
        
    Returns:
        dict with statistics
    """
    stimuli = [s for s, _ in trials]
    outcomes = [o for _, o in trials]
    
    stats = {
        'n_trials': len(trials),
        'n_unique_stimuli': len(set(stimuli)),
        'mean_outcome': np.mean(outcomes),
        'outcome_variance': np.var(outcomes),
        'stimulus_balance': {s: stimuli.count(s) for s in set(stimuli)}
    }
    
    return stats


# Example usage
def demo_experiments():
    """Demonstrate different experiment types"""
    
    print("=" * 60)
    print("EXPERIMENT DESIGN DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Simple discrimination
    print("\n1. Simple Discrimination Task")
    trials = create_standard_experiment('discrimination', n_trials=50)
    stats = analyze_trial_sequence(trials)
    print(f"   Trials: {stats['n_trials']}")
    print(f"   Mean outcome: {stats['mean_outcome']:.2f}")
    print(f"   Example trials: {trials[:5]}")
    
    # 2. Generalization
    print("\n2. Generalization Task")
    trials = create_standard_experiment('generalization', 
                                       n_training_trials=60,
                                       n_test_trials=20,
                                       trained_stimuli=[3, 7])
    print(f"   Training + test trials: {len(trials)}")
    print(f"   First 3 trials: {trials[:3]}")
    print(f"   Last 3 trials (test): {trials[-3:]}")
    
    # 3. Reversal learning
    print("\n3. Reversal Learning Task")
    trials = create_standard_experiment('reversal', n_trials=80, reversal_trial=40)
    print(f"   Before reversal (trial 20): {trials[20]}")
    print(f"   After reversal (trial 50): {trials[50]}")
    
    # 4. Probabilistic
    print("\n4. Probabilistic Reward Task")
    trials = create_standard_experiment('probabilistic', n_trials=50)
    stats = analyze_trial_sequence(trials)
    print(f"   Mean outcome: {stats['mean_outcome']:.2f} (should be ~0.5)")
    print(f"   Outcome variance: {stats['outcome_variance']:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_experiments()