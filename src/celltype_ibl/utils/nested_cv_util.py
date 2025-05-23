from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

class StratifiedKFoldHandler:
    def __init__(self, n_splits, n_repeats, random_state=None):
        """
        Initialize the StratifiedKFoldHandler with RepeatedStratifiedKFold parameters.
        
        Parameters:
        - n_splits: Number of splits for cross-validation.
        - n_repeats: Number of times to repeat the cross-validation process.
        - random_state: Seed for the random number generator (optional).
        """
        self.rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits
        
    def get_all_folds(self, X, y):
        """
        Get all train-test indices for each fold and repeat.
        
        Parameters:
        - X: Features of the dataset.
        - y: Target labels of the dataset.
        
        Returns:
        - all_folds: A list where each element is a tuple of (train_indices, test_indices) for a fold.
        """
        all_folds = []
        for train_idx, test_idx in self.rskf.split(X, y):
            all_folds.append((train_idx, test_idx))
        return all_folds

    def get_folds_for_repeat(self, X, y, repeat_id):
        """
        Get train-test indices for all folds in a specific repeat.
        
        Parameters:
        - X: Features of the dataset.
        - y: Target labels of the dataset.
        - repeat_id: The index of the repeat (starting from 0).
        
        Returns:
        - repeat_folds: A list of tuples where each tuple is (train_indices, test_indices) for a fold in the specified repeat.
        """
        repeat_folds = []
        # Total number of splits
        total_splits = self.n_splits
        
        # Generate folds
        for current_repeat_id, (train_idx, test_idx) in enumerate(self.rskf.split(X, y)):
            if current_repeat_id // total_splits == repeat_id:
                repeat_folds.append((train_idx, test_idx))
        
        if not repeat_folds:
            raise IndexError(f"Repeat ID {repeat_id} is out of range. Valid repeat IDs are from 0 to {self.rskf.n_repeats - 1}.")
        
        return repeat_folds
    