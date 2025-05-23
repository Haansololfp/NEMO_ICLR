import numpy as np


class CrossSessionLeaveOneOut:
    """
    K-Fold by held-out sessions.
    Provides train/test indeces stratified by session.
    Useful for Allen dataset
    """

    def __init__(self, session_labels: np.ndarray):
        self.session_labels = session_labels
        self.unique_sessions = np.unique(session_labels)
        self.n_splits = len(self.unique_sessions)

    def split(self, X=None, y=None):
        for test_session in self.unique_sessions:
            train_idx = np.where(self.session_labels != test_session)[0]
            test_idx = np.where(self.session_labels == test_session)[0]
            yield train_idx, test_idx
