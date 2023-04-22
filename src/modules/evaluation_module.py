import numpy as np


class Session():
    
    __slots__ = ['learner', 'initial_unlabeled_idx', 'all_queried_idx']

    def __init__(self, learner, unlabeled_idx) -> None:
        self.learner = learner
        self.initial_unlabeled_idx = unlabeled_idx
        self.all_queried_idx = np.empty()

    def append(self, queried_idx):
        self.all_queried_idx = np.concatenate([self.all_queried_idx, queried_idx])


class Evaluation():

    __slots__ = ['unlabeled_count', 'query_count', 'n_sessions']

    def __init__(self, dataset_size, sessions) -> None:
        self.unlabeled_count = np.zeros(dataset_size)
        self.query_count = np.zeros(dataset_size)
        self.n_sessions = len(sessions)

        # TODO

    def append(self, session):
        raise NotImplementedError
    

    def to_csv(self, path):
        raise NotImplementedError
    

def from_csv() -> Evaluation:
    raise NotImplementedError