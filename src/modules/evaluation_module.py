import numpy as np


class Session():
    
    __slots__ = ['dataset', 'learner', 'initial_unlabeled_idx', 'all_queried_idx']

    def __init__(self, dataset, learner) -> None:
        self.dataset = dataset
        self.learner = learner

        self.initial_unlabeled_idx = dataset.unlabeled_idx
        self.all_queried_idx = np.empty(0, dtype=np.int32)

    def update(self):
        self.all_queried_idx = np.concatenate([self.all_queried_idx, self.dataset.last_labeled_idx])


class Evaluation():

    __slots__ = ['_unlabeled_count', '_query_count']

    def __init__(self, dataset_size, sessions = []) -> None:
        self._unlabeled_count = np.zeros(dataset_size, dtype=np.int32)
        self._query_count = np.zeros(dataset_size, dtype=np.int32)
        self.append(sessions)

    @property
    def frequency(self):
        # doing the division this way ensured that if unlabeled_count[i] == 0
        # then frequency[i] == 0 instead of throwing an exception.
        return np.divide(
            self._query_count, 
            self._unlabeled_count, 
            out=np.zeros_like(self._query_count), 
            where=self._unlabeled_count != 0)

    def append(self, sessions : Session):
        if type(sessions) is Session:
            sessions = [sessions]
        for session in sessions:
            self._unlabeled_count[session.initial_unlabeled_idx] += 1
            self._query_count[session.all_queried_idx] += 1

    def merge(self, other, inplace=False):
        assert(type(other) is Evaluation)
        assert(len(self) == len(other))

        if not inplace:
            new_evaluation = Evaluation(len(self))
            new_evaluation.merge(self, inplace=True)
            new_evaluation.merge(other, inplace=True)
            return new_evaluation
        else:
            self._unlabeled_count += other._unlabeled_count
            self._query_count += other._query_count
        return None
            

    def to_csv(self, path):
        with open(path, 'w') as f:
            for i in range(len(self._unlabeled_count)):
                f.write(f"{self._unlabeled_count[i]},{self._query_count[i]}\n")

    
    def __len__(self):
        return len(self._unlabeled_count)
    

def from_csv() -> Evaluation:
    raise NotImplementedError