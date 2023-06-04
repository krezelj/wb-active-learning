import numpy as np


class Session():
    """

    A class representing an Active Learning session.
    ...
    Attributes
    ----------
    dataset : data_module.ActiveDataset
        The dataset used in the session.
    learner : learner_module.ActiveLearner
        The learner model used in the session.
    initial_unlabeled_idx : numpy.ndarray
        The initial indices of unlabeled examples in the dataset.
    all_queried_idx : numpy.ndarray
        The indices of examples that have been queried for labeling.

    Methods
    -------
    update()
        Updates the `all_queried_idx` attribute
        by appending the indices of the last labeled examples.
    """
    
    __slots__ = ['dataset', 'learner', 'initial_unlabeled_idx', 'all_queried_idx']

    def __init__(self, dataset, learner) -> None:
        """
        Parameters
        ----------
        dataset : data_module.ActiveDataset
            The dataset used in the session.
        learner : learner_module.ActiveLearner
            The learner model used in the session.
        """

        self.dataset = dataset
        self.learner = learner

        self.initial_unlabeled_idx = dataset.unlabeled_idx
        self.all_queried_idx = np.empty(0, dtype=np.int32)

    def update(self) -> None:
        """
        Updates the `all_queried_idx` attribute
        by appending the indices of the last labeled examples.
        """
        self.all_queried_idx = np.concatenate([self.all_queried_idx, self.dataset.last_labeled_idx])


class Evaluation():
    """
        A class used to represent the evaluation of a process.
        ...
        Attributes
        ----------
        _unlabeled_count : numpy.ndarray
            An array storing the count of unlabeled examples for each index.
        _query_count : numpy.ndarray
            An array storing the count of queried examples for each index.

        Methods
        -------
        frequency()
            Returns the frequency of querying for each example in the evaluation.
        estimate_frequency()
            Returns an estimated frequency of querying for each example in the evaluation.
        top_queried(k, most_queried=True, use_estimate=True)
            Returns the indices of the top-k most or least queried examples based on the frequency.
        append(sessions)
            Appends sessions to the evaluation, updating the counts of unlabeled and queried examples.
        merge(other, inplace=False)
            Merges the evaluation with another evaluation object.
        to_csv(path)
            Writes the evaluation counts to a CSV file.
        """

    __slots__ = ['_unlabeled_count', '_query_count']

    def __init__(self, dataset_size, sessions = []) -> None:
        """
        Parameters
        ----------
        dataset_size : int
            The size of the dataset.
        sessions : list, optional
            A list of Session objects to be appended to the evaluation. (default = [])
        """

        self._unlabeled_count = np.zeros(dataset_size, dtype=np.int32)
        self._query_count = np.zeros(dataset_size, dtype=np.int32)
        self.append(sessions)

    @property
    def frequency(self):
        """
        Calculates the frequency of querying for each example in the evaluation.

        Returns
        -------
        numpy.ndarray
            An array representing the frequency of querying for each example.
        """
        # doing the division this way ensured that if unlabeled_count[i] == 0
        # then frequency[i] == 0 instead of throwing an exception.
        return np.divide(
            self._query_count, 
            self._unlabeled_count, 
            out=np.zeros_like(self._query_count).astype(np.float32), 
            where=self._unlabeled_count != 0)
    
    @property
    def estimate_frequency(self):
        """
        Estimates the frequency of querying for each example in the evaluation.

        Returns
        -------
        numpy.ndarray
            An array representing the estimated frequency of querying for each example.
        """

        # assume two more sessions are added, in one each sample was chosen and in the other
        # none of the samples were chosen, calculate frequency based on that
        return np.divide(
            self._query_count + 1, 
            self._unlabeled_count + 2)
    

    def top_queried(self, k, most_queried=True, use_estimate=True):
        """
         Returns the indices of the top-k most or least queried examples based on the frequency.

         Parameters
         ----------
         k : int
             The number of indices to return.
         most_queried : bool, optional
             Determines whether to return the most queried or least queried examples. (default = True)
         use_estimate : bool, optional
             Determines whether to use the estimated frequency or the actual frequency for ranking. (default = True)

         Returns
         -------
         numpy.ndarray
             An array of indices representing the top-k most or least queried examples.
         """

        # if most queried is False then the least queried are returned
        if use_estimate:
            fq_zip = zip(self.estimate_frequency, self._query_count, self._unlabeled_count, np.arange(len(self)))
        else:
            fq_zip = zip(self.frequency, self._query_count, self._unlabeled_count, np.arange(len(self)))
        
        if most_queried:
            fq_sorted = sorted(fq_zip, key=lambda x : (x[0], x[1]), reverse=True)
        else:
            fq_sorted = sorted(fq_zip, key=lambda x : (x[0], -x[2]))
        return np.array([idx for *_, idx in fq_sorted[:k]])

    def append(self, sessions : Session):
        """
        Appends sessions to the evaluation, updating the counts of unlabeled and queried examples.

        Parameters
        ----------
        sessions : Session or list of Sessions
            The Session objects to be appended to the evaluation.
        """

        if type(sessions) is Session:
            sessions = [sessions]
        for session in sessions:
            self._unlabeled_count[session.initial_unlabeled_idx] += 1
            self._query_count[session.all_queried_idx] += 1

    def merge(self, other, inplace=False):
        """
         Merges the evaluation with another evaluation object.

         Parameters
         ----------
         other : Evaluation
             The Evaluation object to merge with.
         inplace : bool, optional
             Determines whether to perform the merge in-place or return a new Evaluation object. (default = False)

         Returns
         -------
         Evaluation or None
             If `inplace` is False, a new Evaluation object containing the merged counts.
             If `inplace` is True, returns None.
         """

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
            

    def to_csv(self, path) -> None:
        """
        Writes the evaluation counts to a CSV file.

        Parameters
        ----------
        path : str
            The path of the CSV file to write.
        """

        with open(path, 'w') as f:
            for i in range(len(self._unlabeled_count)):
                f.write(f"{self._unlabeled_count[i]},{self._query_count[i]}\n")

    
    def __len__(self):
        return len(self._unlabeled_count)
    

def read_csv(path) -> Evaluation:
    """
    Reads an evaluation from a CSV file.

    Parameters
    ----------
    path : str
        The path of the CSV file to read.
    """

    with open(path, 'r') as f:
        lines = f.readlines()
        evaluation = Evaluation(len(lines))
        for i, line in enumerate(lines):
            u, q = line.split(',')
            evaluation._unlabeled_count[i] = int(u)
            evaluation._query_count[i] = int(q)

    return evaluation
