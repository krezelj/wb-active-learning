import os
from typing import Callable, Optional, Any, Tuple, Union, Literal

import torch
from torchvision.datasets import MNIST, PCAM, FashionMNIST
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Subset
import numpy as np

import src.modules.pcamx as pcamx
from src.modules.pcamx import PCAMLL

# Note: this variable should not be updated directly.
# Instead, use the update_data_dir() function to ensure that the stored path
# is always absolute.
_data_dir = ''
# (default value for __data_dir is assigned below update_data_dir() function declaration)


def update_data_dir(path, silent=False) -> None:
    """
    Updates the path to a directory where datasets are stored and downloaded to

    Parameters
    ----------
    path: str|os.PathLike
        Path to a new data directory
    silent: bool 
        Whether to notify about the new path. (default = False)
    """
    # expanduser: to process paths that begin with ~ on unix
    # expandvars: to handle environmental vars in a path such as $HOME
    path_absolute = os.path.abspath(
        os.path.expanduser(
            os.path.expandvars(path)
        )
    )
    global _data_dir
    _data_dir = path_absolute
    if not silent:
        print(f'Data directory set to: {_data_dir}')


# initialise a default data directory:
update_data_dir('./data', silent=True)
print(f'Default data directory set to {_data_dir}')
print('To change this path, use the update_data_dir() function '
      'from the data_module')


class IndexedSubset(Subset):
    """
    Class that extends the functionality of the Subset class.
    ...
    Attributes
    ----------
    dataset: Dataset
        The original dataset.
    indices: sequence 
        Indices to extract from the original dataset.
    """

    def __init__(self, dataset, indices) -> None:
        """
        Parameters
        ----------
        dataset: Dataset 
            The original dataset.
        indices: sequence 
            Indices to extract from the original dataset.
        """

        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        """
        Attributes
        ----------
        idx: int
            Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the data, target, and index.

        """

        data, target = super().__getitem__(idx)
        return data, target, idx



class ActiveDataset():
    """
    Class representing an active learning dataset.
    ...
    Parameters
    ----------
    _full_train_set: Datase
        The full training dataset.
    _full_test_set: Satase
        The full test dataset.
    labeled_idx: numpy.ndarray
        Indices of labeled samples in the training subset.
    unlabeled_idx: numpy.ndarray
        Indices of unlabeled samples in the training subset.
    last_labeled_idx: numpy.ndarray
        Indices of the last labeled samples.
    train_subset_idx: numpy.ndarray
        Indices of the training subset.
    test_subset_idx: numpy.ndarray
        Indices of the test subset.
    _cached_test_set: IndexedSubset
        Cached IndexedSubset object for the test subset.
    _cached_labeled_set: IndexedSubset
        Cached IndexedSubset object for the labeled samples.
    _cached_unlabeled_set: IndexedSubset
        Cached IndexedSubset object for the unlabeled samples.
    _cached_last_labeled_set: IndexedSubset
        Cached IndexedSubset object for the last labeled samples.

    Methods
    -------
    labeled_set()
        Property returning the labeled samples subset.
    labeled_targets()
        Property returning the targets of the labeled samples.
    unlabeled_set()
        Property returning the unlabeled samples subset.
    unlabeled_targets()
        Property returning the targets of the unlabeled samples.
    last_labeled_set()
        Property returning the subset of the last labeled samples.
    last_labeled_targets()
        Property returning the targets of the last labeled samples.
    test_set()
        Property returning the test subset.
    test_targets()
        Property returning the targets of the test samples.
    __get_balanced_train_subset(train_subset_size, ratio_classes)
        Generates a balanced train subset based on the specified ratio of classes.
    __get_from_source(source)
        Retrieves the datasets from the specified source.
    get_label_by_idx(indices, move_sample=True)
        Gets the label of an unlabeled sample.
    get_bootstrap_set(size=None, weights=None)
        Generates a bootstrap set from the labeled samples.
    """

    # TODO Add ability to manually set test set so that it's consistent across several tests
    
    __slots__ = ['_full_train_set', 
                 '_full_test_set', 
                 'labeled_idx', 
                 'unlabeled_idx', 
                 'last_labeled_idx', 
                 'train_subset_idx', 
                 'test_subset_idx',
                 '_cached_test_set', 
                 '_cached_labeled_set', 
                 '_cached_unlabeled_set', 
                 '_cached_last_labeled_set']

    @property
    def labeled_set(self):
        """
        Property returning the labeled samples subset.

        Returns
        -------
            IndexedSubset: Subset containing the labeled samples.
        """

        if self._cached_labeled_set is None:
            self._cached_labeled_set = IndexedSubset(self._full_train_set, self.labeled_idx)
        return self._cached_labeled_set
    
    @property 
    def labeled_targets(self):
        """
        Property returning the targets of the labeled samples.

        Returns
        -------
            torch.Tensor: Tensor containing the targets of the labeled samples.
        """

        return self._full_train_set.targets[self.labeled_idx]
    
    @property
    def unlabeled_set(self):
        """
        Property returning the unlabeled samples subset.

        Returns
        -------
            IndexedSubset: Subset containing the unlabeled samples.
        """

        if self._cached_unlabeled_set is None:
            self._cached_unlabeled_set = IndexedSubset(self._full_train_set, self.unlabeled_idx)
        return self._cached_unlabeled_set
    
    @property
    def unlabeled_targets(self):
        """
        Property returning the targets of the unlabeled samples.

        Returns
        -------
            torch.Tensor: Tensor containing the targets of the unlabeled samples.
        """

        return self._full_train_set.targets[self.unlabeled_idx]

    @property
    def last_labeled_set(self):
        """
        Property returning the subset of the last labeled samples.

        Returns
        -------
            IndexedSubset: Subset containing the last labeled samples.
        """

        if self._cached_last_labeled_set is None:
            self._cached_last_labeled_set = IndexedSubset(self._full_train_set, self.last_labeled_idx)
        return self._cached_last_labeled_set
    
    @property
    def last_labeled_targets(self):
        """
        Property returning the targets of the last labeled samples.

        Returns
        -------
            torch.Tensor: Tensor containing the targets of the last labeled samples.
        """

        return self._full_train_set.targets[self.last_labeled_idx]
    
    @property
    def test_set(self):
        """
        Property returning the test subset.

        Returns
        -------
            IndexedSubset: Subset containing the test samples.
        """

        if self._cached_test_set is None:
            self._cached_test_set = IndexedSubset(self._full_test_set, self.test_subset_idx)
        return self._cached_test_set
    
    @property
    def test_targets(self):
        """
        Property returning the targets of the test samples.

        Returns
        -------
            torch.Tensor: Tensor containing the targets of the test samples.
        """

        return self._full_test_set.targets[self.test_subset_idx]
        

    def __init__(self, source, 
                 train_subset_size, 
                 test_subset_size,
                 test_subset_idx = None,
                 ratio_labeled=0.05, 
                 ratio_classes=None, 
                 balanced_split=True) -> None:
        """
        Parameters
        ----------
        source: str ("mnist" | "fashion" | "pcam")
            Name of the source dataset.
        train_subset_size: int|float 
            Size of the train subset. If the argument is of type `float` it's treated as a ratio.
        test_subset_size: int|float 
            Size of the test subset. If the argument is of type `float` it's treated as a ratio.
        test_subset_idx: numpy.array, optional 
            Manually sets the array of test indices. Overrules `test_subset_size` argument. (default = None)
        ratio_labeled: float, optional
            Ratio of the train subset that is initially labeled. (default = 0.05)
        ratio_classes: float 
            Ratio of the classes in the train subset. If None equal unform distribution is generated. (default = None)
        balanced_split: boolean 
            Decides whether to use `ratio_classes` to balance the split between classes. (defaul = True)
        """

        self.__get_from_source(source)
        self._cached_labeled_set = None
        self._cached_unlabeled_set = None
        self._cached_test_set = None
        self._cached_last_labeled_set = None

        train_size = len(self._full_train_set)
        test_size = len(self._full_test_set)

        # randomly choose labeled indices
        # we want the indices inside `labeled_idx` and `unlabled_idx` to be
        # global indices so that no matter what the subset is chosen
        # an index 'i' will always refer to the exact same sample
        # this is necessary for easier evaluation later on
        train_all_idx = np.arange(train_size)
        if balanced_split:
            if ratio_classes is None:
                ratio_classes = np.ones(len(self._full_train_set.classes))/len(self._full_train_set.classes)
                                                            
            self.train_subset_idx  = self.__get_balanced_train_subset(train_subset_size, 
                                                                 ratio_classes)
        else:  
            self.train_subset_idx = np.random.choice(train_all_idx, size=train_subset_size, replace=False)

        n_labeled = int(train_subset_size * ratio_labeled)
        self.labeled_idx = np.random.choice(self.train_subset_idx , size=n_labeled, replace=False)
        self.unlabeled_idx = np.setdiff1d(self.train_subset_idx , self.labeled_idx)
        self.last_labeled_idx = np.empty(0)

        if test_subset_idx is None:
            # get random test set
            test_all_idx = np.arange(test_size)
            self.test_subset_idx = np.random.choice(test_all_idx, size=test_subset_size, replace=False)
        else:
            self.test_subset_idx = test_subset_idx
        
    def __get_balanced_train_subset(self,train_subset_size, ratio_classes):
        """
        Generates a balanced train subset based on the specified ratio of classes.

        Parameters
        ----------
        train_subset_size: int 
            Size of the train subset.
            ratio_classes: float 
            Ratio of the classes in the train subset.

        Returns
        -------
            numpy.array: Indices of the balanced train subset.
        """
        classes_idx = {}
        for target in range(len(self._full_train_set.classes)):
            idx = np.where(self._full_train_set.targets == target)[0]
            classes_idx[target] = idx

        if abs(sum(ratio_classes) - 1) > 10**(-10):
            raise ValueError("Ratios of the classes should sum to 1")
        
        if len(ratio_classes) != len(self._full_train_set.classes):
            raise ValueError("Ratio classes arrays should be same length as total number of classes.")
        
        classes_size = {}
        for i, ratio in enumerate(ratio_classes):
            classes_size[i] = int(ratio*train_subset_size)

        classes_idx_subsets = {}
        for key, class_idx in classes_idx.items():
            class_indexes = class_idx
            
            if len(class_indexes) == 0:
                raise ValueError(f"There are not any indexes connected to class {self._full_train_set.classes[key]}")
            subset_idx_for_class = np.random.choice(class_indexes, size=classes_size[key], replace=False)

            if len(subset_idx_for_class) == 0:
                raise ValueError(f"In the subset there are not any indexes connected to class {self._full_train_set.classes[key]}")
            classes_idx_subsets[key] = subset_idx_for_class

        train_subset_idx  = []
        for value in classes_idx_subsets.values():
            train_subset_idx .extend(value)
        train_subset_idx  = np.array(train_subset_idx )
        return train_subset_idx 
    
    def __get_from_source(self, source):
        """
        Retrieves the datasets from the specified source.

        Parameters
        ----------
        source: str
            Name of the source dataset.

        Raises
        ------
            ValueError: If source name is invalid.
        """
        if source == "mnist":
            self._full_train_set = MNIST(root=_data_dir, download=False, train=True,
                                         transform=ToTensor(),
                                         target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self._full_test_set = MNIST(root=_data_dir, download=False, train=False,
                                        transform=ToTensor(),
                                        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif source == "fashion":
            self._full_train_set = FashionMNIST(root=_data_dir, download=False, train=True,
                                                transform=ToTensor(),
                                                target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self._full_test_set = FashionMNIST(root=_data_dir, download=False, train=False,
                                               transform=ToTensor(),
                                               target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        
        elif source == "pcam":
            # raise NotImplementedError
            self._full_train_set = pcamx.PCAMX(root=_data_dir, download=False, split='train', 
                                        transform=ToTensor(),
                                        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self._full_test_set = pcamx.PCAMX(root=_data_dir, download=False, split='val', 
                                      transform=ToTensor(),
                                      target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        else:
            raise ValueError("Invalid source name")

    def get_label_by_idx(self, indices, move_sample=True):
        """
        Gets the label of an unlabeled sample.

        Parameters
        ----------
        indices: int or list or numpy.array
            Index or indices of the samples in the unlabeled subset.
        move_sample: bool, optional
            Determines whether to move the sample to the labeled subset. (defaul = True)

        Returns
        -------
            list: List of labels of the specified unlabeled samples.
        """

        self.last_labeled_idx = self.unlabeled_idx[indices]

        # ensure indices dimension is not 0 (it's 0 when it's just a number i.e. one index not a list)
        if len(self.last_labeled_idx.shape) == 0:
            self.last_labeled_idx = self.last_labeled_idx.reshape(-1) 

        if move_sample:
            self._cached_labeled_set = None
            self._cached_unlabeled_set = None
            self._cached_last_labeled_set = None
            self.labeled_idx = np.concatenate([self.labeled_idx, self.last_labeled_idx])
            self.unlabeled_idx = np.setdiff1d(self.unlabeled_idx, self.labeled_idx)
        return [self._full_train_set.targets[global_idx] for global_idx in self.last_labeled_idx]
        
    def get_bootstrap_set(self, size=None, weights=None):
        """
        Generates a bootstrap set from the labeled samples.

        Parameters
        ----------
        size: int, optional
            Size of the bootstrap set. (defaul = None)
        weights: numpy.array or torch.Tensor
            Weights for sampling the labeled samples. (defaul = None)

        Returns
        -------
            IndexedSubset: Bootstrap set.
        """

        if size is None:
            size = len(self.labeled_idx)
        if weights is None:
            weights = np.ones(size)
        elif type(weights) == torch.Tensor:
            weights = weights.detach().numpy()
        p = weights / np.sum(weights)
            

        bootstrap_idx = np.random.choice(self.labeled_idx, size=size, replace=True, p=p)
        return IndexedSubset(self._full_train_set, bootstrap_idx)


def download_data():
    """
    Downloads necessary datasets of considerable size.

    This function prompts the user for confirmation before downloading the datasets. It downloads the following datasets:
    - MNIST: Train and test sets.
    - PCAM: Train, validation, and test sets.
    - FashionMNIST: Train and test sets.

    Note
    ----
        This function requires an active internet connection to download the datasets.
    """

    print("WARNING! You are about to download necessary datasets of considerable size.")
    answer = input("Proceed? [y/n]")
    if answer.lower() in ['y', 'yes']:
        MNIST(root=_data_dir, download = True, train=False)
        MNIST(root=_data_dir, download = True, train=True)

        PCAM(root=_data_dir, download = True, split='train')
        PCAM(root=_data_dir, download = True, split='val')
        PCAM(root=_data_dir, download = True, split='test')

        FashionMNIST(root=_data_dir, download=True, train=False)
        FashionMNIST(root=_data_dir, download=True, train=True)
    else:
        print("Aborted.")

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    main()