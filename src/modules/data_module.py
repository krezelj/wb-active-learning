import torch
from torchvision.datasets import MNIST, PCAM, FashionMNIST
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Subset
import numpy as np


class IndexedSubset(Subset):

    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        return data, target, idx



class ActiveDataset():

    # TODO Add ability to manually set test set so that it's consistent across several tests
    
    __slots__ = ['_full_train_set', '_full_test_set', 'labeled_idx', 'unlabeled_idx', 'last_labeled_idx', 'test_idx',
                 '_cached_test_set', '_cached_labeled_set', '_cached_unlabeled_set', '_cached_last_labeled_set']

    @property
    def labeled_set(self):
        if self._cached_labeled_set is None:
            self._cached_labeled_set = IndexedSubset(self._full_train_set, self.labeled_idx)
        return self._cached_labeled_set
    
    @property 
    def labeled_targets(self):
        return self._full_train_set.targets[self.labeled_idx]
    
    @property
    def unlabeled_set(self):
        if self._cached_unlabeled_set is None:
            self._cached_unlabeled_set = IndexedSubset(self._full_train_set, self.unlabeled_idx)
        return self._cached_unlabeled_set
    
    @property
    def unlabeled_targets(self):
        return self._full_train_set.targets[self.unlabeled_idx]

    @property
    def last_labeled_set(self):
        if self._cached_last_labeled_set is None:
            self._cached_last_labeled_set = IndexedSubset(self._full_train_set, self.last_labeled_idx)
        return self._cached_last_labeled_set
    
    @property
    def last_labeled_targets(self):
        return self._full_train_set.targets[self.last_labeled_idx]
    
    @property
    def test_set(self):
        if self._cached_test_set is None:
            self._cached_test_set = IndexedSubset(self._full_test_set, self.test_idx)
        return self._cached_test_set
    
    @property
    def test_targets(self):
        return self._full_test_set.targets[self.test_idx]
        

    def __init__(self, source, train_subset_size, test_subset_size, 
                 ratio_labeled=0.05, ratio_classes=None, balanced_split=True) -> None:
        """
        Initialises the dataset object

        Arguments:
            `souce` ("mnist" | "pcam"): name of the source\n
            `size` (int|float):\n
        
        Parameters:
            `ratio_labeled` (float):\n
            `ratio_classes` (float):\n
            `balanced_split` (boolean):\n
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
        train_subset_indices = np.random.choice(train_all_idx, size=train_subset_size, replace=False)

        n_labeled = int(train_subset_size * ratio_labeled)
        self.labeled_idx = np.random.choice(train_subset_indices, size=n_labeled, replace=False)
        self.unlabeled_idx = np.setdiff1d(train_subset_indices, self.labeled_idx)
        self.last_labeled_idx = np.empty(0)

        # get random test set
        test_all_idx = np.arange(test_size)
        self.test_idx = np.random.choice(test_all_idx, size=test_subset_size, replace=False)

        # TODO Implement class balancing
        # Suggested way to do this:
        # for each class calculate number of samples of that class (class_i_proportion * subset_size)
        # get indices of all classes in separate arrays (one array per class containing indices for all samples of that class)
        # np.random.choice(class_i_indices, size=class_i_size, replace=False)
        
    def __get_from_source(self, source):
        # TODO fix pathing, instead of '../../data' make it a statis variable (or something)

        if source == "mnist":
            self._full_train_set = MNIST(root="../../data", download=False, train=True, 
                                       transform=ToTensor(),
                                       target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self._full_test_set = MNIST(root="../../data", download=False, train=False, 
                                      transform=ToTensor(),
                                      target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif source == "fashion":
            self._full_train_set = FashionMNIST(root="../../data", download=False, train=True, 
                                       transform=ToTensor(),
                                       target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self._full_test_set = FashionMNIST(root="../../data", download=False, train=False, 
                                      transform=ToTensor(),
                                      target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        
        elif source == "pcam":
            raise NotImplementedError
            # self.train_dataset = PCAM(root="./data", download=False, train=True, transform=ToTensor())
            # self.test_dataset = PCAM(root="./data", download=False, train=False, transform=ToTensor())
        else:
            raise ValueError("Invalid source name")

    def get_label_by_idx(self, indices, move_sample=True):
        """
        Gets the label of an unlabeled sample.

        Arguments:
            `idx` (int): index of the sample in the **unlabeled** subset\n
        Parameters:
            `move_sample` (bool): decided whether to move the sample to the labeled
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
    print("WARNING! You are about to download necessary datasets of considerable size.")
    answer = input("Proceed? [y/n]")
    if answer.lower() in ['y', 'yes']:
        MNIST(root = '../../data/', download = True)
        PCAM(root='../../data', download=True)
        FashionMNIST(root='../../data', download=True)
    else:
        print("Aborted.")

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    main()