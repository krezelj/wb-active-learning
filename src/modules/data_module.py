import os
from typing import Callable, Optional, Any, Tuple, Union, Literal

import torch
from torchvision.datasets import MNIST, PCAM, FashionMNIST
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Subset
import numpy as np


# Note: this variable should not be updated directly.
# Instead, use the update_data_dir() function to ensure that the stored path
# is always absolute.
_data_dir = ''
# (default value for __data_dir is assigned below update_data_dir() function declaration)


def update_data_dir(path, silent=False):
    """
    Updates the path to a directory where datasets are stored and downloaded to

    Arguments:
        `path` (str|os.PathLike): path to a new data directory\n
    Parameters:
        `silent` (bool): whether to notify about the new path
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


class PCAMLazyLoader():

    __slots__ = ['cached_data']

    # source: https://github.com/basveeling/pcam
    _TRAIN_SIZE = 262_144   # 2^18
    _TEST_SIZE = 32_768     # 2^15
    _VAL_SIZE = 32_768

    def __init__(self):
        self.cached_data = {
            'train': {
                'images': [None] * self._TRAIN_SIZE,
                'targets': [None] * self._TRAIN_SIZE
            },
            'test': {
                'images': [None] * self._TEST_SIZE,
                'targets': [None] * self._TEST_SIZE,
            },
            'val': {
                'images': [None] * self._VAL_SIZE,
                'targets': [None] * self._VAL_SIZE,
            }
        }

    def getitem(self, split, idx):
        image = self.cached_data[split]['images'][idx]
        target = self.cached_data[split]['targets'][idx]
        return image, target


    def putitem(self, split, idx, image, target):
        self.cached_data[split]['images'][idx] = image
        self.cached_data[split]['targets'][idx] = target

print("Initialising Lazy Loader (PCAMLL)")
PCAMLL = PCAMLazyLoader()

class PCAMX(PCAM):
    """
    Extended implementation of the PCAM dataset that includes `classes` and `targets` fields.
    """

    class PCAMTargets():
        """
        Class for dynamic target loading.
        """

        __slots__ = ['parent_dataset']

        def __init__(self, parent_dataset):
            self.parent_dataset = parent_dataset

        def __getitem__(self, idxs):
            idxs = np.reshape(idxs, -1)

            targets_file = self.parent_dataset._FILES[self.parent_dataset._split]["targets"][0]
            with self.parent_dataset.h5py.File(self.parent_dataset._base_folder / targets_file) as targets_data:
                targets = []
                for idx in idxs:
                    targets.append(int(targets_data["y"][idx, 0, 0, 0]))
            return torch.tensor(targets)


    __slots__ = ['classes', 'targets']

    def __init__(self, root: str, split: str = "train", transform = None, target_transform = None, download: bool = False):
        super().__init__(root, split, transform, target_transform, download)

        self.classes = ['0 - no tumor tissue', '1 - tumor tissue present']
        self.targets = self.PCAMTargets(self)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        cropped_data, target = PCAMLL.getitem(self._split, idx)
        if cropped_data is None:
            data, target = super().__getitem__(idx)
            cropped_data = data[:,32:64,32:64]
            PCAMLL.putitem(self._split, idx, cropped_data, target)
        return cropped_data, target


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
        

    def __init__(self, source, 
                 train_subset_size, 
                 test_subset_size,
                 test_idx = None,
                 ratio_labeled=0.05, 
                 ratio_classes=None, 
                 balanced_split=True) -> None:
        """
        Initialises the active dataset object

        Arguments:
            `source` ("mnist" | "fashion" | "pcam"): name of the source dataset.
            `train_subset_size` (int|float): size of the train subset. If the argument is of type `float` it's treated as a ratio.
            `test_subset_size` (int|float): size of the test subset. If the argument is of type `float` it's treated as a ratio.
        
        Parameters:
            `test_idx` (numpy.array): manually sets the array of test indices. Overrules `test_subset_size` argument.
            `ratio_labeled` (float): ratio of the train subset that is initially labeled.
            `ratio_classes` (float): ratio of the classes in the train subset. If None equal unform distribution is generated.
            `balanced_split` (boolean): decides whether to use `ratio_classes` to balance the split between classes.
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
                                                            
            train_subset_idx  = self.__get_balanced_train_subset(train_subset_size, 
                                                                 ratio_classes)

        else:  
            train_subset_idx = np.random.choice(train_all_idx, size=train_subset_size, replace=False)

        n_labeled = int(train_subset_size * ratio_labeled)
        self.labeled_idx = np.random.choice(train_subset_idx , size=n_labeled, replace=False)
        self.unlabeled_idx = np.setdiff1d(train_subset_idx , self.labeled_idx)
        self.last_labeled_idx = np.empty(0)

        if test_idx is None:
            # get random test set
            test_all_idx = np.arange(test_size)
            self.test_idx = np.random.choice(test_all_idx, size=test_subset_size, replace=False)
        else:
            self.test_idx = test_idx
        
    def __get_balanced_train_subset(self,train_subset_size, ratio_classes):
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
            self._full_train_set = PCAMX(root=_data_dir, download=False, split='train', 
                                        transform=ToTensor(),
                                        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self._full_test_set = PCAMX(root=_data_dir, download=False, split='val', 
                                      transform=ToTensor(),
                                      target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
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