import torch
from torchvision.datasets import MNIST, PCAM
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Subset
import numpy as np


# Used to create custom `Subset` classes that return index along with data and target
# this is necessary because if we want to apply weight during mini batch training
# we need to know which weights to choose 
def indexed_subset(cls):
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class IndexedSubset(Subset):

    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        return data, target, idx



class ActiveDataset():
    
    # TODO Add sample weights as property
    # weights only apply to labeled samples
    # additionally there should be a function "update_weights"
    # that updates the weights based on the number of feedback loop iterations
    # 
    # The exact implementation is to be discussed

    __slots__ = ['train_dataset', 'test_dataset', 'labeled_idx', 'unlabeled_idx']

    @property
    def labeled_set(self):
        return IndexedSubset(self.train_dataset, self.labeled_idx)
    
    @property
    def unlabeled_set(self):
        return IndexedSubset(self.train_dataset, self.unlabeled_idx)
        

    def __init__(self, source, subset_size, ratio_labeled=0.05, ratio_classes=None, balanced_split=True) -> None:
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
        size = len(self.train_dataset)

        # randomly choose labeled indices
        # we want the indices inside `labeled_idx` and `unlabled_idx` to be
        # global indices so that no matter what the subset is chosen
        # an index 'i' will always refer to the exact same sample
        # this is necessary for easier evaluation later on
        all_indices = np.arange(size)
        subset_indices = np.random.choice(all_indices, size=subset_size, replace=False)

        n_labeled = int(subset_size * ratio_labeled)
        self.labeled_idx = np.random.choice(subset_indices, size=n_labeled, replace=False)
        self.unlabeled_idx = np.setdiff1d(subset_indices, self.labeled_idx)

        # TODO Implement class balancing
        # Suggested way to do this:
        # for each class calculate number of samples of that class (class_i_proportion * subset_size)
        # get indices of all classes in separate arrays (one array per class containing indices for all samples of that class)
        # np.random.choice(class_i_indices, size=class_i_size, replace=False)
        
    def __get_from_source(self, source):
        # TODO fix pathing, instead of '../../data' make it a statis variable (or something)

        if source == "mnist":
            self.train_dataset = MNIST(root="../../data", download=False, train=True, 
                                       transform=ToTensor(),
                                       target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self.test_dataset = MNIST(root="../../data", download=False, train=False, 
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
            `idx` (int): index of the sample in the **unlabeled** set\n
        Parameters:
            `move_sample` (bool): decided whether to move the sample to the labeled
        """

        global_indices = self.unlabeled_idx[indices]
        if move_sample:
            self.labeled_idx = np.concatenate([self.labeled_idx, global_indices])
            self.unlabeled_idx = np.setdiff1d(self.unlabeled_idx, self.labeled_idx)
        return [self.train_dataset.targets[global_idx] for global_idx in global_indices]
        
    

def download_data():
    print("WARNING! You are about to download necessary datasets of considerable size.")
    answer = input("Proceed? [y/n]")
    if answer.lower() in ['y', 'yes']:
        MNIST(root = './data/', download = True)
        PCAM(root='./data', download=True)
    else:
        print("Aborted.")

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    main()