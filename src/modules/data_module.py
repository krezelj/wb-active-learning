import torch
from torchvision.datasets import MNIST, PCAM
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Subset
import numpy as np


class Dataset():
    
    # TODO Add sample weights as property
    # weights only apply to labeled samples
    # additionally there should be a function "update_weights"
    # that updates the weights based on the number of feedback loop iterations
    # 
    # The exact implementation is to be discussed

    __slots__ = ['train_dataset', 'test_dataset', 'labeled_idx', 'unlabeled_idx']

    @property
    def labeled_set(self):
        return Subset(self.train_dataset, self.labeled_idx)
    
    @property
    def unlabeled_set(self):
        return Subset(self.train_dataset, self.unlabeled_idx)
    

    def __init__(self, source, size, ratio_labeled=0.05, ratio_classes=None, balanced_split=True) -> None:
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
        n_labeled = int(size * ratio_labeled)
        all_indices = np.arange(size)
        self.labeled_idx = np.random.choice(np.arange(size), size=n_labeled, replace=False)
        self.unlabeled_idx = np.setdiff1d(all_indices, self.labeled_idx)

        # TODO Implement class balancing
        
    def __get_from_source(self, source):
        if source == "mnist":
            self.train_dataset = MNIST(root="./data", download=False, train=True, 
                                       transform=ToTensor(),
                                       target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
            self.test_dataset = MNIST(root="./data", download=False, train=False, 
                                      transform=ToTensor(),
                                      target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        elif source == "pcam":
            raise NotImplementedError
            # self.train_dataset = PCAM(root="./data", download=False, train=True, transform=ToTensor())
            # self.test_dataset = PCAM(root="./data", download=False, train=False, transform=ToTensor())
        else:
            raise ValueError("Invalid source name")

    def get_label_by_idx(self, idx, move_sample=True):
        """
        Gets the label of an unlabeled sample.

        Arguments:
            `idx` (int): index of the sample in the **unlabeled** set\n
        Parameters:
            `move_sample` (bool): decided whether to move the sample to the labeled
        """

        global_idx = self.unlabeled_idx[idx]
        if move_sample:
            self.labeled_idx = np.concatenate([self.labeled_idx, [global_idx]])
            self.unlabeled_idx = np.concatenate([self.unlabeled_idx[:idx], self.unlabeled_idx[idx+1:]])
        return self.train_dataset[global_idx][1]
    

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