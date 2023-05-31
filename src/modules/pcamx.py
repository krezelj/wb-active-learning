from typing import Callable, Optional, Any, Tuple, Union, Literal

import torch
from torchvision.datasets import PCAM
import numpy as np
import os

CROP_MIN = 0
CROP_MAX = 96
CROPPED_SIZE = CROP_MAX - CROP_MIN

class PCAMLazyLoader():
    """
    A lazy loader for the PCAM dataset.
    ...
    Attributes
    ----------
    cached_data: dict
        Stores cashed data for each split in the Lazy Loader onstance.
    cached_idx: dict
        Keeps track of the indices that have been cashed for each split.
    idx_maps: dict
        Maps the original indices to tensor indices for each split.

    Methods
    -------
    reset(train_idx=None, test_idx=None, val_idx=None)
        Resets the instance of PCAMLazyLoader with the provided or default indices.
    __init_idx_maps(train_idx, test_idx, val_idx)
        Initializes the index maps for the training, testing, and validation sets 
        based on the provided or default indices.
    __init_idx_map(split, idx)
        Initializes an index map for a specific split with the given indices.
    __get_tensor_idx(split, idx)
        Returns the tensor index for a specific split and index.
    append_idx(split, idx)
        Appends new indices to the specified split. 
        Extends the existing tensor sizes to accommodate the new indices.
    set_full_data(split: str, full_images: torch.Tensor, full_targets: torch.Tensor)
        Sets the full data for a specific split. Assumes that the provided tensors 
        contain all the data for the given split.
    getitem(split, idx)
        Retrieves an item, from the specified split and index.
    putitem(split, idx, image, target)
       Stores an item in the specified split and index. 
    """

    __slots__ = ['cached_data', 'cached_idx', 'idx_maps']

    # source: https://github.com/basveeling/pcam
    _TRAIN_SIZE = 262_144   # 2^18
    _TEST_SIZE = 32_768     # 2^15
    _VAL_SIZE = 32_768

    def __init__(self, train_idx=None, test_idx=None, val_idx=None) -> None:
        """

        Parameters
        ----------
        train_idx: numpy.ndarray, optional
            Indices for the training set. (default = None)
        test_idx: numpy.ndarray, optional
            Indices for the testing set. (default = None)
        val_idx: numpy.ndarray, optional
            Indices for the validation set. (default = None)
        """
        
        self.reset(train_idx, test_idx, val_idx)

    def reset(self, train_idx=None, test_idx=None, val_idx=None) -> None:
        """
        Resets the instance of PCAMLazyLoader with the provided or default indices.

        Parameters
        ----------
        train_idx: numpy.ndarray, optional
            Indices for the training set. (default = None)
        test_idx: numpy.ndarray, optional
            Indices for the testing set. (default = None)
        val_idx: numpy.ndarray, optional
            Indices for the validation set. (default = None)
        """

        self.__init_idx_maps(train_idx, test_idx, val_idx)

        train_size = len(self.idx_maps['train'])
        test_size = len(self.idx_maps['test'])
        val_size = len(self.idx_maps['val'])

        self.cached_data = {
            'train': {
                'images': torch.zeros(size=(train_size, 3, CROPPED_SIZE, CROPPED_SIZE)),
                'targets': torch.zeros(size=(train_size,2)),
            },
            'test': {
                'images': torch.zeros(size=(test_size, 3, CROPPED_SIZE, CROPPED_SIZE)),
                'targets': torch.zeros(size=(test_size,2)),
            },
            'val': {
                'images': torch.zeros(size=(val_size, 3, CROPPED_SIZE, CROPPED_SIZE)),
                'targets': torch.zeros(size=(val_size,2)),
            }
        }
        self.cached_idx = {
            'train': set(),
            'test': set(),
            'val': set()
        }

    def __init_idx_maps(self, train_idx, test_idx, val_idx) -> None:
        """
        Initializes the index maps for the training, testing, and validation sets based on the provided or default indices.

        Parameters
        ----------
        train_idx: numpy.ndarray, optional
            Indices for the training set. (default = None)
        test_idx: numpy.ndarray, optional
            Indices for the testing set. (default = None)
        val_idx: numpy.ndarray, optional
            Indices for the validation set. (default = None)
        """

        self.idx_maps = {
            'train': {},
            'test': {},
            'val': {}
        }

        if train_idx is None:
            train_idx = np.arange(self._TRAIN_SIZE)
        self.__init_idx_map('train', train_idx)

        if test_idx is None:
            test_idx = np.arange(self._TEST_SIZE)
        self.__init_idx_map('test', test_idx)

        if val_idx is None:
            val_idx = np.arange(self._VAL_SIZE)
        self.__init_idx_map('val', val_idx)

    def __init_idx_map(self, split, idx) -> None:
        """
        Initializes an index map for a specific split with the given indices.

        Args:
        split: str 
            Split name ('train', 'test', 'val').
        idx: numpy.ndarray 
            Indices for the split.
        """

        self.idx_maps[split] = {}
        for i, idx in enumerate(idx):
            self.idx_maps[split][idx] = i

    def __get_tensor_idx(self, split, idx):
        """
        Returns the tensor index for a specific split and index.

        Parameters
        ----------
        split: str
            Split name ('train', 'test', 'val').
        idx: int
            Index within the split.

        Returns:
            int: Tensor index.
        
        Raises:
            KeyError: If the memory for the given index is not currently allocated.
        """

        try:
            return self.idx_maps[split][idx]
        except:
            raise KeyError(f"Memory for index {idx} is not currently allocated.")

    def append_idx(self, split, idx) -> None:
        """
        Appends new indices to the specified split. 
        Extends the existing tensor sizes to accommodate the new indices.

        Parameters
        ----------
        split: str 
            Split name ('train', 'test', 'val').
        idx: numpy.ndarray
            Indices to append.
        """

        idx = np.reshape(idx, -1) # ensure idx is an array

        current_idx = np.array(list(self.idx_maps[split].keys()))
        i_offset = len(current_idx)
        new_idx = np.setdiff1d(idx, current_idx)

        new_images = torch.zeros(size=(len(new_idx), 3, CROPPED_SIZE, CROPPED_SIZE))
        new_targets = torch.zeros(size=(len(new_idx),2))
        self.cached_data[split]['images'] = torch.cat([self.cached_data[split]['images'], new_images])
        self.cached_data[split]['targets'] = torch.cat([self.cached_data[split]['targets'], new_targets])

        for i, idx in enumerate(new_idx):
            self.idx_maps[split][idx] = i + i_offset


    def set_full_data(self, split : str, full_images : torch.Tensor, full_targets : torch.Tensor) -> None:
        """
        Sets the full data for a specific split. 
        Assumes that the provided tensors contain all the data for the given split.

        Parameters
        ----------
        split: str 
            Split name ('train', 'test', 'val').
        full_images: torch.Tensor 
            Full images tensor for the split.
        full_targets: torch.Tensor 
            Full targets tensor for the split.
        """

        idx = np.arange(len(full_targets))
        self.__init_idx_map(split, idx)
        self.cached_data[split]['images'] = full_images
        self.cached_data[split]['targets'] = full_targets
        self.cached_idx[split] = set(idx)


    def getitem(self, split, idx):
        """
        Retrieves an item from the specified split and index.

        Parameters
        ----------
        split: str 
            Split name ('train', 'test', 'val').
        idx: int 
            Index within the split.

        Returns:
            tuple: Tuple containing the image and target tensors.
        """

        tensor_idx = self.__get_tensor_idx(split, idx)
        if tensor_idx not in self.cached_idx[split]:
            return None, None
        image = self.cached_data[split]['images'][tensor_idx]
        target = self.cached_data[split]['targets'][tensor_idx]
        return image, target

    def putitem(self, split, idx, image, target) -> None:
        """
        Stores an item in the specified split and index.

        Parameters
        ----------
        split: str
            Split name ('train', 'test', 'val').
        idx: int
            Index within the split.
        image: torch.Tensor
            Image tensor.
        target: torch.Tensor
            Target tensor.
        """
        tensor_idx = self.__get_tensor_idx(split, idx)
        self.cached_idx[split].add(tensor_idx)
        self.cached_data[split]['images'][tensor_idx] = image
        self.cached_data[split]['targets'][tensor_idx] = target

# Example usage:
print("Initialising Lazy Loader (PCAMLL)")
PCAMLL = PCAMLazyLoader(train_idx=[], test_idx=[], val_idx=[])

class PCAMX(PCAM):
    """
    Extended implementation of the PCAM dataset that includes `classes` and `targets` fields.
    ...
    Attributes
    ----------
    targes: torch.Tensor
        Represents the target labels associated with each data sample in the dataset.
    
    Methods
    -------
    __getitem__(idx: int)
         Retrieves an item from the dataset at the specified index.
    """
    __slots__ = ['targets']

    classes = [0, 1]

    def __init__(self, root: str, split: str = "train", transform = None, target_transform = None, download: bool = False) -> None:
        """
        Parameters
        ----------
        root: str 
            Root directory of the dataset.
        split: str, optional 
            Split name ('train', 'test', 'val'). (default = 'train')
        transform: callable, optional 
            Optional transform to be applied on the image samples. (default = None)
        target_transform: callable, optional
            Optional transform to be applied on the target samples. (default = None)
        download: bool, optional
            If True, downloads the dataset if it's not already downloaded. (default = None)
        """
        super().__init__(root, split, transform, target_transform, download)
        path_to_targets = os.path.join(root, 'pcam', f'pcamx_split_{split}_y.pt')
        if os.path.isfile(path_to_targets):
            self.targets = torch.load(path_to_targets)
        else:
            print(f"WARNING! Could not find target tensor ({path_to_targets}).\n\
                  You will not be able to use balanced split on this set ({split} split).")
            self.targets = PCAMTargets(self)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Parameters
        ----------
        idx: int
            Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the cropped image data and the target.

        Notes:
            - If the cropped_data is not already cached, it retrieves the original data and crops it.
            - Caches the cropped_data and target in the PCAMLazyLoader instance for faster access.

        """
        cropped_data, target = PCAMLL.getitem(self._split, idx)
        if cropped_data is None:
            data, target = super().__getitem__(idx)
            cropped_data = data[:,CROP_MIN:CROP_MAX,CROP_MIN:CROP_MAX]
            PCAMLL.putitem(self._split, idx, cropped_data, target)
        return cropped_data, target


class PCAMTargets():
    """
    Represents the target labels associated with the PCAMX dataset. 
    ...
    Attributes
    ----------
    parent_dataset: PCAMX
        Instance of the PCAMX class, representing the parent dataset 
        that holds the image samples and their corresponding target labels.
    """

    __slots__ = ['parent_dataset']

    def __init__(self, parent_dataset : PCAMX) -> None:
        """
        Parameters
        ----------
        parent_dataset: PCAMX
            The parent PCAMX dataset.

        """

        self.parent_dataset = parent_dataset

    def __getitem__(self, idxs):
        """
        Parameters
        ----------
        idxs: int or array-like 
            Indices of the target labels to retrieve.

        Returns:
            torch.Tensor: Tensor containing the target labels corresponding to the specified indices.

        """

        idxs = np.reshape(idxs, -1)

        targets = torch.tensor([], dtype=torch.int32)
        for idx in idxs:
            targets = torch.cat((targets, torch.tensor([self.parent_dataset[idx][1].argmax()])))
            return targets