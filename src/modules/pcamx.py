from typing import Callable, Optional, Any, Tuple, Union, Literal

import torch
from torchvision.datasets import PCAM
import numpy as np
import os

CROP_MIN = 0
CROP_MAX = 96
CROPPED_SIZE = CROP_MAX - CROP_MIN

class PCAMLazyLoader():

    __slots__ = ['cached_data', 'cached_idx', 'idx_maps']

    # source: https://github.com/basveeling/pcam
    _TRAIN_SIZE = 262_144   # 2^18
    _TEST_SIZE = 32_768     # 2^15
    _VAL_SIZE = 32_768

    def __init__(self, train_idx=None, test_idx=None, val_idx=None):
        self.reset(train_idx, test_idx, val_idx)

    def reset(self, train_idx=None, test_idx=None, val_idx=None):
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

    def __init_idx_maps(self, train_idx, test_idx, val_idx):
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

    def __init_idx_map(self, split, idx):
        self.idx_maps[split] = {}
        for i, idx in enumerate(idx):
            self.idx_maps[split][idx] = i

    def __get_tensor_idx(self, split, idx):
        try:
            return self.idx_maps[split][idx]
        except:
            raise KeyError(f"Memory for index {idx} is not currently allocated.")

    def append_idx(self, split, idx):
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


    def set_full_data(self, split : str, full_images : torch.Tensor, full_targets : torch.Tensor):
        # IMPORTANT! Assumes that provided tensors contains all of the data for the given split
        idx = np.arange(len(full_targets))
        self.__init_idx_map(split, idx)
        self.cached_data[split]['images'] = full_images
        self.cached_data[split]['targets'] = full_targets
        self.cached_idx[split] = set(idx)


    def getitem(self, split, idx):
        tensor_idx = self.__get_tensor_idx(split, idx)
        if tensor_idx not in self.cached_idx[split]:
            return None, None
        image = self.cached_data[split]['images'][tensor_idx]
        target = self.cached_data[split]['targets'][tensor_idx]
        return image, target

    def putitem(self, split, idx, image, target):
        tensor_idx = self.__get_tensor_idx(split, idx)
        self.cached_idx[split].add(tensor_idx)
        self.cached_data[split]['images'][tensor_idx] = image
        self.cached_data[split]['targets'][tensor_idx] = target

print("Initialising Lazy Loader (PCAMLL)")
PCAMLL = PCAMLazyLoader(train_idx=[], test_idx=[], val_idx=[])

class PCAMX(PCAM):
    """
    Extended implementation of the PCAM dataset that includes `classes` and `targets` fields.
    """
    __slots__ = ['targets']

    classes = [0, 1]

    def __init__(self, root: str, split: str = "train", transform = None, target_transform = None, download: bool = False):
        super().__init__(root, split, transform, target_transform, download)
        path_to_targets = os.path.join(root, 'pcam', f'pcamx_split_{split}_y.pt')
        if os.path.isfile(path_to_targets):
            self.targets = torch.load(path_to_targets)
        else:
            print(f"WARNING! Could not find target tensor ({path_to_targets}).\n\
                  You will not be able to use balanced split on this set ({split} split).")
            self.targets = PCAMTargets(self)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        cropped_data, target = PCAMLL.getitem(self._split, idx)
        if cropped_data is None:
            data, target = super().__getitem__(idx)
            cropped_data = data[:,CROP_MIN:CROP_MAX,CROP_MIN:CROP_MAX]
            PCAMLL.putitem(self._split, idx, cropped_data, target)
        return cropped_data, target


class PCAMTargets():
        __slots__ = ['parent_dataset']

        def __init__(self, parent_dataset : PCAMX):
            self.parent_dataset = parent_dataset

        def __getitem__(self, idxs):
            idxs = np.reshape(idxs, -1)

            targets = torch.tensor([], dtype=torch.int32)
            for idx in idxs:
                targets = torch.cat((targets, torch.tensor([self.parent_dataset[idx][1].argmax()])))
            return targets