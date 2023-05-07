from typing import Callable, Optional, Any, Tuple, Union, Literal

import torch
from torchvision.datasets import PCAM
import numpy as np

class PCAMLazyLoader():

    __slots__ = ['cached_data', 'cached_idx']

    # source: https://github.com/basveeling/pcam
    _TRAIN_SIZE = 262_144   # 2^18
    _TEST_SIZE = 32_768     # 2^15
    _VAL_SIZE = 32_768

    def __init__(self):
        self.cached_data = {
            'train': {
                'images': torch.zeros(size=(self._TRAIN_SIZE, 3, 32, 32)),
                'targets': torch.zeros(size=(self._TRAIN_SIZE,2)),
            },
            'test': {
                'images': torch.zeros(size=(self._TEST_SIZE, 3, 32, 32)),
                'targets': torch.zeros(size=(self._TEST_SIZE,2)),
            },
            'val': {
                'images': torch.zeros(size=(self._VAL_SIZE, 3, 32, 32)),
                'targets': torch.zeros(size=(self._VAL_SIZE,2)),
            }
        }
        self.cached_idx = {
            'train': set(),
            'test': set(),
            'val': set()
        }


    def eagerload():
        # pre load ALL of the data at once so that all trials are fast
        raise NotImplementedError


    def getitem(self, split, idx):
        if idx not in self.cached_idx[split]:
            return None, None
        image = self.cached_data[split]['images'][idx]
        target = self.cached_data[split]['targets'][idx]
        return image, target

    def putitem(self, split, idx, image, target):
        self.cached_idx[split].add(idx)
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


    __slots__ = ['targets']

    classes = ['0 - no tumor tissue', '1 - tumor tissue present']

    def __init__(self, root: str, split: str = "train", transform = None, target_transform = None, download: bool = False):
        super().__init__(root, split, transform, target_transform, download)
        self.targets = self.PCAMTargets(self)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        cropped_data, target = PCAMLL.getitem(self._split, idx)
        if cropped_data is None:
            data, target = super().__getitem__(idx)
            cropped_data = data[:,32:64,32:64]
            PCAMLL.putitem(self._split, idx, cropped_data, target)
        return cropped_data, target