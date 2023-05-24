
import torch
import h5py
import pathlib
from torchvision.datasets import PCAM

TRAIN_SIZE = 262144

root = 'data'
base_folder = pathlib.Path(root) / "pcam"
targets_file = PCAM._FILES["train"]["targets"][0]

with h5py.File(base_folder / targets_file) as targets_data:
    targets = torch.zeros(size=(TRAIN_SIZE,)) 
    for idx in range(TRAIN_SIZE):
        targets[idx] = int(targets_data["y"][idx, 0, 0, 0])
    torch.save(targets, base_folder / "pcamx_split_train_y.pt")
print("Finished")