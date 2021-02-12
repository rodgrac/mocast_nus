import h5py
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class NuScenes_HDF(Dataset):
    def __init__(self, hdf_path, transform):
        self.hf = h5py.File(hdf_path, 'r')
        self.transform = transform

    def __getitem__(self, index):
        dict = {}
        for gp in self.hf.keys():
            if gp == 'token':
                dict[gp] = self.hf.get(gp).get(str(index)).asstr()[0]
            elif gp == 'image':
                dict[gp] = self.transform(np.array(self.hf.get(gp).get(str(index))))
            else:
                dict[gp] = torch.Tensor(np.array(self.hf.get(gp).get(str(index)))).squeeze()

        return dict

    def __len__(self):
        return len(self.hf.get('token').values())

    def close_hf(self):
        self.hf.close()
