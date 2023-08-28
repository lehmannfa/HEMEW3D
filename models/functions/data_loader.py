import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import re
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from models.functions.helpers import Normalization


class GeologyTracesDataset(Dataset):
    def __init__(self, path_data, dir_data=['inputs3D_S32_Z32_T320_fmax5_train'], T_out=320, S_in=32, S_in_z=32, S_out=32, transform_a='normal', N=None):
        ''' 
        path_data: string, path to directory with all data
        dir_data: list of strings, name of folders in path_data where data are stored
        T_out: int, number of time steps in outputs
        S_in: int, number of grid points along x and y, in inputs
        S_in_z: int, number of grid points along z, in inputs
        S_out: int, number of grid points along x and y, in outputs
        transform_a: string, normalization method for inputs. choice between "normal" and "scalar_normal"
        N: number of elements in total
        '''
        self.path_data = path_data # folder with all data
        self.dir_data = dir_data # name of folder in path_data where data are stored
        self.T_out = T_out # number of time steps in outputs
        self.S_in = S_in # number of grid points along x and y, in inputs
        self.S_in_z = S_in_z # number of grid points along z, in inputs
        self.S_out = S_out # number of grid points along x and y, in outputs
        self.transform_a = transform_a # normalization type

        a_mean = np.load(path_data + dir_data[0] + '/a_mean.npy')
        a_std = np.load(path_data + dir_data[0] + '/a_std.npy')

        if self.transform_a == 'scalar_normal':
            self.a_mean = np.mean(a_mean)
            self.a_std = np.mean(a_std)
        else:
            self.a_mean = a_mean
            self.a_std = a_std
        
        if self.transform_a == 'normal' or self.transform_a == 'scalar_normal':
            self.ANorm = Normalization(1, norm_type='normal', x_mean=self.a_mean, x_std=4*self.a_std) # the first argument has no influence since we are imposing the mean and std

        # list of all files
        self.all_files = []
        for indiv_dir_data in dir_data:
            l = os.listdir(self.path_data + indiv_dir_data)
            l = [item for item in l if item[:6] == 'sample']
            if len(l) == 0:
                raise Exception(f"folder {self.path_data + indiv_dir_data} is empty")
            l = sorted(l, key=lambda s: int(re.search(r'\d+', s).group()))
            self.all_files += [self.path_data+indiv_dir_data+'/'+li for li in l]

        if N is not None:
            self.all_files = self.all_files[:N]
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        f = h5py.File(self.all_files[idx], 'r')
        a = f['a'][:]
        if self.transform_a is not None:
            a = self.ANorm.forward(a)
        a = np.expand_dims(a, axis=3)
        
        uE = f['uE'][:]
        uE = np.expand_dims(uE, axis=3)
        
        uN = f['uN'][:]
        uN = np.expand_dims(uN, axis=3)
        
        uZ = f['uZ'][:]
        uZ = np.expand_dims(uZ, axis=3)

        f.close()
        
        return a, uE, uN, uZ
