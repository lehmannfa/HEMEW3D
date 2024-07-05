import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import argparse
import h5py

from simulation.simulation_files import butter_lowpass_filter

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of test samples")
parser.add_argument('@S_out', type=int, default=32, help="Size of the sensors array (interpolated)")
parser.add_argument('@Nt', type=int, default=320, help="Number of time steps")
parser.add_argument('@f', type=int, default=50, help="Sampling frequency")
parser.add_argument('@fmax', type=int, default=5, help="Maximum frequency to filter signals")
parser.add_argument('@Irun0', type=int, default=100000, help='Index of the first element to use in the folder path_u')

options = parser.parse_args().__dict__

S_out = options['S_out'] # size of the wavefield grid
Nt = options['Nt'] # number of time steps
f = options['f'] # sampling frequency
assert 100 % f ==0, f'you asked a sampling frequency of {f}Hz that is not a divider of the recording frequency of 100Hz'
fmax = options['fmax']
assert fmax <= 5, f'The mesh is designed up to a 5Hz frequency, you cannot request {fmax}Hz'
Ntrain = options['Ntrain']
Nval = options['Nval']
Irun0 = options['Irun0']

path_save = '../inputs/' # where to save the machine learning inputs we are creating
path_u = '../../data/' # path to the original raw data
folder_name = f'inputs3D_S{S_out}_Z{S_out}_T{Nt}_fmax{fmax}' # folder containing ML inputs


for i in range(Irun0, Irun0+Ntrain):
    g = h5py.File(path_u + f'sample{i}.h5', 'r')
    for comp in ['uE', 'uN', 'uZ']:
        u = g[comp][:].reshape(S_out*S_out, -1) # reshape to (x*y, time)
        u = butter_lowpass_filter(u, fmax, dt=0.01, order=4) # filter to remove high frequencies
        u = u.reshape(S_out, S_out, -1) # reshape to (x, y, time)
        u = u[:, :, ::int(100/f)] # temporal downscaling
        u = u[:, :, :Nt] # restrict the time window
        
        with h5py.File(path_save + folder_name + f'_train/sample{i}.h5', 'a') as h:
            h.create_dataset(comp, data = u, dtype=np.float32)

    if i%1000 == 0:
        print(f"done for files train {i}/{Irun0+Ntrain}")

        
for i in range(Irun0+Ntrain, Irun0+Ntrain+Nval):
    g = h5py.File(path_u + f'sample{i}.h5', 'r')
    for comp in ['uE', 'uN', 'uZ']:
        u = g[comp][:].reshape(S_out*S_out, -1) # reshape to (x*y, time)
        u = butter_lowpass_filter(u, fmax, dt=0.01, order=4) # filter to remove high frequencies
        u = u.reshape(S_out, S_out, -1) # reshape to (x, y, time)
        u = u[:, :, ::int(100/f)] # temporal downscaling
        u = u[:, :, :Nt] # restrict the time window
        
        with h5py.File(path_save + folder_name + f'_val/sample{i}.h5', 'a') as h:
            h.create_dataset(comp, data = u, dtype=np.float32)

    if i%1000 == 0:
        print(f"done for files val {i}/{Irun0+Ntrain+Nval}")