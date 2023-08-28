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
parser.add_argument('@Irun0', type=int, default=0, help='Index of the first element to use in the folder path_u')
parser.add_argument('@interpolate', action='store_true', help='Whether to interpolate or to keep the original dimensions')

options = parser.parse_args().__dict__

interpolate=options['interpolate']
interpolate_method = 'nearest'
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
traces_name = 'velocity'
folder_name = f'inputs3D_S{S_out}_Z{S_out}_T{Nt}_fmax{fmax}' # folder containing ML inputs


## Create the original and interpolated axes to increase the size of the spatial dimensions x and y
if interpolate:
    x = np.linspace(150, 9450, 16)
    x2 = np.linspace(150, 9450, S_out)
    z = np.linspace(150, 9450, 16)
    z2 = np.linspace(150, 9450, S_out)
    t = np.linspace(0, (Nt-1)/f, Nt) # values are not important because interpolation is only on x,y

    # array of points containing the new values of the grid points
    indices = pd.MultiIndex.from_product([x2, z2, t])
    indices = indices.to_frame(name=['x','z','t'])
    points = indices.sort_values(by=['x','z','t']).values

    
## Load each batch of velocity fields
# To process all data, replace the 2 following lines (example code) by the commented lines below
for ibatch in [0]:
    data_u = np.zeros((10, S_out, S_out, Nt), dtype=np.float32)
    traces = pd.read_feather(f"{path_u}{traces_name}{Irun0+ibatch}-{Irun0+ibatch+9}.feather")
    
# for ibatch in range(0, Ntrain+Nval, 100):
#     data_u = np.zeros((100, S_out, S_out, Nt), dtype=np.float32)
#     traces = pd.read_feather(f"{path_u}{traces_name}{Irun0+ibatch}-{Irun0+ibatch+99}.feather")
    
    traces.sort_values(by=['run', 'y', 'x'], inplace=True)
    traces.drop(['x','y','z'], axis=1, inplace=True)

    # filter frequencies higher than fmax
    traces_filtered = pd.DataFrame(butter_lowpass_filter(traces.iloc[:,2:], fmax, dt=0.01, order=4),
                                   index = traces.index, columns = traces.columns[2:])
    traces_filtered = pd.concat([traces.loc[:,['run','field']], traces_filtered], axis=1)
    
    # decrease the sampling frequency and limit to the interesting time steps (from 1s and up to Nt timesteps)
    traces_short = pd.concat([traces.loc[:,['run','field']], traces_filtered.iloc[:,::int(100/f)].iloc[:,f:Nt+f]], axis=1)

    # iterate over the components
    for comp in ['E', 'N', 'Z']:
        temp = traces_short.loc[(traces_short.field==f'Veloc {comp}')].iloc[:,2:].values.reshape(data_u.shape[0],16,16,Nt)

        for i in range(data_u.shape[0]): # operate one run after the other
            if interpolate:
                f_interp = RegularGridInterpolator((x, z, t), temp[i], bounds_error=True, method=interpolate_method)
                u = f_interp(points).reshape(S_out, S_out, Nt)
            else:
                u = temp[i]

            # add velocity fields to the files created by create_data_material.py
            if ibatch+i < Ntrain:
                with h5py.File(f'{path_save}{folder_name}_train/sample{Irun0+ibatch+i}.h5', 'a') as h5_file:
                    if f'u{comp}' in h5_file.keys():
                        h5_file[f'u{comp}'][:] = u.astype(np.float32)
                    else:
                        h5_file.create_dataset(f'u{comp}', data=u.astype(np.float32))
                print(f'saved {Irun0+ibatch+i} in train')
            else:
                with h5py.File(f'{path_save}{folder_name}_val/sample{Irun0+ibatch+i}.h5', 'a') as h5_file:
                    if f'u{comp}' in h5_file.keys():
                        h5_file[f'u{comp}'][:] = u.astype(np.float32)
                    else:
                        h5_file.create_dataset(f'u{comp}', data=u.astype(np.float32))
                print(f'saved {Irun0+ibatch+i} in val')

