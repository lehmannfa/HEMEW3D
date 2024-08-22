import numpy as np
import argparse
import h5py

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of test samples")
parser.add_argument('@Irun0', type=int, default=0, help="Offset index")
parser.add_argument('@S_in', type=int, default=32, help="Size of the geology for the network")
parser.add_argument('@S_in_z', type=int, default=32, help="Depth of the geology for the network")
parser.add_argument('@Nt', type=int, default=320, help='Number of time steps in the output (only used for the folder name here)')
parser.add_argument('@fmax', type=int, default=5, help='Maximum frequency for filtering (only used for the folder name here)')
options = parser.parse_args().__dict__

S_in = options['S_in'] # size of the geology (x and y) for the network input
S_in_z = options['S_in_z']
Nt = options['Nt']
Ntrain = options['Ntrain']
Nval = options['Nval']
Irun0 = options['Irun0']
fmax = options['fmax']

path_save = '../inputs/' # where to save the machine learning inputs
path_a = '../../data/' # path to original raw data
folder_name = f'inputs3D_S{S_in}_Z{S_in_z}_T{Nt}_fmax{fmax}'


# Load materials
data_a = np.load(path_a + 'materials0-9.npy')
data_a = data_a[Irun0:Irun0+Ntrain+Nval, :32, :32, :32]
data_a = data_a.astype(np.float32)

# Save data to individual .h5 files
for i in range(Ntrain):
    with h5py.File(f'{path_save}{folder_name}_train/sample{Irun0+i}.h5', 'w') as f:
        f.create_dataset('a', data=data_a[i])
for i in range(Ntrain, Ntrain+Nval):
    with h5py.File(f'{path_save}{folder_name}_val/sample{Irun0+i}.h5', 'w') as f:
        f.create_dataset('a', data=data_a[i])
