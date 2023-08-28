''' This code is largely inspired by https://github.com/ashiq24/UNO '''

import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functions.fourier_blocks import *


class UNO_3D(nn.Module):
    def __init__(self, in_width, width, S, list_dv, list_Dx, list_Dy, list_Dz, list_Mx, list_My, list_Mz, padding = 0, verbose=False):
        ''' in_width: int, number of channels in the input (6 in 3D: sin(x), cos(x), sin(y), cos(y), z, a(x,y,z))
        width: int, number of channels created by the uplift operator
        S: int, 
        list_dv: list of ints, number of channels after each block
        list_Dx, list_Dy, list_Dz: list of ints, dimensions along x,y,z after each block
        list_Mx, list_My, list_Mz: list of ints, number of Fourier mode inside each Fourier block
        padding: int, number of zeros padding after the uplift operator 
        verbose: boolean, print intermediate inputs 
        '''

        super(UNO_3D, self).__init__()

        self.in_width = in_width
        self.width = width
        self.S = S
        self.padding = padding
        self.verbose = verbose

        # uplift operator : dx x dy x dt x in_width --> dx x dy x dt x width
        self.P0 = nn.Linear(self.in_width, self.width//2)
        self.P1 = nn.Linear(self.width//2, self.width)

        # reduction path
        self.L0 = OperatorBlock_3D(self.width, list_dv[0],  
                                   list_Dx[0], list_Dy[0], list_Dz[0],
                                   list_Mx[0], list_My[0], list_Mz[0])

        self.L1 = OperatorBlock_3D(list_dv[0], list_dv[1], 
                                   list_Dx[1], list_Dy[1], list_Dz[1],
                                   list_Mx[1], list_My[1], list_Mz[1])

        self.L2 = OperatorBlock_3D(list_dv[1], list_dv[2], 
                                   list_Dx[2], list_Dy[2], list_Dz[2],
                                   list_Mx[2], list_My[2], list_Mz[2])

        self.L3 = OperatorBlock_3D(list_dv[2], list_dv[3], 
                                   list_Dx[3], list_Dy[3], list_Dz[3],
                                   list_Mx[3], list_My[3], list_Mz[3])
        
        # first augmentation
        self.L4 = OperatorBlock_3D(list_dv[3], list_dv[4], 
                                   list_Dx[4], list_Dy[4], list_Dz[4],
                                   list_Mx[4], list_My[4], list_Mz[4])
        
        # augmentations with concatenated inputs, hence the factor 2
        self.L5 = OperatorBlock_3D(int(2*list_dv[4]), list_dv[5],
                                   list_Dx[5], list_Dy[5], list_Dz[5],
                                   list_Mx[5], list_My[5], list_Mz[5])
        
        self.L6 = OperatorBlock_3D(int(2*list_dv[5]), list_dv[6],
                                   list_Dx[6], list_Dy[6], list_Dz[6],
                                   list_Mx[6], list_My[6], list_Mz[6])
        
        self.L7 = OperatorBlock_3D(int(2*list_dv[6]), list_dv[7],
                                   list_Dx[7], list_Dy[7], list_Dz[7],
                                   list_Mx[7], list_My[7], list_Mz[7])

        assert list_dv[7] == self.width

        # projection operator
        # I attribute names to each component but it has no effect. You can permute channels as wanted
        # Default: E-W component
        self.Q0E = nn.Linear(self.width, 4*self.width)
        self.Q1E = nn.Linear(4*self.width, 1)

        # N-S component
        self.Q0N = nn.Linear(self.width, 4*self.width)
        self.Q1N = nn.Linear(4*self.width, 1)

        # Z component
        self.Q0Z = nn.Linear(self.width, 4*self.width)
        self.Q1Z = nn.Linear(4*self.width, 1)
        

    def forward(self, x):
        if self.verbose:
            print('x', x.shape)
        grid = self.get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)
        if self.verbose:
            print('x after concat', x.shape)
        # uplift
        x_P0 = self.P0(x)
        x_P0 = F.gelu(x_P0)

        x_P1 = self.P1(x_P0)
        x_P1 = F.gelu(x_P1)

        if self.verbose:
            print('v0 before permutation', x_P1.shape)
        
        x_P1 = x_P1.permute(0, 4, 1, 2, 3) # move channel to axis 1
        if self.verbose:
            print('v0 after permutation', x_P1.shape)
            
        self.padding = int(self.padding*0.1*x_P1.shape[-1])
        v0 = F.pad(x_P1, [0, self.padding, 0, 0, 0, 0], mode ='constant')
        if self.verbose:
            print('v0 after permutation and padding', v0.shape)
        
        # reduction path
        v1 = self.L0(v0)
        if self.verbose:
            print('v1', v1.shape)
            
        v2 = self.L1(v1)
        if self.verbose:
            print('v2', v2.shape)
        v3 = self.L2(v2)
        if self.verbose:
            print('v3', v3.shape)
        v4 = self.L3(v3)
        if self.verbose:
            print('v4', v4.shape)
        
        # augmentation path
        v5 = self.L4(v4)    
        if self.verbose:
            print('v5 before concat', v5.shape)
        # we concatenate along axis 1, however axis 4 (time) does not necessarily have the same size
        v3_interp = torch.nn.functional.interpolate(v3, 
                    size = (v5.shape[2], v5.shape[3], v5.shape[4]),
                    mode = 'trilinear', align_corners=True)
        v5 = torch.cat([v5, v3_interp], dim=1)
        if self.verbose:
            print('v5', v5.shape)
        
        v6 = self.L5(v5)
        if self.verbose:
            print('v6 before concat', v6.shape)
        v2_interp = torch.nn.functional.interpolate(v2, 
                    size = (v6.shape[2], v6.shape[3], v6.shape[4]),
                    mode = 'trilinear', align_corners=True)
        v6 = torch.cat([v6, v2_interp], dim=1)
        if self.verbose:
            print('v6', v6.shape)
        
        v7 = self.L6(v6)
        if self.verbose:
            print('v7 before concat', v7.shape)
        v1_interp = torch.nn.functional.interpolate(v1, 
                    size = (v7.shape[2], v7.shape[3], v7.shape[4]),
                    mode = 'trilinear', align_corners=True)
        v7 = torch.cat([v7, v1_interp], dim=1)
        if self.verbose:
            print('v7', v7.shape)
        
        v8 = self.L7(v7)
        if self.verbose:
            print('v8', v8.shape)

        if self.padding!=0:
            v8 = v8[..., :-4*self.padding]

        v8 = v8.permute(0, 2, 3, 4, 1)
        if self.verbose:
            print('v8 after permute', v8.shape)
        
        # projection
        x_Q0E = self.Q0E(v8)
        x_Q0E = F.gelu(x_Q0E)
        x_outE = self.Q1E(x_Q0E)
        if self.verbose:
            print('x_outE', x_outE.shape, '\n\n')

        x_Q0N = self.Q0N(v8)
        x_Q0N = F.gelu(x_Q0N)
        x_outN = self.Q1N(x_Q0N)  

        x_Q0Z = self.Q0Z(v8)
        x_Q0Z = F.gelu(x_Q0Z)
        x_outZ = self.Q1E(x_Q0Z)
        
        return x_outE, x_outZ, x_outN
        
    
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        
        return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy), gridz), dim=-1).to(device)
