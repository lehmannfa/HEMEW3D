import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functions.fourier_blocks import *


class FNO_3D(nn.Module):
    def __init__(self, in_width, width, S, list_dv, list_D1, list_D2, list_D3, list_M1, list_M2, list_M3, padding = 3, verbose=False):
        super(FNO_3D, self).__init__()

        self.in_width = in_width # input channel
        self.width = width # dimension of lifted function v0
        self.S = S
        self.padding = padding
        self.verbose = verbose

        # uplift operator : D1 x D2 x D3 x in_width --> D1 x D2 x D3 x width
        self.P0 = nn.Linear(self.in_width, self.width//2)
        self.P1 = nn.Linear(self.width//2, self.width)

        # reduction path
        self.L0 = OperatorBlock_3D(self.width, list_dv[0],  
                                   list_D1[0], list_D2[0], list_D3[0],
                                   list_M1[0], list_M2[0], list_M3[0])

        self.L1 = OperatorBlock_3D(list_dv[0], list_dv[1], 
                                   list_D1[1], list_D2[1], list_D3[1],
                                   list_M1[1], list_M2[1], list_M3[1])

        self.L2 = OperatorBlock_3D(list_dv[1], list_dv[2], 
                                   list_D1[2], list_D2[2], list_D3[2],
                                   list_M1[2], list_M2[2], list_M3[2])

        self.L3 = OperatorBlock_3D(list_dv[2], list_dv[3], 
                                   list_D1[3], list_D2[3], list_D3[3],
                                   list_M1[3], list_M2[3], list_M3[3])

        # projection operator
        # East-West component
        self.Q0E = nn.Linear(self.width, 4*self.width)
        self.Q1E = nn.Linear(4*self.width, 1)

        # North-South component
        self.Q0N = nn.Linear(self.width, 4*self.width)
        self.Q1N = nn.Linear(4*self.width, 1)
            
        # Vertical component
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

        vf = v4 # final v
        if self.padding!=0:
            vf = vf[..., :-4*self.padding]

        vf = vf.permute(0, 2, 3, 4, 1)
        if self.verbose:
            print('vf after permute', vf.shape)
        
        # projection
        x_Q0E = self.Q0E(vf)
        x_Q0E = F.gelu(x_Q0E)
        x_outE = self.Q1E(x_Q0E)
        if self.verbose:
            print('x_outE', x_outE.shape, '\n\n')

        x_Q0N = self.Q0N(vf)
        x_Q0N = F.gelu(x_Q0N)
        x_outN = self.Q1N(x_Q0N)

        x_Q0Z = self.Q0Z(vf)
        x_Q0Z = F.gelu(x_Q0Z)
        x_outZ = self.Q1E(x_Q0Z)
        return x_outE, x_outN, x_outZ
        
    
    
    def get_grid(self, shape, device):
        batchsize, size1, size2, size3 = shape[0], shape[1], shape[2], shape[3]
        grid1 = torch.tensor(np.linspace(0, 2*np.pi, size1), dtype=torch.float)
        grid1 = grid1.reshape(1, size1, 1, 1, 1).repeat([batchsize, 1, size2, size3, 1])
        
        grid2 = torch.tensor(np.linspace(0, 2*np.pi, size2), dtype=torch.float)
        grid2 = grid2.reshape(1, 1, size2, 1, 1).repeat([batchsize, size1, 1, size3, 1])
        
        grid3 = torch.tensor(np.linspace(0, 1, size3), dtype=torch.float)
        grid3 = grid3.reshape(1, 1, 1, size3, 1).repeat([batchsize, size1, size2, 1, 1])
        
        return torch.cat((torch.sin(grid1),torch.sin(grid2),torch.cos(grid1),torch.cos(grid2), grid3), dim=-1).to(device)


