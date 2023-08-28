"""
original authors: Zongyi Li and Daniel Zhengyu Huang https://github.com/alasdairtran/fourierflow/tree/main
modified by: Fanny Lehmann
"""

import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.functions.utils_ffno import FeedForward, WNLinear


class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, D1, D2, D3, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        x = self.forward_fourier(x)

        x = x.permute(0, 2, 3, 4, 1)
        b = self.backcast_ff(x)
        b = b.permute(0, 4, 1, 2, 3)
        if self.use_fork:
            f = self.forecast_ff(x)
            f = f.permute(0, 4, 1, 2, 3)
        else:
            f = None
        return b, f

    def forward_fourier(self, x):
        B, I, S1, S2, S3 = x.shape

        # Dimension Z
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, I, self.D1, self.D2, self.D3 // 2 + 1)
        
        out_ft[:, :, :S1, :S2, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))

        xz = torch.fft.irfft(out_ft, n=self.D3, dim=-1, norm='ortho')

        # Dimension Y
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, self.D1, self.D2 // 2 + 1, self.D3)
        
        out_ft[:, :, :S1, :self.modes_y, :S3] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))

        xy = torch.fft.irfft(out_ft, n=self.D2, dim=-2, norm='ortho')
        
        # Dimension X
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, self.D1 // 2 + 1, self.D2, self.D3)
        
        out_ft[:, :, :self.modes_x, :S2, :S3] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=self.D1, dim=-3, norm='ortho')
        
        # Combining Dimensions
        x = xx + xy + xz

        return x
    
    
class ModifyDimensions3d(nn.Module):
    ''' Modify the shape of input from batch, S1, S2, S3, c to batch, dim1, dim2, dim3, c '''
    def __init__(self, dim1, dim2, dim3):
        super(ModifyDimensions3d,self).__init__()
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x):
        ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm='forward')
        d1 = min(ft.shape[2]//2, self.dim1//2)
        d2 = min(ft.shape[3]//2, self.dim2//2)
        d3 = max(min(ft.shape[4]//2, self.dim3//2), 1)
        ft_u = torch.zeros((ft.shape[0], ft.shape[1], self.dim1, self.dim2, self.dim3//2+1), dtype=torch.cfloat, device=ft.device)
        ft_u[:, :, :d1, :d2, :d3] = ft[:, :, :d1, :d2, :d3]
        ft_u[:, :, -d1:, :d2, :d3] = ft[:, :, -d1:, :d2, :d3]
        ft_u[:, :, :d1, -d2:, :d3] = ft[:, :, :d1, -d2:, :d3]
        ft_u[:, :, -d1:, -d2:, :d3] = ft[:, :, -d1:, -d2:, :d3]
        x_out = torch.fft.irfftn(ft_u, s=(self.dim1, self.dim2, self.dim3), norm='forward')

        return x_out


class FFNO_3D(nn.Module):
    def __init__(self, list_D1, list_D2, list_D3, list_M1, list_M2, list_M3, width, input_dim, output_dim,
                 n_layers=4, factor=4, ff_weight_norm=True, n_ff_layers=2, layer_norm=False, padding=8):
        ''' I (Fanny) impose share_weight = False since we need different modes in different layers '''
        super().__init__()
        self.padding = padding # pad the domain if input is non-periodic
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_weight = None
        self.n_layers = n_layers
        
        self.P = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        
        self.spectral_layers = nn.ModuleList([])
        self.modif_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.spectral_layers.append(FactorizedSpectralConv3d(in_dim=width, out_dim=width, 
                                                                 D1=list_D1[i], D2=list_D2[i], D3=list_D3[i],
                                           modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                           forecast_ff=None, backcast_ff=None, fourier_weight=None, factor=factor, 
                                           ff_weight_norm=ff_weight_norm,
                                           n_ff_layers=n_ff_layers, layer_norm=layer_norm, use_fork=False, dropout=0.0))
            self.modif_layers.append(ModifyDimensions3d(list_D1[i], list_D2[i], list_D3[i]))

        self.QE = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))
        
        self.QN = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))
        
        self.QZ = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.P(x)

        x = x.permute(0, 4, 1, 2, 3)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
            
        for i in range(self.n_layers):
            layer_s = self.spectral_layers[i]
            layer_m = self.modif_layers[i]
            b, _ = layer_s(x)
            x = layer_m(x) + b      

        if self.padding != 0:
            b = b[..., :, :-self.padding, :-self.padding, :-self.padding]
        
        xf = b
        xf = xf.permute(0, 2, 3, 4, 1)
        
        uE = self.QE(xf)
        uN = self.QN(xf)
        uZ = self.QZ(xf)

        return uE, uN, uZ

    def get_grid(self, shape, device):
        batchsize, size1, size2, size3 = shape[0], shape[1], shape[2], shape[3]
        grid1 = torch.tensor(np.linspace(0, 1, size1), dtype=torch.float)
        grid1 = grid1.reshape(1, size1, 1, 1, 1).repeat(
            [batchsize, 1, size2, size3, 1])
        grid2 = torch.tensor(np.linspace(0, 1, size2), dtype=torch.float)
        grid2 = grid2.reshape(1, 1, size2, 1, 1).repeat(
            [batchsize, size1, 1, size3, 1])
        grid3 = torch.tensor(np.linspace(0, 1, size3), dtype=torch.float)
        grid3 = grid3.reshape(1, 1, 1, size3, 1).repeat(
            [batchsize, size1, size2, 1, 1])
        return torch.cat((grid1, grid2, grid3), dim=-1).to(device)