''' This code is largely inspired by https://github.com/ashiq24/UNO '''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3, verbose=False):
        super(SpectralConv3d_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Output grid size along x (or 1st dimension of output domain) 
        dim2 = Output grid size along y ( or 2nd dimension of output domain)
        dim3 = Output grid size along z or time t ( or 3rd dimension of output domain)
        modes1, modes2, modes3 = Number of fourier modes to consider for the integral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2)
                                      modes2 <= min( dim2/2, input_dim2/2)
                                      modes3 <= min( dim3//2+1, input_dim3//2+1)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domain dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim

        # Check that the number of modes is compatible with the output dimension (we do not test for the input dimensions since it would involver adding this check inside each forward)
        assert modes1 <= dim1//2, f"received modes1={modes1} and dim1={dim1}"
        assert modes2 <= dim2//2, f"received modes2={modes2} and dim2={dim2}"
        assert modes3 <= dim3//2+1, f"received modes3={modes3} and dim3={dim3}"

        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3 
        
        self.verbose = verbose

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1_real = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        self.weights1_imag = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        self.weights2_real = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        self.weights2_imag = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        
        self.weights3_real = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        self.weights3_imag = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        
        self.weights4_real = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))
        self.weights4_imag = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.float32))

        
    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1, dim2, dim3)
        """

        if self.verbose:
            print('SPECTRAL CONV')
            print('  dim1', self.dim1, 'dim2', self.dim2, 'dim3', self.dim3)
            print('  modes1', self.modes1, 'modes2', self.modes2, 'modes3', self.modes3)
            print('  x', x.shape)

        weights1 = self.weights1_real + 1j*self.weights1_imag
        weights2 = self.weights2_real +	1j*self.weights2_imag
        weights3 = self.weights3_real +	1j*self.weights3_imag
        weights4 = self.weights4_real +	1j*self.weights4_imag
            
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm = 'forward')
        if self.verbose:
            print('  x_ft', x_ft.shape)
            print('  weights', weights1.shape)

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm = 'forward')

        if self.verbose:
            print('  product x_ft and weights', out_ft.shape)
            print('  output spectral conv', x.shape)
        return x



class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, verbose=False):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)
        self.verbose = verbose

    def forward(self, x):
        x_out = self.conv(x)
        if self.verbose:
            print('POINTWISE OP')
            print('  x', x.shape)
            print('  x conv', x_out.shape)

        ft = torch.fft.rfftn(x_out, dim=[-3,-2,-1], norm='forward')
        d1 = min(ft.shape[2]//2, self.dim1//2)
        d2 = min(ft.shape[3]//2, self.dim2//2)
        d3 = max(min(ft.shape[4]//2, self.dim3//2), 1)
        ft_u = torch.zeros((ft.shape[0], ft.shape[1], self.dim1, self.dim2, self.dim3//2+1), dtype=torch.cfloat, device=ft.device)
        ft_u[:, :, :d1, :d2, :d3] = ft[:, :, :d1, :d2, :d3]
        ft_u[:, :, -d1:, :d2, :d3] = ft[:, :, -d1:, :d2, :d3]
        ft_u[:, :, :d1, -d2:, :d3] = ft[:, :, :d1, -d2:, :d3]
        ft_u[:, :, -d1:, -d2:, :d3] = ft[:, :, -d1:, -d2:, :d3]
        x_out = torch.fft.irfftn(ft_u, s=(self.dim1, self.dim2, self.dim3), norm='forward')

        if self.verbose:
            print('  x_ft', ft.shape)
            print(f'  select Fourier modes lower than {d1}, {d2}, {d3}')
            print('  FFT-1 (x_ft)', x_out.shape)
            x_out_before_interp = x_out.clone()

        # is theoretically and pratically useless
        #x_out = torch.nn.functional.interpolate(x_out, size = (self.dim1, self.dim2, self.dim3), mode = 'trilinear', align_corners=True)
        if self.verbose:
            print('  after interpolation, output', x_out.shape)
            print('  difference created by interpolation', torch.norm(x_out - x_out_before_interp))
        return x_out
    

class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3, Normalize = False, Non_Lin = True, verbose=False):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3, verbose=verbose)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1, dim2, dim3, verbose=verbose)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim),affine=True)


    def forward(self, x):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x)
        x2_out = self.w(x)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
