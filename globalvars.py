import numpy as np

DX = DY = DZ = 300 # in meters, spatial resolution of the mesh
FMAX = 5 # in Hz, maximum frequency to propagate
NGLL = 7 # number of quadrature points (Gauss-Lobato-Legendre)
VS_MIN = DX * FMAX * 5/NGLL # minimum velocity that can be propagated 
VS_MAX = 4500 # (m/s), S-wave velocity in the mantle
CV_MEAN = 0.2 # mean coefficient of variation for random fields
VS_RANGE = (VS_MIN/(1-2*CV_MEAN), VS_MAX/(1+2*CV_MEAN)) # boundary for S-wave velocities before adding random fields
DEPTH_TOTAL, DEPTH_BOTTOM = 32 * DX, 6 * DX # bottom layer is the same for all profiles to guarantee consistent source behaviour                                                                 
XMIN, XMAX = 0, 32 * DX
YMIN, YMAX = 0, 32 * DX                               
ZMAX = 0