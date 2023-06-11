import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator


def ComputeDsfromCp(Cp):
    ''' Compute the material density from the P wave velocity
    Cp : float or array in m/s

    return : float or array of same dimensions as Cp (kg/m^3)'''
    
    if (Cp<100).any():
        raise Exception(f'The P-wave velocity seems unphysical (min={Cp.min()}m/s')
    
    Ds = 1.6612*(Cp/1000)-0.4721*(Cp/1000)**2+0.0671*(Cp/1000)**3-0.0043*(Cp/1000)**4+0.000106*(Cp/1000)**5
    return Ds*1000


def ComputeQsfromCs(Cs):
    ''' Compute the S-wave attenuation factor from the S-wave velocity '''
    return Cs/10
    
    
def ComputeQpfromCpCs(Cp, Cs):
    ''' Compute the P-wave attenuation factor from the S-wave and P-wave velocities 
    Qp = max(Cp/20, Cs/5) '''
    return np.where(Cp/20 >= Cs/5, Cp/20, Cs/5)


def array2dataframe(mat, Xvec, Yvec, Zvec, column, order='xyz'):
    ''' Helper function to transform a 3D array into a Pandas dataframe with 4 columns
    mat : 3D array indexed by "order"
    Xvec, Yvec, Zvec : 1D array giving the axis coordinates
    order : string giving the reshape order (permutation of x, y, z)
    
    return : DataFrame with 4 columns ('x', 'y', 'z', and "column") where "column" contains the flattened values of mat '''
    
    Nx = Xvec.shape[0]
    Ny = Yvec.shape[0]
    Nz = Zvec.shape[0]
    
    # axis coordinates
    vecs = {'x' : Xvec,
            'y' : Yvec,
            'z' : Zvec}
    
    # axis numbers
    i = {'x' : order.find('x'),
         'y' : order.find('y'),
         'z' : order.find('z')
         }
    
    temp = pd.MultiIndex.from_product([vecs[order[0]], vecs[order[1]], vecs[order[2]]])
    df = pd.DataFrame({'x' : temp.get_level_values(i['x']),
                       'y' : temp.get_level_values(i['y']),
                       'z' : temp.get_level_values(i['z']),
                       column : mat.flatten()
                       })
    return df


def dataframe2array(df, column, order='xyz'):
    ''' df : DataFrame containing at least 4 columns : x, y, z and "column"
    column : string
    order : string giving the reshape order (permutation of x, y, z)
    
    return : 3D array containing the values of "column" reshape in the order given by "order" '''
    
    # axes coordinates
    vecs = {'x' : np.unique(df.x), # Xvec
            'y' : np.unique(df.y), # Yvec
            'z' : np.unique(df.z)} # Zvec
    
    # size of each axis
    N = {'x' : np.shape(vecs['x'])[0],
         'y' : np.shape(vecs['y'])[0],
         'z' : np.shape(vecs['z'])[0],
        }
    
    # order the columns to preserve the elements' order in the reshape
    mat = df.sort_values(by=[order[0], order[1], order[2]]).loc[:,column].values
    mat = mat.reshape(N[order[0]], N[order[1]], N[order[2]])
    return mat


def read_random_field(field, material, min_field=-np.inf, max_field = np.inf,
                      Xinterp=None, Yinterp=None, Zinterp=None, interpolation=True,
                      verbose=False, path_case='./'):

    ''' reads the material created by Random Field and returns a DataFrame
    field : string, name of the random property to read (usually, Vp, Vs, Density)
    material : int, index of the material
    min_field, max_field : floats, minimum and maximum values of the field to cap random values
    Xinterp, Yinterp, Zinterp : arrays of coordinates to optionnally interpolate the field values
    interpolation : boolean, whether to interpolate the values or use SEM grid

    return : DataFrame with columns x, y, z containing the points coordinates and a column "field" containing the values of the random field '''

    
    f = h5py.File(f'{path_case}mat/h5/Mat_{material}_{field}.h5')
    field_values=f['samples'][:]

    # name conventions
    if field=='Density':
        field_name='Rho'
    else:
        field_name=field
    
    # cap values
    field_values=np.where(field_values<min_field, min_field, field_values)
    field_values=np.where(field_values>max_field, max_field, field_values)
    
    xMinGlob = f.attrs['xMinGlob']
    xMaxGlob = f.attrs['xMaxGlob']
    xStep = f.attrs['xStep']

    # axis coordinates
    Xvec=np.arange(xMinGlob[0], xMaxGlob[0]+xStep[0], xStep[0])
    Yvec=np.arange(xMinGlob[1], xMaxGlob[1]+xStep[1], xStep[1])
    Zvec=np.arange(xMinGlob[2], xMaxGlob[2]+xStep[2], xStep[2])

    if interpolation: # from (Xvec, Yvec, Zvec) to (Xinterp, Yinterp, Zinterp)
        # if interpolation data have been given only as spatial steps, create the interpolation array
        if not isinstance(Xinterp, (list, np.ndarray)):
            Xinterp = np.linspace(Xvec[0], Xvec[-1], int((Xvec[-1]-Xvec[0])/Xinterp)+1)
        if not isinstance(Yinterp, (list, np.ndarray)):
            Yinterp = np.linspace(Yvec[0], Yvec[-1], int((Yvec[-1]-Yvec[0])/Yinterp)+1)
        if not isinstance(Zinterp, (list, np.ndarray)):
            Zinterp = np.linspace(Zvec[0], Zvec[-1], int((Zvec[-1]-Zvec[0])/Zinterp)+1)
        Zinterp = Zinterp[(Zinterp>=Zvec[0])&(Zinterp<=Zvec[-1])]

        interp = RegularGridInterpolator((Zvec, Yvec, Xvec), field_values)
        
        temp=pd.MultiIndex.from_product([Xinterp, Yinterp, Zinterp])
        df=pd.DataFrame({'x':temp.get_level_values(0),
                         'y':temp.get_level_values(1),
                         'z':temp.get_level_values(2)
                        })
        df.loc[:,field]=interp(df.loc[:,['z','y','x']])
        
    else:
        df = helparrays.array2dataframe(field_values, Xvec, Yvec, Zvec, 
                           column=field, order='zyx')
    if verbose:
        print(df)

    return df


def plot3Dcube(mat, Xvec, Yvec, Zvec, mask=None, cmap='viridis',
               title='', figsize=(12, 6),
               vmin=None, vmax=None,
               xlabel='x', ylabel='y', zlabel='z', vlabel='v'):
    ''' 
    mat : (Nx, Ny, Nz) 3D array of scalars indexed by x, y, z 
    Xvec : (Nx,) coordinates of the x axis
    Yvec : (Ny,) coordinates of the y axis
    Zvec : (Nz,) coordinates of the z axis
    mask : (Nx, Ny, Nz) 3D array containing ones for pixel to draw and 0 for pixels to hide
    '''

    if Xvec[1]<=Xvec[0]:
        raise Exception('Xvec must be in ascending order')
    if Yvec[1]<=Yvec[0]:
        raise Exception('Yvec must be in ascending order')
    if Zvec[1]<=Zvec[0]:
        raise Exception('Zvec must be in ascending order')

    if Xvec.shape[0]!=mat.shape[0]:
        raise Exception(f'The shape of Xvec ({Xvec.shape}) does not match the first axis of mat ({mat.shape[0]})')
    if Yvec.shape[0]!=mat.shape[1]:
        raise Exception(f'The shape of Yvec ({Yvec.shape}) does not match the second axis of mat ({mat.shape[1]})')
    if Zvec.shape[0]!=mat.shape[2]:
        raise Exception(f'The shape of Zvec ({Zvec.shape}) does not match the third axis of mat ({mat.shape[2]})')

    # we extend each coordinates by one step to include the boundaries of each axis
    Xvec2 = np.concatenate([Xvec, [Xvec[-1]+Xvec[1]-Xvec[0]]])
    Yvec2 = np.concatenate([Yvec, [Yvec[-1]+Yvec[1]-Yvec[0]]])
    Zvec2 = np.concatenate([Zvec, [Zvec[-1]+Zvec[1]-Zvec[0]]])
    Xmat, Ymat, Zmat = np.meshgrid(Xvec2, Yvec2, Zvec2, indexing='ij')

    # create the color array from normalized data
    if vmin is None:
        vmin = mat.min()
    if vmax is None:
        vmax = mat.max()

    cmap = cm.get_cmap(cmap, 256)
    cmap_array = cmap((mat-vmin)/(vmax-vmin)) # R G B alpha
    colors = np.zeros(mat.shape + (4,))
    colors[..., 0] = cmap_array[:,:,:,0]
    colors[..., 1] = cmap_array[:,:,:,1]
    colors[..., 2] = cmap_array[:,:,:,2]
    colors[..., 3] = 1

    # mask to show only some selected cells
    if mask is None:
        filled = np.ones_like(mat)
    else:
        assert mask.shape == mat.shape
        filled = mask

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([Xvec2[0],Xvec2[-1]])
    ax.set_ylim([Yvec2[0],Yvec2[-1]])
    ax.set_zlim([Zvec2[0],Zvec2[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.voxels(Xmat, Ymat, Zmat,
              filled,
              facecolors=colors,
              edgecolors = colors,
              linewidth=0.5)    
    
    m = cm.ScalarMappable(cmap=cmap, 
                          norm=Normalize(vmin=vmin, vmax=vmax)
                         )
    m.set_array([])
    if title!='':
        ax.set_title(title)
    plt.colorbar(m, shrink=0.7, pad=0.1, label=vlabel)
    plt.tight_layout()
    
    plt.show()