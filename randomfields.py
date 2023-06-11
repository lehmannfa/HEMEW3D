import numpy as np
import pandas as pd
import datetime
import time
import h5py
import os, shutil
from scipy.interpolate import RegularGridInterpolator
#import functions.helpers.arrays3d as helparrays
#import functions.sem.geology as geol


### TEXT HELPER FUNCTIONS ###
def my_find(full_str,str1,indices=False):
    ''' Returns the substring of full_str starting with str1
    str1 must not be contained in a comment (otherwise, it raises an error
    If indices=True, also returns the first index of this substring
    /!\ We do not expect str1 to be found in full_str
    and we do not want to throw an error if it is not found '''
    
    i1=full_str.find(str1)
    if i1==-1:
        if indices:
            return '',-1
        else:
            return ''
        
    index_comment=full_str[:i1][::-1].find('#')
    index_newline=full_str[:i1][::-1].find('\n')
    #print(i1,index_comment,index_newline)

    # if we found a comment before str1 (index_comment!=-1)
    # and there was no newline before the comment (index_comment<index_newline)
    # we look for the next occurence of str1
    while (index_newline==-1 & index_comment!=-1) | (index_comment!=-1 and index_comment<index_newline):
        i1=full_str.find(str1,i1+1)
        if i1!=-1:
            index_comment=full_str[:i1][::-1].find('#')
            index_newline=full_str[:i1][::-1].find('\n')
        # If there is no occurence of str1 out of a comment, return an empty string
        elif indices:
            return '',-1
        else:
            return ''

    if indices:
        return full_str[i1:],i1
    else:
        return full_str[i1:]


def find_substr_between(full_str,str1,str2,indices=False):
    ''' Returns the substring contained between str1 and str2 (not contained in the output)'''
    i1=my_find(full_str,str1,indices=True)[1]
    i2=my_find(full_str[i1+len(str1):],str2,indices=True)[1]
    if indices:
        return full_str[i1+len(str1):i1+len(str1)+i2],i1,i1+len(str1)+i2+len(str2)
    else:
        return full_str[i1+len(str1):i1+len(str1)+i2]

    
    
### RANDOM FIELDS FUNCTIONS ###


def write_RF_files(corr_lens, coef_var_, L, layers, correlation_mode=3,
                   properties = ['Vp','Vs','Rho'],
                   marginal = 1, seed = -1, path_case=''):
    ''' writes the file "random.spec" that defines the properties of the random field

    corr_lens : list containing corr_length_x, corr_length_y, corr_length_z (in meters)
    coef_var : float or list of floats, coefficient of variation (standard deviation / mean)
    L : list of random layers (by their index)
    layers : dataframe with one layer per row and the layer specification in columns
    correlation mode : 1= Gaussian, 2=Exponential, 3=Von Karman
    marginal : 1 = Gaussian, 2 = LogNormal
    seed : -1 for determination by the clock time, otherwise fixed to any integer
    path_case : string specifying the folder where to write files

    return : the list of layers that were assigned to random
    '''
    
    nb_random_layers = L.shape[0]
    assert 0 <= L.shape[0] < layers.shape[0], f"the number of random layers should be between 0 and {nb_layers}"

    if seed==-1:
        # initial time to avoid large seed integers
        t0=datetime.datetime(2023,5,1)
        t0=1e5*int(time.mktime(t0.timetuple()))
    
    # if random fields parameters are the same for all layers, transform to list
    def float_to_list(param):
        if not isinstance(param, (list, np.ndarray)):
            return [param]*nb_random_layers
        else:
            return param

    coef_var = float_to_list(coef_var_)
    corr_length_x = float_to_list(corr_lens[0])
    corr_length_y = float_to_list(corr_lens[1])
    corr_length_z = float_to_list(corr_lens[2])
            
    with open(path_case+'random.spec', 'w') as f:
        f.write('# Number of random materials\n')
        f.write(str(L.shape[0])+'\n')

        if seed == -1:
            seed = int(1e5*time.time() - t0)
            seed = int(str(seed)[3:])

        for l, cv, cx, cy, cz in zip(L, coef_var, corr_length_x, corr_length_y, corr_length_z):                
            seed +=1
            
            f.write('# Material number\n')
            f.write(str(l)+'\n')
            f.write('# Vp_Vs_Rho_flag: Calculate Lambda, Kappa, Mu and Density average from Vp Vs Density (0 = false, 1 = true) \n')
            f.write('# If previous option = 1, give Vp, Vs and Density on the same line \n')
            f.write(str(0) + '\t' + str(layers.loc[l,'Vp']) + '\t' + str(layers.loc[l,'Vs']) + '\t' + str(layers.loc[l,'Rho'])+'\n')
            f.write('# Number of properties\n')
            f.write(str(len(properties)) + '\n')
            f.write('# Property_Name\tavg\tcorrMod\tcorrL_x\tcorrL_y\tcorrL_z\tmargiF\tCV\tseedStart\n')
            
            if 'Rho' in properties:
                f.write('"Density"\t' + str(layers.loc[l,'Rho']) + '\t' + str(correlation_mode) + '\t')
                f.write(str(cx) + '\t' + str(cy) + '\t' + str(cz) + '\t')
                f.write(str(marginal) + '\t' + str(cv) + '\t' + str(seed) +'\n')

            if 'Vp' in properties:
                f.write('"Vp"\t' + str(layers.loc[l,'Vp']) + '\t' + str(correlation_mode) + '\t')
                f.write(str(cx) + '\t' + str(cy) + '\t' + str(cz) + '\t')
                f.write(str(marginal) + '\t' + str(cv) + '\t' + str(seed) +'\n')

            if 'Vs' in properties:
                f.write('"Vs"\t' + str(layers.loc[l,'Vs']) + '\t' + str(correlation_mode) + '\t')
                f.write(str(cx) + '\t' + str(cy) + '\t' + str(cz) + '\t')
                f.write(str(marginal) + '\t' + str(cv) + '\t' + str(seed) +'\n')
    return L


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


def create_deterministic_layer(field, material, Xlayer, Ylayer, Zlayer, layers):
    ''' field : string, name of the random property to read (usually, Vp, Vs, Density)
    material : int, index of the material
    Xlayer, Ylayer, Zlayer : arrays, coordinates of the axis indexing the layer "material"
    layers : dataframe with one layer per row and the layer specification in columns

    return : a DataFrame with columns x, y, z, and "field" containing the value assigned in the reference DataFrame "layers"
    '''

    # DataFrame of coordinates
    temp=pd.MultiIndex.from_product([Xlayer, Ylayer, Zlayer])
    df=pd.DataFrame({'x':temp.get_level_values(0),
                     'y':temp.get_level_values(1),
                     'z':temp.get_level_values(2)
                    })

    # give each cell the same value as determined in "layers"
    if field=='Density': # name convention 'Rho'='Density'
        df.loc[:,field]=layers.loc[material, 'Rho']
    else:
        df.loc[:,field]=layers.loc[material, field]
    return df


def assemble_layers(Nx, Ny, Nz, lines, layers, L, max_deviation = 0.5, interpolation=True, verbose=False, path_case='./'):
    ''' Nx, Ny, Nz : int, number of steps in each direction (for the whole domain)
    lines : list containing the domain properties xmin,xmax,xstep,ymin,ymax,ystep,zmax,nb_layers
    layers : dataframe with one layer per row and the layer specification in columns
    L : index of layers with random properties, the other are determinist
    max_deviation : maximum authorized difference with the mean value

    return : DataFrame with columns x, y, z, Vp, Vs, Rho
    '''

    if interpolation==False and len(L)<layers.shape[0]:
        raise Exception('I cannot determine the coordinates if there are deterministic layers')
    
    [xmin, xmax, xstep, ymin, ymax, ystep, zmax, nb_layers] = lines
    
    Xnew = np.linspace(xmin,xmax,Nx)
    Ynew = np.linspace(ymin,ymax,Ny)
    upper_layersZ = list(np.arange(-4000,zmax+1,200)) # better resolution in the uppermost layers (step=200m above -4000m) 
    Znew = np.array(list(np.linspace(zmax-layers.thickness.sum(),-4000,Nz-len(upper_layersZ)+1)[:-1])+upper_layersZ)
    # concatenate the coarser grid for the bottom layers with the finer grid

    # initiate empty DataFrames
    Vs = pd.DataFrame(columns=['x','y','z','Vs'])
    Vp = pd.DataFrame(columns=['x','y','z','Vp'])
    Rho = pd.DataFrame(columns=['x','y','z','Density'])

    for l in range(layers.shape[0]):
        layer_top = zmax-layers.iloc[:l].thickness.sum()
        layer_bottom = layer_top - layers.loc[l, 'thickness']
        # Z coordinates inside the layer
        if l==0:
            Zlayer = Znew[(Znew>=layer_bottom)&(Znew<=layer_top)]
        else:
            Zlayer = Znew[(Znew>=layer_bottom)&(Znew<layer_top)]

        if Zlayer.shape[0]>0: # if the layer is not too small and contains at least one point
            if l in L: # random layer
                if verbose:
                    print(f'layer {l} is random')
                Vs=Vs.append(read_random_field('Vs', l, layers.loc[l,'Vs'], max_deviation=max_deviation,
                                               Xinterp=Xnew, Yinterp=Ynew, Zinterp=Zlayer, interpolation=interpolation, verbose=verbose, path_case=path_case))
                Vp=Vp.append(read_random_field('Vp', l, layers.loc[l,'Vp'], max_deviation=max_deviation,
                                               Xinterp=Xnew, Yinterp=Ynew, Zinterp=Zlayer, interpolation=interpolation, verbose=verbose, path_case=path_case))
                Rho=Rho.append(read_random_field('Density', l, layers.loc[l,'Rho'], max_deviation=max_deviation,
                                               Xinterp=Xnew, Yinterp=Ynew, Zinterp=Zlayer, interpolation=interpolation, verbose=verbose, path_case=path_case))
                
            else: # deterministic layer
                if verbose:
                    print(f'layer {l} is determinist')
                Vs=Vs.append(create_deterministic_layer('Vs', l, Xnew, Ynew, Zlayer, layers))
                Vp=Vp.append(create_deterministic_layer('Vp', l, Xnew, Ynew, Zlayer, layers))
                Rho=Rho.append(create_deterministic_layer('Density', l, Xnew, Ynew, Zlayer, layers))

    geology=Vp.copy()
    geology.loc[:,'Vs']=Vs.Vs
    geology.loc[:,'Rho']=Rho.Density
    
    return geology.astype(float, copy=False)


def generate_random_geology(run_name, domain, PMLs, layers, corr_lens, coef_var, nb_random_layers,
                            modify_layers=True, max_deviation = 0.5, correlation_mode=3, L=None,
                            Nx=32, Ny=32, Nz=96, interpolation=True, verbose=False,
                            path_save='db_rand', path_case='./temp/',
                            PATH_SEM_BUILD='/cea/home/b4/lehmannf/SEM3D/SEM_BUILD',
                            PATH_SEM_RF='/cea/home/b4/lehmannf/SEM3D/SEM_RF'):
    '''
    run_name : string used to identify the simulation
    domain : list containing [xmin,xmax,xstep,ymin,ymax,ystep,zmax,nb_layers]
    PMLs : list containing [PML_sides, PML_top, PML_bottom, PML_layers, mesh_type]
    layers : dataframe with one layer per row and the layer specification in columns
    corr_lens : list [corr_length_x, corr_length_y, corr_length_z] in meters
    coef_var : float, coefficient of variation for each random layer
    nb_random_layers : int, number of layers to make random
    modify_layers : bool, whether to modify the initial layers thickness
    max_deviation : float between 0 and 1 to cap extreme values of the random fields
    correlation_mode : 1, 2, 3
    L : list of layers index to make random, otherwise randomly selected
    Nx, Ny, Nz : int, number of steps in each direction for the final interpolated domain

    return : save a 3D array in format .npy in the folder "path_save"
    returns the name of this saved file
    '''

    # create the folder to store temporary files needed by SEM
    if not path_case.replace('.','').replace('/','') in os.listdir():
        os.mkdir(path_case.replace('.','').replace('/',''))
    
    # randomly alter the thickness of some randomly selected layers
    if modify_layers:
        layers=geol.modify_thickness_layers(layers, verbose=verbose)

    geol.write_MESH_files(run_name, domain, PMLs, layers, path_case=path_case)
    
    # generate the mesh
    current_dir = os.getcwd()
    os.chdir(path_case)

    output = os.system(PATH_SEM_BUILD + 'MESH/mesher < mesh.input')
    if output!=0:
        os.chdir(current_dir)
        raise Exception('The mesher could not run "'+run_name+'"')
    os.chdir(current_dir)
    
    # generate random fields
    random_layers = write_RF_files(corr_lens, coef_var, nb_random_layers, layers,
                                   correlation_mode=correlation_mode, L=L, path_case=path_case)
    current_dir = os.getcwd()
    os.chdir(path_case)
    if 'mat' in os.listdir(): # remove old files
        shutil.rmtree('mat')

    output = os.system(PATH_SEM_RF + 'randomField.exe')
    if output!=0:
        os.chdir(current_dir)
        raise Exception('RandomField could not run "'+run_name+'"')
    os.chdir(current_dir)
        
    geology = assemble_layers(Nx, Ny, Nz, domain, layers,
                                 random_layers, max_deviation = max_deviation, interpolation=interpolation,
                                 verbose = verbose, path_case=path_case)    
    
    if verbose:
        print(geology.dtypes)
        print(geology.head())
    
    nu=0.5*(1-1/((geology.Vp/geology.Vs)**2-1))
    if nu.min()>0 and nu.max()<0.5: # check that the Poisson coefficient if physically possible
        if modify_layers:
            filename=f'{path_save}configrand_corr{int(corr_lens[0])}-{int(corr_lens[1])}-{int(corr_lens[2])}_coefvar{int(coef_var*1e3)}e-3_{run_name}.npy'
            np.save(filename, geology)
        else:
            filename=f'{path_save}config_corr{int(corr_lens[0])}-{int(corr_lens[1])}-{int(corr_lens[2])}_coefvar{int(coef_var*1e3)}e-3_{run_name}.npy'
            np.save(filename, geology)
    else:
        print(f'Poisson coefficient not physical for corr{int(corr_lens[0])}-{int(corr_lens[1])}-{int(corr_lens[2])}_coefvar{int(coef_var*1e3)}e-3_{run_name}')
        
    return filename
