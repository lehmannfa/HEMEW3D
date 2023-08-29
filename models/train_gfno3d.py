import os, sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import h5py
import argparse

from models.functions.helpers_deeplearning import get_device, get_batch_size, loss_criterion, RunningAverage, EarlyStopper
from models.functions.gfno_model import GFNO_3D
from models.functions.data_loader import GeologyTracesDataset


parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@S_in', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('@S_in_z', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('@S_out', type=int, default=32, help="Size of the spatial output grid")
parser.add_argument('@T_out', type=int, default=320, help="Number of time steps")
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of validation samples")
parser.add_argument('@batch_size', type=int, default=16, help = 'batch size')
parser.add_argument('@epochs', type=int, default=150, help = 'Number of epochs')
parser.add_argument('@dv', type=int, default=11, help = "Number of channels after the liting")
parser.add_argument('@list_D', type=int, nargs='+', default = [32, 32, 32, 32], help = "Dimensions along the 1st and 2nd dimension after each block")
parser.add_argument('@list_D3', type=int, nargs='+', default = [64, 128, 256, 320], help = "Dimensions along the 3rd dimension after each block")
parser.add_argument('@list_M', type=int, nargs='+', default = [8, 8, 8, 8], help = "Number of modes along the 1st and 2nd dimension after each block")
parser.add_argument('@list_M3', type=int, nargs='+', default = [8, 8, 8, 8], help = "Number of modes along the 3rd dimension after each block")
parser.add_argument('@padding', type=int, default=0, help = "Number of pixels for padding on each side of x and y")
parser.add_argument('@learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('@loss_weights', type=float, nargs='+', default = [1.0, 0.0], help = "Weight of L1 loss, L2 loss")
parser.add_argument('@dir_data_train', type=str, nargs='+', default=['inputs3D_S32_Z32_T320_fmax5_train'], help="Name of folders with training data")
parser.add_argument('@dir_data_val', type=str, nargs='+', default=['inputs3D_S32_Z32_T320_fmax5_val'], help="Name of folders with training data")
parser.add_argument('@additional_name', type=str, default="", help="string to add to the configuration name for saved outputs")
options = parser.parse_args().__dict__


path_data = './inputs/'
batch_size = options['batch_size']
Ntrain = options['Ntrain']
Nval = options['Nval']
learning_rate = options['learning_rate']
weight_decay = 0.00001
patience = 60 # number of epochs to wait for the validation loss to decrease before stopping the training
epochs = options['epochs']
loss_weights = options['loss_weights']


# G-FNO parameters
dv = options['dv']
S_in = options['S_in']
S_in_z = options['S_in_z']
S_out = options['S_out']
T_out = options['T_out']
padding = options['padding']


if __name__ == '__main__':
    
    ### DATA
    train_data = GeologyTracesDataset(path_data, options['dir_data_train'], S_in=S_in, S_in_z=S_in_z, S_out=S_out, T_out=T_out, transform_a='normal', N=Ntrain)
    val_data = GeologyTracesDataset(path_data, options['dir_data_val'], S_in=S_in, S_in_z=S_in_z, S_out=S_out, T_out=T_out, transform_a='normal', N=Nval)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)
    
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=2)

    ### MODEL
    device = get_device()
    print('Using device ', device)

    list_D = np.array(options['list_D'])
    list_D3 = np.array(options['list_D3'])
    list_M = np.array(options['list_M'])
    list_M3 = np.array(options['list_M3'])

    num_channels = 1
    model = GFNO_3D(num_channels, T_out, list_D, list_D3, list_M, list_M3, width=dv, padding=padding, initial_step=1, reflection=False, grid_type='cartesian')
    
    if torch.cuda.device_count() >= 1:
        NGPUs = torch.cuda.device_count()
    else:
        NGPUs = 0
    print(f'Using {NGPUs} GPUs for training')
    model.to(device)
    
    name_config = f"GFNO3D-dv{dv}-S{S_in}-T{T_out}-padding{padding}-learningrate{str(learning_rate).replace('.','p')}-" \
        f"L1loss{str(loss_weights[0]).replace('.','p')}-L2loss{str(loss_weights[1]).replace('.','p')}-"
    name_config += f"Ntrain{Ntrain}-batchsize{batch_size}"
    name_config += options['additional_name']

    # Store losses history
    train_history = {'loss_relative':[], 'loss_absolute':[]}
    val_history = {'loss_relative':[], 'loss_absolute':[]}
    best_loss = np.inf

    
    ### OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True) 
    early_stopper = EarlyStopper(patience=patience, min_delta=0.0001)

    
    ### TRAINING    
    for ep in range(epochs):
        t1 = timeit.default_timer()
        model.train()
        train_losses_relative = RunningAverage()
        train_losses_absolute = RunningAverage()
        
        # training
        for _ in train_loader:
            a = _[0].to(device)
            uE = _[1].to(device)
            uN = _[2].to(device)
            uZ = _[3].to(device)
            outE, outN, outZ = model(a)
            loss_rel = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=True)
            loss_abs = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=False)
            
            train_losses_relative.update(loss_rel.item(), get_batch_size(a))
            train_losses_absolute.update(loss_abs.item(), get_batch_size(a))
            
            # compute gradients and update parameters
            optimizer.zero_grad()
            loss_rel.backward()
            optimizer.step()
        
        train_history['loss_relative'].append(train_losses_relative.avg)
        train_history['loss_absolute'].append(train_losses_absolute.avg)

        # validation
        model.eval()
        with torch.no_grad():
            val_losses_relative = RunningAverage()
            val_losses_absolute = RunningAverage()

            # training
            for _ in val_loader:
                a = _[0].to(device)
                uE = _[1].to(device)
                uN = _[2].to(device)
                uZ = _[3].to(device)
                outE, outN, outZ = model(a)
                loss_rel_val = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=True)
                loss_abs_val = loss_criterion((outE,outN,outZ), (uE,uN,uZ), loss_weights, relative=False)

                val_losses_relative.update(loss_rel_val.item(), get_batch_size(a))
                val_losses_absolute.update(loss_abs_val.item(), get_batch_size(a))

            val_history['loss_relative'].append(val_losses_relative.avg)
            val_history['loss_absolute'].append(val_losses_absolute.avg)
            
            lr_scheduler.step(val_losses_relative.avg)

            t2 = timeit.default_timer()
            print(f'Epoch {ep+1}/{epochs}: {t2-t1:.2f}s - Training loss = {train_losses_relative.avg:.5f} - Validation loss = {val_losses_relative.avg:.5f}'\
                  f' - Training accuracy = {train_losses_absolute.avg:.5f} - Validation accuracy = {val_losses_absolute.avg:.5f}')

            # save the model
            if val_losses_relative.avg < best_loss:
                best_loss = val_losses_relative.avg
                torch.save(model.state_dict(), './logs/models/bestmodel-'+name_config+f'-epochs{epochs}.pt')
            
            if early_stopper.early_stop(val_losses_relative.avg):
                break

            # save intermediate losses
            if ep%2==0 and ep>0:
                with h5py.File(f'./logs/loss/loss-{name_config}-epoch{ep}on{epochs}.h5', 'w') as f:
                    f.create_dataset('train_loss_relative', data=train_history['loss_relative'])
                    f.create_dataset('train_loss_absolute', data=train_history['loss_absolute'])
                    f.create_dataset('val_loss_relative', data=val_history['loss_relative'])
                    f.create_dataset('val_loss_absolute', data=val_history['loss_absolute'])

                # remove the previous losses saved
                last_epoch_saved = ep
                if ep>2:
                    os.remove(f'./logs/loss/loss-{name_config}-epoch{ep-2}on{epochs}.h5')

    # save the final loss
    with h5py.File(f'./logs/loss/loss-{name_config}-epochs{ep+1}.h5', 'w') as f:
        f.create_dataset('train_loss_relative', data=train_history['loss_relative'])
        f.create_dataset('train_loss_absolute', data=train_history['loss_absolute'])
        f.create_dataset('val_loss_relative', data=val_history['loss_relative'])
        f.create_dataset('val_loss_absolute', data=val_history['loss_absolute'])

    os.remove(f'./logs/loss/loss-{name_config}-epoch{last_epoch_saved}on{epochs}.h5')
    
