import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

def score_metric(metric1, metric2, method='exponential'):
    ''' metric1: float or 1D-array of floats, metrics of the ground truth
    metric2: float or 1D-array of floats, metrics of the prediction
    method: string, exponential, bias or percentage, to define the score
    
    output: same shape as metric1 and metric2, score assessing the difference between the two metrics '''

    if method == 'exponential':
        min_metric = np.where(metric1<metric2, metric1, metric2) # min of metric1 and metric2
        score = 10*np.exp(-((metric1 - metric2)/min_metric)**2)

    elif method == 'bias':
        score = (metric2 - metric1)/metric1

    elif method == 'percentage':
        score = 100*(metric2 - metric1)/metric1

    else:
        raise Exception(f"method '{method}' is not defined, choose between 'exponential', 'bias' and 'percentage'")
    return score


def Arias_integral(velocity_values, dt=0.01):
    ''' velocity values: array (N, Nt) where N is the number of signals and Nt is the number of equally-spaced time steps
    dt: float, time step in seconds 
    
    output: array (N, Nt-2) '''
    # acceleration is computed with centered finite-differences from the velocity timeseries
    accel_values = (velocity_values[:, 2:] - velocity_values[:,:-2])/(2*dt)
    IA = np.cumsum(accel_values**2, axis=1)*np.pi*dt/(2*9.81)
    return IA


def relative_significant_duration(Aint, dt = 0.01):
    ''' Aint: array (N, Nt-2) contains the Arias integral for all signals and all timesteps
    
    output: array (N,) '''
    IA = Aint[:,-1]    
    IA_array = np.repeat(IA.reshape(-1,1), Aint.shape[1], axis=1)
    t05 = np.argmin(Aint < 0.05*IA_array, axis=1)
    t95 = np.argmin(Aint < 0.95*IA_array, axis=1)
    return dt*(t95 - t05)


def energy_integral(velocity_values, dt=0.01):
    ''' velocity values: array (N, Nt) where N is the number of signals and Nt is the number of equally-spaced time steps
    dt: float, time step in seconds 
    
    output: array (N, ) '''
    IE = np.sum(velocity_values**2, axis=1)*dt
    return IE


def cumulative_absolute_velocity(velocity_values, dt=0.01):
    ''' velocity values: array (N, Nt) where N is the number of signals and Nt is the number of equally-spaced time steps
    dt: float, time step in seconds 
    
    output: array (N, ) '''
    # acceleration is computed with centered finite-differences from the velocity timeseries
    accel_values = (velocity_values[:, 2:] - velocity_values[:,:-2])/(2*dt)
    return np.sum(np.abs(accel_values), axis=1)*dt


def fourier_spectra(velocity_values, low_freq, mid_freq, high_freq, dt=0.01):
    ''' velocity values: array (N, Nt) where N is the number of signals and Nt is the number of equally-spaced time steps
    low_freq: tuple (float, float), frequency range of the low frequency band
    mid_freq: tuple (float, float), frequency range of the medium frequency band
    high_freq: tuple (float, float), frequency range of the high frequency band
    dt: float, time step in seconds 
    
    output: tuple (array (N,), array (N,), array (N,)) where each array contains the mean of Fourier coefficients
    for the corresponding frequencies '''
    
    # compute the Fourier coefficients of each signal and associate the corresponding frequency
    fourier = pd.DataFrame(rfft(velocity_values, axis=1).T, 
                           index=rfftfreq(velocity_values.shape[1], d=dt))
    fourier_low_freq = fourier.loc[(fourier.index>=low_freq[0])&(fourier.index<=low_freq[1])] # all frequencies the low frequency band
    fourier_low_freq = fourier_low_freq.mean(axis=0).abs().values # mean of all frequencies 
    
    fourier_mid_freq = fourier.loc[(fourier.index>=mid_freq[0])&(fourier.index<=mid_freq[1])]
    fourier_mid_freq = fourier_mid_freq.mean(axis=0).abs().values
    
    fourier_high_freq = fourier.loc[(fourier.index>=high_freq[0])&(fourier.index<=high_freq[1])]
    fourier_high_freq = fourier_high_freq.mean(axis=0).abs().values
    
    return (fourier_low_freq, fourier_mid_freq, fourier_high_freq)
