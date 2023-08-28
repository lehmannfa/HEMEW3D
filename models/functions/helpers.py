import numpy as np

class Normalization:
    ''' X: array (n, p) is a matrix of samples
    x: array (p,) is one sample to be normalized 
    y: array(p,) is one normalized sample that we want to denormalize
    '''
    
    def __init__(self, X, x_mean=None, x_std=None, norm_type=None):
        self.eps = 1e-6
        
        if x_mean is None:
            if norm_type=='log-normal':
                self.x_mean = np.mean(np.log(X))
            else:
                self.x_mean = X.mean()
        else:
            self.x_mean = x_mean
        
        if x_std is None:
            if norm_type=='log-normal':
                self.x_std = 4*np.std(np.log(X))
            else:
                self.x_std = 4*X.std()
        else:
            self.x_std = x_std

        if norm_type == 'normal-minmax':
            self.x_min = np.min((X - self.x_mean)/(self.x_std + self.eps))
            self.x_max = np.max((X - self.x_mean)/(self.x_std + self.eps))
            
        self.norm_type = norm_type
    
    def forward(self, x):
        if self.norm_type == 'normal':
            return (x - self.x_mean)/(self.x_std + self.eps)
        if self.norm_type == 'log':
            return np.log(x)
        if self.norm_type == 'log-normal':
            return (np.log(x) - self.x_mean)/self.x_std
        if self.norm_type == 'normal-minmax':
            temp = (x - self.x_mean)/(self.x_std + self.eps)
            return (temp - self.x_min)/(self.x_max - self.x_min)
        if self.norm_type is None:
            return x
    
    def inverse(self, y):
        if self.norm_type == 'normal' or self.norm_type == 'log-normal':
            return self.x_mean + self.x_std * y
        if self.norm_type == 'log':
            return np.exp(y)
        if self.norm_type == 'normal-minmax':
            temp = self.x_min + y*(self.x_max - self.x_min)
            return self.x_mean + self.x_std * temp
        if self.norm_type is None:
            return y
        

def linear_normalize(x, a=0, b=1):
    ''' normalize x initially between m and M to [a, b] '''
    m = np.min(x)
    M = np.max(x)
    return (b-a)*(x-m)/(M-m) + a

def stat_array(x):
    return f'shape: {np.shape(x)} - min: {np.min(x):.4f} - mean: {np.mean(x):.4f} - max: {np.max(x):.4f} - std: {np.std(x):.4f}'
