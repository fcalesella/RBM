# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:12:28 2020

@author: Federico Calesella
"""

import torch

class Data:
    
    def __init__(self, data, device=None):
        
        """
        Initialize a data a object.
        Inputs:
            data: the dataset.
            device: specify the device of the tensor. When not specified, if a GPU
                    is available, the device will be the first GPU, otherwise the CPU.
        """
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data.to(device=self.device)
        self.mean = torch.mean(data)
        self.mean_var = torch.mean(data, 0)
        self.std = torch.std(data)
        self.std_var = torch.std(data, 0)
        self.min = torch.min(data)
        self.max = torch.max(data)
        self.n = int(data.shape[0])
        self.dims = int(data.shape[1])
        if len(data.shape) > 3:
            raise TypeError("Argument data has too many dimensions: maximum = 3")
        elif len(data.shape) == 3:
            self.numbatches = int(data.shape[2])
            self.batchsize = self.n
            self.n = self.batchsize * self.numbatches
            self.mean_var = torch.mean(self.mean_var, 1)
            self.std_var = torch.std(self.std_var, 1)
            self.batchdata = data
        else:
            self.data = data.to(device=self.device)
        
###############################################################################
        
    def normalize(self):
        
        """
        Normalize the data.
        """
        
        norm = (self.data - self.min) / (self.max - self.min)
            
        return norm
    
###############################################################################
    
    def normalize_(self):
        
        """
        Normalize the data and create/overwrite the attribute norm.
        """
        
        self.norm = (self.data - self.min) / (self.max - self.min)
        
###############################################################################
        
    def standardize(self, entire=False):
        
        """
        Standardize the data.
        Inputs:
            entire: if True, the mean and the standard deviation of the whole dataset
                    are used to standardize the data. Otherwise, each variable is
                    individually standardized.
        """
        
        if entire:
            stand = (self.data - self.mean) / self.std
        elif not entire:
            stand = (self.data - self.mean_var) / self.std_var
        else:
            raise ValueError('Variable entire must be boolean')
            
        return stand
    
###############################################################################
    
    def standardize_(self, entire=False):
        
        """
        Standardize the data and create/overwrite the attribute standard.
        Inputs:
            entire: if True, the mean and the standard deviation of the whole dataset
                    are used to standardize the data. Otherwise, each variable is
                    individually standardized.
        """
        
        if entire:
            self.standard = (self.data - self.mean) / self.std
        elif not entire:
            self.standard = (self.data - self.mean_var) / self.std_var
        else:
            raise ValueError('Variable entire must be boolean')
        
###############################################################################
            
    def makebatch(self, *args, **kwargs):
        
        """
        Organize data in bacthes.
        Inputs:
            **kwargs:
                batchsize:  size (number of observations) in each batch.
                numbatches: number of batches.
                data:       dataset (if different from the initialized one).
                device:     the device of the data.
                inplace:    if True, the batchdata attribute will be created/overwritten.
        """
        
        if 'batchsize' not in kwargs and 'numbatches' not in kwargs:
            raise ValueError("Expected at least one of: batchsize or numbatches")
        if 'batchsize' in kwargs:
            batchsize = int(kwargs['batchsize'])
            numbatches = int(self.n / batchsize)
        elif 'numbatches' in kwargs:
            numbatches = int(kwargs['numbatches'])
            batchsize = int(self.n / 'numbatches')
        data = kwargs['data'] if 'data' in kwargs else self.data
        batchdata = torch.reshape(data, [numbatches, batchsize, self.dims])
        batchdata = batchdata.permute(1, 2, 0)
        if 'device' in kwargs:
            batchdata = batchdata.to(device=kwargs['device'])
        if 'inplace' in kwargs:
            if kwargs['inplace']:
                self.batchdata = batchdata
        self.batchsize = batchsize
        self.numbatches = numbatches
            
        return batchdata
    
###############################################################################
    
    def unbatch(self, *args, **kwargs):
        
        """
        Reorganize batch-data in 2D matrix.
        """
        
        if not hasattr(self, 'batchdata'):
            raise TypeError("Only batched data can be unbatched") 
        self.data = self.batchdata.permute(2, 0, 1)
        self.data = torch.reshape(self.data, [self.n, self.dims])
        
        return self.data