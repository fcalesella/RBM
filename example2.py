# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:51:59 2020

@author: Federico Calesella
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import time
import torch
import sys
sys.path.append('Directory_of_the_package_folder')
from data import Data
from rbm_new import RBM, DBN
from parameter_tuner import ParameterTuner

batchsize = 10
device = torch.device("cpu")

h = [10, 10]
species = 'gaussian-bernoulli'
optimizer = 'sgd'
drop_type = None
drop = None

params1 = {
        'maxepochs': torch.tensor([5]),
        'lr': torch.linspace(0.01, 0.001, 10),
        'lr_rate': torch.tensor([1]),
        'lr_freq': torch.tensor([1000]),
        'mmt1': torch.tensor([0.5]),
        'mmt2': torch.tensor([0.9]),
        'mmt_rate': torch.tensor([0.4]),
        'mmt_freq': torch.tensor([5]),
        'l2': torch.linspace(0.1, 0.0001, 10),
        'l1': torch.tensor([0]),
        'sparsw': torch.tensor([0]),
        'sparsity': torch.tensor([0])
        }


params2 = {
        'maxepochs': torch.tensor([5]),
        'lr': torch.linspace(0.1, 0.2, 10),
        'lr_rate': torch.tensor([1]),
        'lr_freq': torch.tensor([1000]),
        'mmt1': torch.tensor([0.5]),
        'mmt2': torch.tensor([0.9]),
        'mmt_rate': torch.tensor([0.4]),
        'mmt_freq': torch.tensor([5]),
        'l2': torch.linspace(0.1, 0.0001, 10),
        'l1': torch.tensor([0]),
        'sparsw': torch.tensor([0]),
        'sparsity': torch.tensor([0])
        }

params = [params1, params2]
data = sio.loadmat('Directory_of_the_data_to_be_loaded.mat') # here a .mat file was to be loaded
data = data['data_name']
data = torch.from_numpy(data).to(torch.float64).to(device)
data = Data(data)
# data.normalize_()
data.standardize_()
data.makebatch(data.standard, batchsize=batchsize, inplace=True)
vis = data.dims

start_time = time.time()

dbn = DBN(visible=vis,hidden=h, species=species, optimizer=optimizer, drop_type=drop_type, drop=drop, seed=1234)
tuner = ParameterTuner(data.batchdata, data.standard, params, seed=1234)
best_model, re, errors = tuner.multi_layer(dbn.layer, 'rs', n=2)

print('Elapsed time: {:.2f} seconds'.format(time.time() - start_time))