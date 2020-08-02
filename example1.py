# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:46:31 2020

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

batchsize = 10
device = torch.device("cpu")

h = 10
species = 'gaussian-bernoulli'
optimizer = 'sgd'
drop_type = None
drop = None

params = {
        'maxepochs': [5, 10],
        'lr': [0.005],
        'lr_rate': 1,
        'lr_freq': 1000,
        'mmt1': 0.5,
        'mmt2': 0.9,
        'mmt_rate': 0.4,
        'mmt_freq': 5,
        'l2': 0,
        'l1': 0,
        'sparsw': 0,
        'sparsity': 0
        }

data = sio.loadmat('Directory_of_the_data_to_be_loaded.mat') # here a .mat file was to be loaded
data = data['data_name']
data = torch.from_numpy(data).to(torch.float64).to(device)
data = Data(data)
# data.normalize_()
data.standardize_()
data.makebatch(data=data.standard, batchsize=batchsize, inplace=True)
vis = data.dims

start_time = time.time()

dbn = DBN(visible=vis,hidden=h, species=species, optimizer=optimizer, drop_type=drop_type, drop=drop, seed=1234)
dbn.train(data.batchdata, params)
dbn.project(data.standard)
re = dbn.reconstruct(data.standard)

print('Elapsed time: {:.2f} seconds'.format(time.time() - start_time))
print('\nReconstruction error:')
if type(dbn) == RBM:
  print('- layer 1: {:.4f} '.format(re))
  plt.plot(dbn.error)
else:
  for l, r in enumerate(re):
      print('- layer {}: {:.4f} '.format((l+1), r))

  if len(dbn.layer) == 1:
      plt.plot(dbn.layer[0].error)
  else:
      fig, axs = plt.subplots(1, len(dbn.layer))
      for i in range(0, len(dbn.layer)):
          axs[i].plot(dbn.layer[i].error)