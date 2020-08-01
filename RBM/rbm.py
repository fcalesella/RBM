# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:34:46 2020

@author: Federico Calesella
"""

import matplotlib.pyplot as plt
import math
import torch
if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

###############################################################################
###############################################################################

class RBM:

  def __init__(self, visible, hidden, species, optimizer='sgd', *args, **kwargs):
      
    """ 
    Initialize a Restricted Boltzmann Machine (RBM).
    Inputs:
        visible:   integer defining the number of visible units (number of features).
        hidden:    integer defining the number of hidden units.
        species:   string defining the activation function of the visibile and hidden units
                   seprated by a hyphen (e.g., 'gaussian-bernoulli'). Combination of 
                   gaussian, bernoulli, and relu activation functions are available.
        optimizer: string defining the learning algorithm: 'sgd' and 'adam' are available.
        **kwargs: 
            drop_type: string defining the type of dropout (if desired). Possibilities 
                       are: 'out' (for Dropout) and 'connect' (for DropConnect, see 
                       Regularization of Neural Networks using DropConnect). If drop_type 
                       is defined, the parameter drop is required.
            drop:      scalar defining probability of connections to drop.
            seed:      integer defining the random seed.
            device:    string defining the execution device. If is not defined the code
                       will be executed on GPU if present, otherwise on cpu.
    """

    species_options = ['gaussian', 'bernoulli', 'relu']
    species = species.split('-')
    if species[0] not in species_options or species[1] not in species_options:
        raise ValueError("Invalid species. Expected one of: %s" % species_options)
    optimizer_options = ['sgd', 'adam']
    if optimizer not in optimizer_options:
        raise ValueError("Invalid optimizer. Expected one of: %s" % optimizer_options)
    self.h = hidden
    self.v = visible
    self.species = species
    self.optimizer = optimizer
    self.drop = kwargs['drop'] if 'drop' in kwargs else None
    self.seed = kwargs['seed'] if 'seed' in kwargs else None
    if 'drop_type' in kwargs:
        self.drop_type = kwargs['drop_type']
        if self.drop_type:
            drop_options = ['out', 'connect'] 
            if self.drop_type not in drop_options:
                raise ValueError("Invalid drop_type. Expected one of: %s" % drop_options)
    else:
        self.drop_type = None
    if 'device' in kwargs:
        self.device = kwargs['device']
    else:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if self.seed:
      torch.manual_seed(self.seed)
    self.set_params()

###############################################################################

  def set_params(self):

    self.w = 0.01 * torch.randn((self.v, self.h), dtype=torch.float64, device=self.device)
    self.hb = torch.zeros((self.h), dtype=torch.float64, device=self.device)
    self.vb = torch.zeros((self.v), dtype=torch.float64, device=self.device)
    self.w.grad = torch.zeros_like(self.w)
    self.vb.grad = torch.zeros_like(self.vb)
    self.hb.grad = torch.zeros_like(self.hb)

###############################################################################

  def reset(self, new_seed=None):
      
    """
    Reset the RBM to the initial state.
    Inputs:
        new_seed: integer defining teh random seed. If specified the RBM will be
                  reset on the new seed, otherwise on the old seed. If new_see = -1,
                  the RBM will be reset without any seed.
    """
    
    if new_seed:
      if new_seed == -1:
        self.seed = None
      else:
        self.seed = new_seed
    if self.seed:
      torch.manual_seed(self.seed)
    self.set_params()
    
###############################################################################
    
  def set_opt(self, params):
      
      if self.optimizer == 'sgd':
          lr = params['lr']
          momentum = params['mmt1'] if 'mmt1' in params else 0
          self.wopt = torch.optim.SGD([self.w], lr=lr, momentum=momentum)
          self.vbopt = torch.optim.SGD([self.vb], lr=lr, momentum=momentum)
          self.hbopt = torch.optim.SGD([self.hb], lr=lr, momentum=momentum)
      elif self.optimizer == 'adam':
          lr = params['lr']
          b1 = params['mmt1'] if 'b1' in params else 0.9
          b2 = params['mmt2'] if 'b2' in params else 0.999
          self.wopt = torch.optim.Adam([self.w], lr=lr, betas=(b1, b2))
          self.vbopt = torch.optim.Adam([self.vb], lr=lr, betas=(b1, b2))
          self.hbopt = torch.optim.Adam([self.hb], lr=lr, betas=(b1, b2))
          
###############################################################################
          
  def momentum_step(self, params, epoch):
      
      mmt1 = params['mmt1']
      mmt2 = params['mmt2']
      mmt_rate = params['mmt_rate']
      mmt_freq = params['mmt_freq']
      
      momentum = mmt1 + mmt_rate * math.floor(epoch/mmt_freq)
      if momentum:
          if momentum > mmt2:
              momentum = mmt2
      
      self.wopt.param_groups[0]['momentum'] = momentum
      self.vbopt.param_groups[0]['momentum'] = momentum
      self.hbopt.param_groups[0]['momentum'] = momentum
  
###############################################################################

  def lr_step(self, params, epoch):
      
      lr = params['lr']
      lr_rate = params['lr_rate']
      lr_freq = params['lr_freq']
      
      lr_now = lr * lr_rate ** math.floor(epoch/lr_freq)
      
      self.wopt.param_groups[0]['lr'] = lr_now
      self.vbopt.param_groups[0]['lr'] = lr_now
      self.hbopt.param_groups[0]['lr'] = lr_now

###############################################################################

  def active(self, data, weights, bias, act_type):
    
    proj_data = torch.mm(data, weights)
    bias = bias.expand_as(proj_data)
    if act_type == 'gaussian':
        probs = proj_data + bias
    elif act_type == 'bernoulli':
        probs = 1 / (1 + torch.exp(-proj_data - bias))
    elif act_type == 'relu':
        probs = (proj_data + bias).clamp(0)
    # for other activation functions:
    # see Annealing Gaussian into ReLU: A New Sampling Strategy for Leaky-ReLU RBM 

    return probs

###############################################################################    

  def grads(self, data, hidden, negdata, neghidden, w):
      
      posprods = torch.mm(data.T, hidden)
      negprods = torch.mm(negdata.T, neghidden)
      posvisact = torch.sum(data, 0)
      negvisact = torch.sum(negdata, 0)
      poshidact = torch.sum(hidden, 0)
      neghidact = torch.sum(neghidden, 0)
      
      l1_reg = self.l1 * torch.sign(w)
      l2_reg = w * self.l2
      reg_term = l1_reg + l2_reg
      # to implementation of elastic net regularization for RBM see:
      # Elastic restricted Boltzmann machines for cancer data analysis
      
      w_grad = -(((posprods - negprods) / self.batchsize) - reg_term)
      vb_grad = -((posvisact - negvisact) / self.batchsize)
      hb_grad = -((poshidact - neghidact) / self.batchsize)
      
      return w_grad, vb_grad, hb_grad
      
###############################################################################
      
  def update(self):
      
      self.wopt.step()
      self.vbopt.step()
      self.hbopt.step()
      self.wopt.zero_grad()
      self.vbopt.zero_grad()
      self.hbopt.zero_grad()
      
###############################################################################

  def sparse(self, q):
      
      spars_reg = self.sparsw * (q - self.sparsity)
      self.w -= spars_reg
      self.hb -= spars_reg

###############################################################################
      
  def cd(self, data, w, vb, hb):
      
      hidden = self.active(data, w, hb, self.species[1])
      if self.species[1] == 'bernoulli':
          threshold = torch.rand_like(hidden, device=self.device)
          states = (hidden > threshold).type(dtype = torch.float64)
      else:
          states = hidden
      negdata = self.active(states, w.T, vb, self.species[0])
      neghidden = self.active(negdata, w, hb, self.species[1])
      
      return hidden, negdata, neghidden
  
###############################################################################

  def step(self, data):

    hidden, negdata, neghidden = self.cd(data, self.w, self.vb, self.hb)
    err = torch.mean((data - negdata)**2)
    q = torch.sum(hidden, 0) / self.batchsize
    w_grad, vb_grad, hb_grad = self.grads(data, hidden, negdata, neghidden, self.w)
    self.w.grad.data = w_grad
    self.vb.grad.data = vb_grad
    self.hb.grad.data = hb_grad
    self.update()
    self.sparse(q)
    
    return hidden, err

###############################################################################

  def step_dropout(self, data):

    numcases = self.batchsize
    err = torch.zeros((numcases), device=self.device)
    q = torch.zeros((self.h), device=self.device)
    w_grad_cum = torch.zeros_like(self.w, device=self.device)
    vb_grad_cum = torch.zeros_like(self.vb, device=self.device)
    hb_grad_cum = torch.zeros_like(self.hb, device=self.device)
    drop_prob = 1 - self.drop
    drop_w = torch.empty((numcases, self.h), device=self.device).bernoulli(drop_prob)
    for ex in range(0, numcases):
      example = data[ex, :].unsqueeze_(0)
      w = self.w * drop_w[ex, :].expand_as(self.w)

      hidden, negdata, neghidden = self.cd(example, w, self.vb, self.hb)
      err[ex] = torch.mean((example - negdata)**2)
      q += torch.squeeze(hidden, 0)
      w_grad, vb_grad, hb_grad = self.grads(example, hidden, negdata, neghidden, w)
      w_grad *= drop_w[ex, :]
      w_grad_cum += w_grad
      vb_grad_cum += vb_grad
      hb_grad_cum += hb_grad
    self.w.grad.data = w_grad_cum
    self.vb.grad.data = vb_grad_cum
    self.hb.grad.data = hb_grad_cum
    self.update()
    q = q / numcases
    self.sparse(q)
    err = torch.mean(err)

    return hidden, err

###############################################################################

  def step_dropconnect(self, data):
      numcases = self.batchsize
      err = torch.zeros((numcases), device=self.device)
      q = torch.zeros((self.h), device=self.device)
      w_grad_cum = torch.zeros_like(self.w, device=self.device)
      vb_grad_cum = torch.zeros_like(self.vb, device=self.device)
      hb_grad_cum = torch.zeros_like(self.hb, device=self.device)
      drop_prob = 1 - self.drop
      drop_w = torch.empty((self.v, self.h, numcases), device=self.device).bernoulli(drop_prob)
      drop_vb = torch.empty((numcases, self.v), device = self.device).bernoulli(drop_prob)
      drop_hb = torch.empty((numcases, self.h), device = self.device).bernoulli(drop_prob)
      for ex in range(0, numcases):
          example = data[ex, :].unsqueeze_(0)
          w = self.w * drop_w[:, :, ex]
          hb = self.hb * drop_hb[ex, :]
          vb = self.vb * drop_vb[ex, :]
          
          hidden, negdata, neghidden = self.cd(example, w, vb, hb)
          err[ex] = torch.mean((example - negdata)**2)
          q += torch.squeeze(hidden, 0)
          w_grad, vb_grad, hb_grad = self.grads(example, hidden, negdata, neghidden, w)
          w_grad *= drop_w[:, :, ex]
          vb_grad *= drop_vb[ex, :]
          hb_grad *= drop_hb[ex, :]
          w_grad_cum += w_grad
          vb_grad_cum += vb_grad
          hb_grad_cum += hb_grad
      self.w.grad.data = w_grad_cum
      self.vb.grad.data = vb_grad_cum
      self.hb.grad.data = hb_grad_cum
      self.update()
      q = q / numcases
      self.sparse(q)
      err = torch.mean(err)
    
      return hidden, err

###############################################################################

  def train(self, data, params, nlayer=0):
      
    """
    Train the RBM.
    Inputs:
        data:   data matrix.
        params: dictionary containing the desired parameters. Available parameters are:
            'maxepochs': integer defining the number of epochs to train.
            'lr':        scalar defining the learning rate.
            'lr_rate':   scalar defining the rate of learning rate decrease.
            'lr_freq':   integer defining the frequency of learning rate decrease.
            'mmt1':      scalar defining the initial momentum (with SGD) or the lower 
                         beta (with Adam).
            'mmt2':      scalar defining the maximum momentum (with SGD) or the upper 
                         beta (with Adam)
            'mmt_rate':  with SGD is the scalar defining the rate of the momentum decrease.
            'mmt_freq':  with SGD is the scalar defining teh frequency of the momentum decrease.
            'l2':        scalar defining the weight of the L2 regularization.
            'l1':        scalar defining the weight of the L1 regularization.
            'sparsw':    scalar defining the weight of the sparsity regularization for units.
            'sparsity':  scalar defining the probability of activation of the units
                         (sparsity regularization for units).
        nlayer: number of DBN layer. Default is 0, since it is the RBM class.
    Outputs:
        batchidden: data created in the negative phase in the last epoch.
        error:      error of each epoch.
    """

    maxepochs = int(params['maxepochs'])
    self.l2 = params['l2'] if 'l2' in params else 0
    self.l1 = params['l1'] if 'l1' in params else 0
    self.sparsw = params['sparsw'] if 'sparsw' in params else 0
    self.sparsity = params['sparsity'] if 'sparsity' in params else 0
    self.batchsize = data.shape[0]
    numbatches = data.shape[2]
    self.batchidden = torch.zeros((self.batchsize, self.h, numbatches), device=self.device)
    self.error = torch.zeros((maxepochs))
    self.set_opt(params)
    
    wbar = tqdm(total=maxepochs, leave=False, desc="Layer {} progress".format(nlayer+1), position=0)

    for epoch in range(0, maxepochs):
        errsum = 0

        for batch in range(0, numbatches):
          
          data_mb = data[:, :, batch]
          data_mb = torch.as_tensor(data_mb, dtype=torch.float64, device=self.device)
          
          if self.drop_type == 'out':
              hidden, err = self.step_dropout(data_mb)
          elif self.drop_type == 'connect':
              hidden, err = self.step_dropconnect(data_mb)
          else:
              hidden, err = self.step(data_mb)

          errsum = errsum + err
          if epoch == maxepochs:
              self.batchidden[:, :, batch] = hidden
            
        if self.optimizer == 'sgd':
            if 'mmt_rate' in params and 'mmt_freq' in params:
                self.momentum_step(params, epoch)
        if 'lr_rate' in params and 'lr_freq' in params:
            self.lr_step(params, epoch)
            
        self.error[epoch] = errsum / numbatches
        wbar.update()
    wbar.close()
    
    return self.batchidden, torch.squeeze(self.error)

###############################################################################

  def project(self, data):
      
    act = self.active(data, self.w, self.hb, self.species[1])
    if self.species[1] == 'bernoulli':
        threshold = torch.rand(act.shape, device=self.device)
        self.feat = (act > threshold).type(dtype = torch.float64)
    else:
        self.feat = act

    return self.feat

###############################################################################

  def reconstruct(self, data, retype=None):

    if retype:
      species = retype
    else:
      species = self.species[0]
    srec = self.active(self.feat, self.w.T, self.vb, species)    
    self.re = torch.mean((srec - data)**2)
        
    return self.re

###############################################################################

  def test(self, data, retype=None):
      
    """
    Test the RBM by projecting the test data in the feature space and back-projecting it.
    Inputs:
        data:   test data.
        retype: if None, the previously selected activation function for the visible
                layer is used. If another activation function is desired, define it.
    Outputs:
        re: recontruction error.
    """

    self.project(data)
    re = self.reconstruct(data, retype)

    return re          

###############################################################################
###############################################################################

class DBN(RBM):

  def __init__(self, visible, hidden, species, optimizer='sgd', *args, **kwargs):

    """ 
    Initialize a Deep Belief Network (DBN) by stacking multiple RBMs.
    Inputs:
        visible:   integer defining the number of visible units (number of features).
        hidden:    integer or list defining the number of hidden units in each layer.
        species:   string or list defining the activation function of the visibile and 
                   hidden units (for each layer) seprated by a hyphen (e.g., 
                   'gaussian-bernoulli' or ['gaussian-bernoulli', 'bernoulli-relu']). 
                   Combination of gaussian, bernoulli, and relu activation functions are 
                   available. If only one string is provided, all the subsequent layers 
                   have the hidden layer activation function (e.g., 'gaussian-bernoulli -> 
                   visible layer is gaussian, hidden layer 1 is bernoulli, and hidden
                   layer 2 is bernoulli too).
        optimizer: string defining the learning algorithm: 'sgd' and 'adam' are available.
                   If different optimizers are desired for each layer, please define them
                   for each layer (in a list).
        **kwargs: 
            drop_type: string defining the type of dropout (if desired). Possibilities 
                       are: 'out' (for Dropout) and 'connect' (for DropConnect, see 
                       Regularization of Neural Networks using DropConnect). If drop_type 
                       is defined, the parameter drop is required. If different drop_type
                       are desired for each layer, please define them for each 
                       layer (in a list).
            drop:      scalar defining probability of connections to drop. If different
                       drop are desired for each layer, please define them for each 
                       layer (in a list).
            seed:      integer defining the random seed.
            device:    string defining the execution device. If is not defined the code
                       will be executed on GPU if present, otherwise on cpu.
    """
    try:
      vis = [visible, *hidden]
    except:
      hidden = [hidden]
      vis = [visible, *hidden]
    if type(species) is not list:
        spec = species.split('-')
        next_spec = spec[1] + '-' + spec[1]
        next_spec = [next_spec] * (len(hidden) - 1)
        species = [species, *next_spec]
    if type(optimizer) is not list:
        optimizer = [optimizer] * len(hidden)
    drop_type = kwargs['drop_type'] if 'drop_type' in kwargs else None
    if type(drop_type) is not list:
        drop_type = [drop_type] * len(hidden)
    drop = kwargs['drop'] if 'drop' in kwargs else None
    if type(drop) is not list:
        drop = [drop] * len(hidden)
    if 'device' in kwargs:
        device = kwargs['device']
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = kwargs['seed'] if 'seed' in kwargs else None
    self.layer = [None]*len(hidden)
    for nl, v in enumerate(vis[:-1]):
      self.layer[nl] = RBM(v, hidden[nl], species[nl], optimizer[nl],
                drop_type=drop_type[nl], drop=drop[nl], device=device, seed=seed)

###############################################################################

  def reset(self, new_seed=None):
      
    """
    Reset the DBN to the initial state.
    Inputs:
        new_seed: integer defining teh random seed. If specified the RBM will be
                  reset on the new seed, otherwise on the old seed. If new_see = -1,
                  the RBM will be reset without any seed.
    """
    
    for nlayer in self.layer:
      nlayer.reset(new_seed)
      
###############################################################################
      
  def get_params(self, params):
      
      layer_par = {}
      par_ls = [None] * len(self.layer)
      for nl in range(len(self.layer)):
          for k in params.keys():
              if type(params[k]) is not list:
                  params[k] = [params[k]]
              if len(params[k]) < nl + 1:
                  params[k].append(params[k][nl-1])
              layer_par[k] = params[k][nl]
              
          par_ls[nl] = layer_par.copy()
          
      return par_ls

###############################################################################

  def train(self, data, params):
      
    """
    Train the DBN.
    Inputs:
        data:   data matrix.
        params: dictionary containing the desired parameters. If different parameters 
                for each layer are desired, please provide a list with values of that
                parameter. If only one value is provided, that one is used for all the layers.
                If less parameter values than the number of layers are provided, than 
                the subsequent layers use the last value available. Available parameters are:
            'maxepochs': integer defining the number of epochs to train.
            'lr':        scalar defining the learning rate.
            'lr_rate':   scalar defining the rate of learning rate decrease.
            'lr_freq':   integer defining the frequency of learning rate decrease.
            'mmt1':      scalar defining the initial momentum (with SGD) or the lower 
                         beta (with Adam).
            'mmt2':      scalar defining the maximum momentum (with SGD) or the upper 
                         beta (with Adam)
            'mmt_rate':  with SGD is the scalar defining the rate of the momentum decrease.
            'mmt_freq':  with SGD is the scalar defining teh frequency of the momentum decrease.
            'l2':        scalar defining the weight of the L2 regularization.
            'l1':        scalar defining the weight of the L1 regularization.
            'sparsw':    scalar defining the weight of the sparsity regularization for units.
            'sparsity':  scalar defining the probability of activation of the units
                         (sparsity regularization for units).
    Outputs:
        batchidden: data created in the negative phase in the last epoch.
        error:      error of each epoch.
    """
    
    error = {}
    par_ls = self.get_params(params)
    for nl, nlayer in enumerate(self.layer):
      data, error[nl] = nlayer.train(data, par_ls[nl])
      
    return data, error

###############################################################################

  def project(self, data):

    feat = {}
    feat[0] = self.layer[0].project(data)
    for nl in range(1, len(self.layer)):
      nlayer = self.layer[nl]
      feat[nl] = nlayer.project(self.layer[nl-1].feat)

    return feat

###############################################################################

  def reconstruct(self, data):

    self.re = torch.zeros(len(self.layer))
    for nl, nlayer in enumerate(self.layer):
      srec = nlayer.active(nlayer.feat, nlayer.w.T, nlayer.vb, nlayer.species[0])
        
      if nl > 0:
        for back in range(nl, 0, -1):
          nlayer = self.layer[back-1]
          srec = nlayer.active(srec, nlayer.w.T, nlayer.vb, nlayer.species[0])     

      self.re[nl] = torch.mean((srec - data)**2)
        
    return self.re

###############################################################################

  def test(self, data, econ=True):
      
    """
    Test the RBM by projecting the test data in the feature space and back-projecting it.
    Inputs:
        data:   test data.
        econ:   if True the function will output only the reconstruction error in the visible
                layer, otherwise the recontruction error of each layer.
    Outputs:
        re: recontruction error.
    """

    _ = self.project(data)
    re = self.reconstruct(data)
    if econ:
      re = re[-1]
    return re