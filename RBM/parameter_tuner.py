# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:16:51 2020

@author: Federico Calesella
"""

import matplotlib.pyplot as plt
import time
import torch
from data import Data
from sklearn.model_selection import ParameterGrid
if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
    
###############################################################################
    
class ParameterTuner:
    
    def __init__(self, train_set, test_set, params, seed=None):
        
        """
        Initialize the tuner.
        Inputs:
            train_set: training data.
            test_set:  test data.
            params:    dictionary with the parameters required to train the estimator.
                       A list of dictionaries can also be provided for multi_tuning
                       and/or multi_layer optimizations.
           seed:       random seed.
        """
        
        self.tr = train_set
        self.tt = test_set
        self.seed = seed
        if type(params) is list:
            self.multi_grid = [None] * len(params)
            for np, par in enumerate(params):
                self.multi_grid[np] = list(ParameterGrid(par))
        else:
            self.grid = list(ParameterGrid(params))
            self.n_models = len(self.grid)
        if self.seed:
            torch.manual_seed(self.seed)
        
###############################################################################
    
    def check(self, obj):
        
        tr_check = getattr(obj, 'train', None)
        if not callable(tr_check):
            raise TypeError("%s has no method: train" % type(obj))
        te_check = getattr(obj, 'test', None)
        if not callable(te_check):
            raise TypeError("%s has no method: test" % type(obj))
        reset_check = getattr(obj, 'train', None)
        if not callable(reset_check):
            raise TypeError("%s has no method: reset" % type(obj))
        
        
###############################################################################

    def sample(self, n):
    
        indices = torch.randperm(self.n_models)[:n]
        self.sampled = [self.grid[int(i)] for i in indices]
    
        return self.sampled

###############################################################################

    def find_best(self, error, models=None, nbest=None):
        
        grid = self.grid if not models else models
        sorter = torch.argsort(error, axis=0)
        ser = error[sorter]
        spop = [grid[int(s)] for i,s in enumerate(sorter)]
        if ser.shape[0] == 1:
            bests = spop[0]
        else:
            ser = torch.squeeze(ser)
            if nbest:
                ser = ser[0:nbest]
            bests = spop[0:nbest] if nbest else spop
            
        return bests, ser

###############################################################################

    def grid_search(self, estimator):
        
        """
        Perform grid search parameter tuning on an estimator.
        Inputs:
            estimator: the estimator (learning function) to be trained and tested.
        Outputs:
            best_model: dictionary with the combination of values of the parameters
                        that minimized the test error.
            least:      minimum test error value recorded.
            re:         reconstruction error of each parameter combination.
        """
        
        self.check(estimator)
        print('Starting simulation: {:,} models will be tested\n'.format(self.n_models))
        start_time = time.time()
        re = self.optimizer_loss(estimator)
        smod, serr = self.find_best(re)
        try:
            least = serr[0]
            best_model = smod[0]
        except:
            least = torch.squeeze(serr)
            best_model = smod
            
        print('All combinations examined (elapsed time: {:.2f} seconds)'.format(time.time() - start_time))
        print('\nBest model:\n{}'.format(best_model))
        print('\nReconstruction error: {:.4f}'.format(least))
        plt.plot(re)
    
        return best_model, least, re

###############################################################################

    def random_search(self, estimator, n):
        
        """
        Perform random search parameter tuning on an estimator.
        Inputs:
            estimator: the estimator (learning function) to be trained and tested.
            n:         integer defining the number of random combinations to be tested.
        Outputs:
            best_model: dictionary with the combination of values of the parameters
                        that minimized the test error.
            least:      minimum test error value recorded.
            error:         reconstruction error of each parameter combination.
        """
        
        self.check(estimator)
        print('Starting simulation: {:,} models will be tested (out of {:,})\n'.format(n, self.n_models))
        start_time = time.time()
        models = self.sample(n)
        self.n_models = n
        error = self.optimizer_loss(estimator, models)
        smod, serr = self.find_best(error, models)
        try:
            least = serr[0]
            best_model = smod[0]
        except:
            least = torch.squeeze(serr)
            best_model = smod
            
        print('All combinations examined (elapsed time: {:.2f} seconds)'.format(time.time() - start_time))
        print('\nBest model:\n{}'.format(best_model))
        print('\nReconstruction error: {:.4f}'.format(least))
        plt.plot(error)
        
        return best_model, least, error
    
###############################################################################

    def next_gen(self, pop, pop_size, mutp):
        
        nbest = len(pop)
        ngenes = len(self.grid[-1].keys())
        nchild = pop_size - nbest
        parents = torch.randint(0, nbest, (nchild, 2))
        children = [None]*nchild
        genes_key = list(self.grid[-1].keys())
        
        for child in range(0, nchild):
            if nchild == 0:
                break
            offspring = {}
            sel_gene = torch.randn((ngenes))
            parent1_id = parents[child, 0]
            parent2_id = parents[child, 1]
            for ind, kp in enumerate(genes_key):
                if sel_gene[ind] > 0:
                    offspring[kp] = pop[parent1_id][kp]
                else:
                    offspring[kp] = pop[parent2_id][kp]
            subject_prob = 0.5 * torch.randn((1)) + 0.5
            if subject_prob > 1:
                subject_prob = 1
            if subject_prob < 0:
                subject_prob = 0
            if subject_prob <= mutp:
                rangen = int(torch.randint(0, ngenes, (1,1)))
                ranmut = int(torch.randint(0, len(self.grid), (1,1)))
                genek = genes_key[rangen]
                offspring[genek] = self.grid[ranmut][genek]
            
            children[child] = offspring
            
        return children

###############################################################################

    def genetic_search(self, estimator, epochs, pop_size, nbest, mutp):
        
        """
        Perform parameter tuning on an estimator using a genetic algorithm.
        Inputs:
            estimator: the estimator (learning function) to be trained and tested.
            epochs:    integer defining the number of evolutionary steps to be tested.
            pop_size:  integer defining the initial number of individuals (parameter 
                       combinations) to be tested.
            nbest:     integer defining the number of offsprings (new parameter combinations).
                       Note that only the offspring will be tested in each epoch, since the 
                       previous population has already been tested.
            mutp:      offspring's mutation probability.
        Outputs:
            best_model: dictionary with the combination of values of the parameters
                        that minimized the test error.
            least:      minimum test error value recorded.
            best_iter:  reconstruction error of the best offspring in each evolutionary step.
        """
        
        self.check(estimator)
        print('Starting simulation: {:,} evolutionary steps will be taken\n'.format(epochs))
        start_time = time.time()
        print('Building generation 0...')
        pop = self.sample(pop_size)
        error = self.optimizer_loss(estimator, pop)
        print('Generation 0 status: done\n')
        pop, serr = self.find_best(error, pop, nbest)
        least = serr[0]
        best_model = pop[0]
        best_iter = torch.zeros((epochs))
        wbar = tqdm(total=epochs, leave=False, desc="Assessing new generations", position=0)

        for iteration in range(0, epochs):
            new_gen = self.next_gen(pop, pop_size, mutp)
            gen_err = self.optimizer_loss(estimator, new_gen)
            pop = pop + new_gen
            error = torch.cat((serr, gen_err))
            pop, serr = self.find_best(error, pop, nbest)
            best_iter[iteration] = serr[0]
            if least > best_iter[iteration]:
                least = best_iter[iteration]
                best_model = pop[0]
            wbar.update()
        wbar.close()
        
        print('All generations completed (elapsed time: {:.2f} seconds)'.format(time.time() - start_time))
        print('\nBest model:\n{}'.format(best_model))
        print('\nReconstruction error: {:.4f}'.format(least))
        plt.plot(best_iter)

        return best_model, least, best_iter

###############################################################################

    def optimizer_loss(self, estimator, models=None):
        
        grid = self.grid.copy() if not models else models.copy()
        n_models = len(grid)
        re = torch.zeros((n_models))
        wbar = tqdm(total=n_models, leave=False, desc="Building models", position=0)
        for ind, par in enumerate(grid):
            estimator.reset()
            estimator.train(self.tr, par)
            re[ind] = estimator.test(self.tt)
            wbar.update()
            
        return re

###############################################################################

    def parameter_tuning(self, estimator, method, *args, **kwargs):
        
        self.check(estimator)
        if method == 'gs':
            best_model, least, error = self.grid_search(estimator)
        elif method == 'rs':
            if 'n' in kwargs:
                nrand = kwargs['n']
                best_model, least, error = self.random_search(estimator, nrand)
            else:
                raise ValueError("The argument n is required for random search tuning")
        elif method == 'ga':
            if 'epochs' and 'nindivid' and 'nbest' and 'mutp' in kwargs:
                epochs = kwargs['epochs']
                pop_size = kwargs['pop_size']
                nbest = kwargs['nbest']
                mutp = kwargs['mutp']
                best_model, least, error = self.genetic_search(estimator, epochs,
                                                               pop_size, nbest, mutp)
            else:
                raise ValueError("Missing argument for genetic search")
                
        return best_model, least, error
    
###############################################################################
    
    def multi_tuning(self, estimators, method, *args, **kwargs):
        
        """
        Perform parameter tuning on multiple estimators.
        Inputs:
            estimators: list containing the estimators (learning function) to be trained 
                        and tested.
            method:     string defining the method used to find the best parameter
                        combination. Possibilities are: 'gs' (for Grid Search), 'rs'
                        (for Random Search), and 'ga' (for Genetic Algorithm).
            **kwargs:   specific parameters to be entered in the tuning method.
        Outputs:
            bm:    dictionary with the combination of values of the parameters
                   that minimized the test error.
            least: minimum test error value recorded.
            re:    reconstruction error of each parameter combination.
        """
        
        if hasattr(self, 'grid'):
            ngrids = len(estimators)
            self.multi_grid = [self.grid] * ngrids
        if len(self.multi_grid) != len(estimators):
            raise ValueError("The number of parameter sets and estimators must be the same")
        
        bm = [None] * len(estimators)
        least = [None] * len(estimators)
        err = [None] * len(estimators)
        for n, estim in enumerate(estimators):
            self.grid = self.multi_grid[n]
            bm[n], least[n], err[n] = self.parameter_tuning(estim, method, **kwargs)
        
        return bm, least, err
    
###############################################################################
    
    def multi_layer(self, layers, method, *argd, **kwargs):
        
        """
        Perform parameter tuning on multiple layers of a deep neural network.
        Inputs:
            estimators: list containing the layers. Here training data are assumed 
                        to be organized in bacthes.
            method:     string defining the method used to find the best parameter
                        combination. Possibilities are: 'gs' (for Grid Search), 'rs'
                        (for Random Search), and 'ga' (for Genetic Algorithm).
            **kwargs:   specific parameters to be entered in the tuning method.
        Outputs:
            bm:    dictionary with the combination of values of the parameters
                   that minimized the test error.
            least: minimum test error value recorded.
            re:    reconstruction error of each parameter combination.
        """
        
        if hasattr(self, 'grid'):
            ngrids = len(layers)
            self.multi_grid = [self.grid] * ngrids
        if len(self.multi_grid) != len(layers):
            raise ValueError("The number of parameter sets and estimators must be the same")
        
        bm = [None] * len(layers)
        least = [None] * len(layers)
        err = [None] * len(layers)
        for nl, lay in enumerate(layers):
            self.grid = self.multi_grid[nl]
            self.n_models = len(self.grid)
            bm[nl], least[nl], err[nl] = self.parameter_tuning(lay, method, **kwargs)
            batched = Data(lay.batchidden)
            self.tr = lay.batchidden
            self.tt = batched.unbatch().to(dtype=torch.float64)
                
        return bm, least, err
          
