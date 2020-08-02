# RBM
 Restricted Boltzmann Machine and Deep Belief Network

# Project overview
The aim of this project is to extract non-linear features in an unsupervised fashion, by building and training a Restricted Boltzmann Machine (RBM) and/or a Deep Belief Network (DBN). 

For this purpose, four classes are available:
- Data: for data managing
- RBM: for RBM initialization and training
- DBN: for DBN initialization and training
- ParameterTuner: for hyperparameter optimization

The code requires the availability of some packages, namely:
- pytorch
- scikit-learn 
- matplotlib
- tqdm
- time
- math

# Data
This class keeps track of the several measures, such as: mean, standard deviation, minimum, maximum, number of oservations, and number of features. It also allows for:
- standardization
- normalization
- organization in batches (2D to 3D)
- back up data to original organization (3D to 2D)

# RBM and DBN
Here, several advanced options are made available for RBM/DBN initialization and training, such as:
- Graphics Processing Unit (GPU) for accelerated computing
- activation functions: gaussian, sigmoid, and ReLU
- optimizer: Stochastic Gradient Descent (SGD) or Adaptive moment estimation (Adam)
- regularizations: dropout, dropconnect, L1 penalty, L2 penalty, and unit sparsity
- learning rate with step decay
- momentum with step increase

Furthermore, many others options can be easily implemented, thanks to the use and integration of pytorch (e.g., other optimizers, activation functions, or leanring rate decays).

# ParameterTuner
Here several algorithms are provided, namely:
- Grid Search
- Random Search
- Genetic algorithm

# Examples
 Here two examples are provided:
 - example1: initialize and train an RBM.
 - example2: parameter optimization of a DBN with two layers.
