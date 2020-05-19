import numpy as np
from scr import  data_processing

"""
Welcome to the python code belonging to the thesis  "Machine Learning In Quantitative Finance, A kernel of truth" 

This is the main file, where one can modify certain parameters and repeat the experiments. I'd like to thank the
authors of the packages Tensorflow (tf), GPyTorch (gpy) and PYMC3 (pym) alleviating the work. When we use one of these
packages, we write "_package"

In order to test multiple methods at the same time, just put them all in the method string. It is also possible to include your own 
data or ignoring the financial aspect of this code by typing 'own_data' in the string corresponding to type.

Of course, the code can be easily extended training different types of options.

Author: Lennert Van der Schraelen

method: 
    Polynomial_Regression, standard_GPR, sparse_FITC, sparse_VFE, sparse_kmeans_FITC, sparse_kmeans_VFE, variational_tf
    standard_gpy, VFE_gpy, SKIP_gpy, MAP_bayesian_pym, full_bayesian_pym
    MAP_bayesian_sparse_pym, full_bayesian_sparse_pym

type:
    vanilla_call, vanilla_put, DOBP, american_call, american_put, own_data 
   
model:
    heston or vg (empty for american options or own data) 
   
amounttraining:
    amount of training points

amounttest:
    amount of testing points (for SKIP the one-dimensional amount)

amountinducing:
    amount of inducing points (if necessary)

"""

method = 'standard_GPR'
type = 'vanilla_call'
model = 'heston'
amounttraining = 500
amounttest = 100
amountinducing = 200

np.random.seed(1)

if __name__ == '__main__':
    data_processing.data_processing_ex(method, type, model, amounttraining, amounttest,amountinducing)
