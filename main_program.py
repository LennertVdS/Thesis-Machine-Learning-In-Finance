from scr import basic_regression
from scr import standard_gpr
from scr import Sparse_gpr
from scr import Sparse_gpr_kmeans
from scr import Variational_gpr
from scr import plot_maker
import numpy as np

method = 'sparse_kmeans_VFE'       # basic, standard, kronecker, sparse_FITC, sparse_VFE, sparse_kmeans_FITC, sparse_kmeans_VFE, variational
type = 'vanilla_call'             # vanilla_call, vanilla_put, DOBP, american_call, american_put
model = 'heston'          # heston or vg (empty for american options)
amounttraining = 1000
amounttest = 1000
amountinducing = 50        # for the sparse, variational and inducing methods

np.random.seed(1)

if method == 'basic':
    basic_regression.basic_regression_ex(amounttraining,amounttest,model,type)
if method == 'standard':
    standard_gpr.standard_gpr_ex(amounttraining,amounttest,model, type)
if method == 'sparse_FITC' or method == 'sparse_VFE':
    Sparse_gpr.sparse_gpr_ex(amounttraining, amountinducing, amounttest, model, type, method)
if method == 'sparse_kmeans_FITC' or method == 'sparse_kmeans_VFE':
    Sparse_gpr_kmeans.sparse_gpr_kmeans_ex(amounttraining, amountinducing, amounttest, model, type, method)
if method == 'variational':
    Variational_gpr.variational_gpr_ex(amounttraining, amountinducing, amounttest, model, type)


if method == 'plotmaker':
    plot_maker.plot_maker_ex(amounttraining, amountinducing, amounttest, model, type, 'sparse_kmeans_VFE')

