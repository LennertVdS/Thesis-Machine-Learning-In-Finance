from scr import basic_regression
from scr import standard_gpr
from scr import Sparse_gpr
from scr import Sparse_gpr_kmeans
from scr import standard_gpr_gpytorch
from scr import sparse_vfe_gpr_gpytorch
from scr import variational_gpr_gpytorch
from scr import skip_pytorch
from scr import map_bayesian_gpr_pymc3
from scr import full_Bayesian_gpr_pymc3
from scr import map_bayesian_gpr_sparse_pymc3
from scr import full_Bayesian_gpr_sparse_pymc3
from scr import data_generator
import pandas as pd
import numpy as np

# data = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/MCHestonBarriers_train_10000.csv',header=None)
# print(data)
# trainingParameters = data.iloc[0:1000,0:9]
# print(trainingParameters)
# trainingValues = data.iloc[0:1000,11]
# print(trainingValues)
#
# trainingValues1 = pd.DataFrame(index=range(1), columns=range(1000))
#
# for i in range(0,1000):
#     trainingValues1.iat[0,i] = trainingValues[i]
#
# print(trainingValues1)
#
#
# trainingValues1.to_csv('trainingValues_DIBP_exact', encoding='utf-8', index=False)
# trainingParameters.to_csv('trainingParameters_DIBP_exact', encoding='utf-8', index=False)

data = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/DOBP_testset.csv',header=None)
print(data)
trainingParameters = data.iloc[0:1000,0:10]
print(trainingParameters)
trainingValues = data.iloc[0:1000,10]
print(trainingValues)

for i in range(0,1000):
    trainingParameters.iat[i,4] = np.sqrt(trainingParameters.iat[i,4])

trainingValues1 = pd.DataFrame(index=range(1), columns=range(1000))

for i in range(0,1000):
    trainingValues1.iat[0,i] = trainingValues[i]

print(trainingValues1)



trainingValues1.to_csv('testValues_DOBP_exact', encoding='utf-8', index=False)
trainingParameters.to_csv('testParameters_DOBP_exact', encoding='utf-8', index=False)

