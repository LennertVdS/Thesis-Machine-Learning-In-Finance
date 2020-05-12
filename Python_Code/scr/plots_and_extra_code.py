from scr import basic_regression
from scr import standard_gpr
from scr import sparse_gpr
from scr import sparse_gpr_kmeans
from scr import standard_gpr_gpytorch
from scr import sparse_vfe_gpr_gpytorch
from scr import skip_pytorch
from scr import map_bayesian_gpr_pymc3
from scr import full_bayesian_gpr_pymc3
from scr import map_bayesian_gpr_sparse_pymc3
from scr import full_bayesian_gpr_sparse_pymc3
from scr import data_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/DOBP_testset.csv',header=None)
# print(data)
# trainingParameters = data.iloc[0:1000,0:10]
# print(trainingParameters)
# trainingValues = data.iloc[0:1000,10]
# print(trainingValues)
#
# for i in range(0,1000):
#     trainingParameters.iat[i,4] = np.sqrt(trainingParameters.iat[i,4])
#
# trainingValues1 = pd.DataFrame(index=range(1), columns=range(1000))
#
# for i in range(0,1000):
#     trainingValues1.iat[0,i] = trainingValues[i]
#
# print(trainingValues1)
#
#
#
# trainingValues1.to_csv('testValues_DOBP_exact', encoding='utf-8', index=False)
# trainingParameters.to_csv('testParameters_DOBP_exact', encoding='utf-8', index=False)


fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.set_title('Prediction time SVG m=400')
ax.set_xlabel('Number of training points (n)')
ax.set_ylabel('Prediction time')
ax.set_ylim([0.012,0.025])
plt.plot([1000, 4000, 7000,10000, 12500, 15000, 17500, 20000], [0.0175, 0.0170, 0.0162, 0.0174, 0.0165, 0.0167, 0.0169, 0.0172], '-ro')
plt.show()
[0.0169,0.0204, 0.0174, 0.0179, 0.0283,0.0160]

fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.set_title('Prediction accuracy SVG m=400')
ax.set_xlabel('Number of training points (n)')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.0014,0.0020])
plt.plot([1000, 4000, 7000, 10000, 12500, 15000, 17500, 20000], [0.00186, 0.00170, 0.00165, 0.00164, 0.00162, 0.00154, 0.00152, 0.00151], '-ro')
plt.show()

fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.set_title('Prediction time VFE BBMM m=400')
ax.set_xlabel('Number of training points (n)')
ax.set_ylabel('Prediction time')
ax.set_ylim([0.012,0.025])
plt.plot([1000, 4000, 7000,10000, 12500, 15000, 17500, 20000], [0.0169, 0.0175, 0.0174, 0.0168, 0.0172, 0.0178, 0.0175, 0.0170], '-ro')
plt.show()


fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.set_title('Prediction accuracy VFE BBMM m=400')
ax.set_xlabel('Number of training points (n)')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.0014,0.0022])
plt.plot([1000, 4000, 7000, 10000, 12500, 15000, 17500, 20000], [0.00208, 0.00183, 0.00177, 0.00156, 0.00152,0.00150,0.00150,.00149], '-ro')
plt.show()



#
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.set_title('Prediction time different models')
# ax.set_xlabel('Amount of training points (n)')
# ax.set_ylabel('Prediction time')
# ax.plot([1000, 2000, 4000], [0.0542, 0.08901, 0.1803], '-ro', label = "GPR")
# ax.plot([1000, 4000, 10000], [0.0084, 0.0082, 0.0081], '-bd', label = "VFE m = 200")
# ax.plot([1000, 4000, 10000], [0.01602, 0.0171, 0.0172], '-bD', label = "VFE m = 400")
# ax.plot([1000, 4000, 10000], [0.0084, 0.0085, 0.0083], '--md', label = "SVG m = 200")
# ax.plot([1000, 4000, 10000], [0.01632, 0.01534, 0.0173], '--mD', label = "SVG m = 400")
# ax.plot([1000, 4000, 10000], [0.0404, 0.1680, 0.4430], '-go', label = "GPR BBMM")
# ax.plot([1000, 4000, 10000], [0.0085, 0.0084, 0.0087], ':cd', label = "VFE BBMM m = 200")
# ax.plot([1000, 4000, 10000], [0.0172, 0.0173, 0.0168], ':cD', label = "VFE BBMM m = 400")
# legend = ax.legend(loc='upper left', shadow=True, prop={'size': 10})
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.set_title('Prediction accuracy different models')
# ax.set_xlabel('Number of training points (n)')
# ax.set_ylabel('AAE')
# ax.set_ylim([0.0003,0.0027])
# ax.plot([1000, 2000, 4000], [0.00077, 0.00065, 0.00040], '-ro', label = "GPR")
# ax.plot([1000, 4000, 10000], [0.00181, 0.00162, 0.00162], '-bd', label = "VFE m = 200")
# ax.plot([1000, 4000, 10000], [0.00122, 0.00112, 0.00074], '-bD', label = "VFE m = 400")
# ax.plot([1000, 4000, 10000], [0.00203, 0.00196, 0.00193], '-md', label = "SVG m = 200")
# ax.plot([1000, 4000, 10000], [0.00174, 0.00161, 0.00160], '-mD', label = "SVG m = 400")
# ax.plot([1000, 4000, 10000], [0.00108, 0.00082, 0.00063], '-go', label = "GPR BBMM")
# ax.plot([1000, 4000, 10000], [0.00214, 0.00201, 0.002061], '-cd', label = "VFE BBMM m = 200")
# ax.plot([1000, 4000, 10000], [0.00234, 0.001887, 0.00173], '-cD', label = "VFE BBMM m = 400")
# legend = ax.legend(loc='upper right', shadow=True, prop={'size': 10}, bbox_to_anchor=(0.85, 0.98),
#           ncol=4)
# plt.show()
#
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.set_title('Training time different models')
# ax.set_xlabel('Number of training points (n)')
# ax.set_ylabel('Training time')
# ax.plot([1000, 2000, 4000], [7.9999, 121.61 ,507.78], '-ro', label = "GPR")
# ax.plot([1000, 4000, 10000], [10.002, 68.471, 220.1254], '-bd', label = "VFE m = 200")
# ax.plot([1000, 4000, 10000], [30.944, 155.48, 568.53], '-bD', label = "VFE m = 400")
# ax.plot([1000, 4000, 10000], [5.7177, 10.981, 28.051], '-md', label = "SVG m = 200")
# ax.plot([1000, 4000, 10000], [17.927, 23.926, 49.520], '-mD', label = "SVG m = 400")
# ax.plot([1000, 4000, 10000], [2.1939, 21.433, 138.54], '-go', label = "GPR BBMM")
# ax.plot([1000, 4000, 10000], [5.3466, 24.525, 52.551], '-cd', label = "VFE BBMM m = 200")
# ax.plot([1000, 4000, 10000], [6.7374, 33.665, 121.27], '-cD', label = "VFE BBMM m = 400")
# legend = ax.legend(loc='upper left', shadow=True, prop={'size': 10})
# plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title('Prediction time different models')
ax.set_xlabel('Amount of training points (n)')
ax.set_ylabel('Prediction time')
ax.plot([1000, 2000, 4000], [0.0410, 0.0818, 0.1649], '-ro', label = "GPR")
ax.plot([1000, 4000, 10000], [0.0077,0.0081 , 0.0087], '-bd', label = "VFE m = 200")
ax.plot([1000, 4000, 10000], [0.0155,0.0162 , 0.0157 ], '-bD', label = "VFE m = 400")
ax.plot([1000, 4000, 10000], [0.0075,0.0084 , 0.0084], '--md', label = "SVG m = 200")
ax.plot([1000, 4000, 10000], [0.0158, 0.0159, 0.0166], '--mD', label = "SVG m = 400")
ax.plot([1000, 4000, 10000], [0.0398, 0.1610, 0.4587], '-go', label = "GPR BBMM")
ax.plot([1000, 4000, 10000], [0.0082,0.0085 , 0.0086], ':cd', label = "VFE BBMM m = 200")
ax.plot([1000, 4000, 10000], [0.0160, 0.0150, 0.0155], ':cD', label = "VFE BBMM m = 400")
legend = ax.legend(loc='upper left', shadow=True, prop={'size': 10})
plt.show()


fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title('Prediction accuracy different models')
ax.set_xlabel('Number of training points (n)')
ax.set_ylabel('AAE')
ax.set_ylim([0.0001,0.0020])
ax.plot([1000, 2000, 4000], [0.00041, 0.00031, 0.00023], '-ro', label = "GPR")
ax.plot([1000, 4000, 10000], [0.00084, 0.00081, 0.00082], '-bd', label = "VFE m = 200")
ax.plot([1000, 4000, 10000], [0.00072, 0.00057, 0.00053], '-bD', label = "VFE m = 400")
ax.plot([1000, 4000, 10000], [0.00096, 0.00088, 0.00088], '-md', label = "SVG m = 200")
ax.plot([1000, 4000, 10000], [0.00087, 0.00087, 0.00086], '-mD', label = "SVG m = 400")
ax.plot([1000, 4000, 10000], [0.00076,0.00045 ,0.00037 ], '-go', label = "GPR BBMM")
ax.plot([1000, 4000, 10000], [0.00153,0.00121 , 0.00105], '-cd', label = "VFE BBMM m = 200")
ax.plot([1000, 4000, 10000], [0.00144, 0.00118, 0.00098 ], '-cD', label = "VFE BBMM m = 400")
legend = ax.legend(loc='upper right', shadow=True, prop={'size': 10}, bbox_to_anchor=(0.85, 0.98),
          ncol=4)
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title('Training time different models')
ax.set_xlabel('Number of training points (n)')
ax.set_ylabel('Training time')
ax.plot([1000, 2000, 4000], [18.027, 94.778,376.68 ], '-ro', label = "GPR")
ax.plot([1000, 4000, 10000], [8.8852,49.050 , 249.63], '-bd', label = "VFE m = 200")
ax.plot([1000, 4000, 10000], [26.810,128.36 ,580.45 ], '-bD', label = "VFE m = 400")
ax.plot([1000, 4000, 10000], [7.5439,9.6718 , 22.547], '-md', label = "SVG m = 200")
ax.plot([1000, 4000, 10000], [11.594, 20.857, 48.616], '-mD', label = "SVG m = 400")
ax.plot([1000, 4000, 10000], [5.0072,21.593 , 129.67], '-go', label = "GPR BBMM")
ax.plot([1000, 4000, 10000], [6.5637, 32.889, 78.772], '-cd', label = "VFE BBMM m = 200")
ax.plot([1000, 4000, 10000], [8.5568, 32.425, 114.53], '-cD', label = "VFE BBMM m = 400")
legend = ax.legend(loc='upper left', shadow=True, prop={'size': 10})
plt.show()

