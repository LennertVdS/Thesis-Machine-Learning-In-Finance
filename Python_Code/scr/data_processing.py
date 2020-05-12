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

"""
This file processes the input by generating data and assigning the right model to train
"""

def data_processing_ex(method, type, model, amountTraining, amountTest, amountInducing):

    if model == 'heston':
        if type == 'vanilla_call' or type == 'vanilla_put' :
            trainingValues , trainingParameters = \
                data_generator.data_generators_heston.training_data_heston_vanillas(amountTraining, type)
            testValues , testParameters = data_generator.data_generators_heston.test_data_heston_vanillas(amountTest, type)
        if type == 'DOBP':
            trainingValues , trainingParameters = data_generator.data_generators_heston.training_data_heston_down_and_out(amountTraining)
            testValues, testParameters = data_generator.data_generators_heston.test_data_heston_down_and_out(amountTest)

    if type == 'american_call' or type == 'american_put' :
        trainingValues, trainingParameters = data_generator.data_generators_american.training_data_american(amountTraining, type)
        testValues, testParameters = data_generator.data_generators_american.test_data_american(amountTest, type)

    if model == 'vg':
        if type == 'vanilla_call' or type == 'vanilla_put' :
            trainingValues , trainingParameters = \
                data_generator.data_generators_vg.training_data_vg_vanillas(amountTraining, type)
            testValues, testParameters = data_generator.data_generators_vg.test_data_vg_vanillas(amountTest, type)

    # data = np.concatenate((trainingParameters.to_numpy(), trainingValues.to_numpy().transpose()), axis=1)
    # data = pd.DataFrame(data)
    # print(data)
    # data.sort_values([5],axis = 0)
    # trainingParameters = data.iloc[:,0:9]
    # trainingValues = data.iloc[:,10]

    #save it

    #trainingValues.to_csv('trainingValues_vc_10000', encoding='utf-8', index=False)
    #trainingParameters.to_csv('trainingParameters_vc_10000', encoding='utf-8', index=False)
    #testValues.to_csv('testValues_vc', encoding='utf-8', index=False)
    #testParameters.to_csv('testParameters_vc', encoding='utf-8', index=False)

    if type == 'own_data':
          trainingValues = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/trainingValues_am_1000_exact')
          trainingParameters = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/trainingParameters_am_1000_exact')
          testValues = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/testValues_am_exact')
          testParameters = pd.read_csv('C:/Users/lenne/Desktop/bestanden/ku leuven/thesis/python/testParameters_am_exact')


    print('Generating data done')

    if method.find('Polynomial_Regression') != -1:
        basic_regression.basic_regression_ex(amountTraining, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('standard_GPR') != -1:
        standard_gpr.standard_gpr_ex(amountTraining, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('sparse_FITC') != -1 or method.find('sparse_VFE') != -1:
        sparse_gpr.method_finder(amountTraining, amountInducing, amountTest, model, type, method, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('sparse_kmeans_FITC') != -1 or method.find('sparse_kmeans_VFE') != -1:
        sparse_gpr_kmeans.method_finder(amountTraining, amountInducing, amountTest, method, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('variational_tf') != -1:
        from scr import variational_gpr_tensorflow
        variational_gpr_tensorflow.variational_gpr_ex(amountTraining, amountInducing, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('standard_gpy') != -1:
        standard_gpr_gpytorch.standard_gpr_pytorch_ex(amountTraining, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('VFE_gpy') != -1:
        sparse_vfe_gpr_gpytorch.sparse_vfe_gpr_pytorch_ex(amountTraining, amountInducing, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('SKIP_gpy') != -1:
        skip_pytorch.skip_torch_ex(amountTraining, amountInducing, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('MAP_bayesian_pym') != -1:
        map_bayesian_gpr_pymc3.map_bayesian_gpr_pymc3_ex(amountTraining, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('full_bayesian_pym') != -1:
        full_bayesian_gpr_pymc3.full_bayesian_gpr_pymc3_ex(amountTraining, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('MAP_bayesian_sparse_pym') != -1:
        map_bayesian_gpr_sparse_pymc3.map_bayesian_gpr_sparse_pymc3_ex(amountTraining, amountInducing, amountTest, trainingValues, trainingParameters, testValues, testParameters)
    if method.find('full_bayesian_sparse_pym') != -1:
        full_bayesian_gpr_sparse_pymc3.full_bayesian_gpr_sparse_pymc3_ex(amountTraining, amountInducing, amountTest, trainingValues, trainingParameters, testValues, testParameters)
