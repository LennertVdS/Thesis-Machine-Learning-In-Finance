import numpy as np
import pandas as pd
from scr import models_algorithms
from timeit import default_timer as timer


class data_generators_heston:

    def training_data_heston_vanillas(amountTraining, type):

        modelListTraining = []
        valuesFFTCallsTraining = pd.DataFrame(index=range(1), columns=range(amountTraining))
        parametersModelsTraining = pd.DataFrame(index=range(amountTraining), columns=range(10))

        for x in range(amountTraining):

            # generate pseudo-random numbers for the Training set

            stock_value = 1
            strike = np.random.uniform(0.4, 1.6) * stock_value
            maturity = np.random.uniform(11 / 12, 1)
            interest = np.random.uniform(0.015, 0.025)
            dividend_yield = np.random.uniform(0, 0.05)

            # heston

            kappa = np.random.uniform(1.4, 2.6)
            rho = np.random.uniform(-0.85, -0.55)
            theta = np.random.uniform(0.45, 0.75)
            eta = np.random.uniform(0.01, 0.1)
            sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.9048)))


            modelListTraining.append(
                models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value,
                                                        interest, dividend_yield))

        for i, model in enumerate(modelListTraining):
            if type == 'vanilla_call':
                valuesFFTCallsTraining[i] = model.heston_carr_madan(0)
            if type == 'vanilla_put':
                valuesFFTCallsTraining[i] = model.heston_carr_madan(1)
            for j, parameter in enumerate(model.get_parameters()):
                parametersModelsTraining.iat[i, j] = parameter

        return valuesFFTCallsTraining, parametersModelsTraining

    def test_data_heston_vanillas(amountTest, type):

        modelListTest = []
        valuesFFTCallsTest = pd.DataFrame(index=range(1), columns=range(amountTest))
        parametersModelsTest = pd.DataFrame(index=range(amountTest), columns=range(10))

        for x in range(amountTest):
            # generate pseudo-random numbers for the Training set
            stock_value = 1
            strike = np.random.uniform(0.5, 1.5) * stock_value
            maturity = np.random.uniform(11 / 12, 1)
            interest = np.random.uniform(0.015, 0.025)
            dividend_yield = np.random.uniform(0, 0.05)

            # heston

            kappa = np.random.uniform(1.5, 2.5)
            rho = np.random.uniform(-0.8, -0.6)
            theta = np.random.uniform(0.5, 0.7)
            eta = np.random.uniform(0.02, 0.1)
            sigma0 = np.sqrt(-np.log(np.random.uniform(0.9802, 0.9048)))

            modelListTest.append(
                    models_algorithms.vanilla_option_heston(kappa, eta, theta, rho, sigma0, strike, maturity, stock_value,
                                                        interest, dividend_yield))

        for i, model in enumerate(modelListTest):
            for j, parameter in enumerate(model.get_parameters()):
                parametersModelsTest.iat[i, j] = parameter

        startPredictingOutSampleTimerFFT = timer()
        for i, model in enumerate(modelListTest):
            if type == 'vanilla_call':
                valuesFFTCallsTest[i] = model.heston_carr_madan(0)
            if type == 'vanilla_put':
                valuesFFTCallsTest[i] = model.heston_carr_madan(1)
        endPredictingOutSampleTimerFFT = timer()

        print('Timer of predicting out sample FFT ' + str(
            endPredictingOutSampleTimerFFT - startPredictingOutSampleTimerFFT))

        return valuesFFTCallsTest, parametersModelsTest

    def training_data_heston_down_and_out(amountTraining):

        modelListTraining = []
        valuesDOBPTraining = pd.DataFrame(index=range(1), columns=range(amountTraining))
        parametersModelsTraining = pd.DataFrame(index=range(amountTraining), columns=range(11))

        for x in range(amountTraining):

            # generate pseudo-random numbers for the Training set

            stock_value = 1
            strike = np.random.uniform(0.4, 1.6) * stock_value
            barrier = np.random.uniform(0.55, 0.99) * stock_value
            maturity = np.random.uniform(11 / 12, 1)
            interest = np.random.uniform(0.015, 0.025)
            dividend_yield = np.random.uniform(0, 0.05)

            # heston

            kappa = np.random.uniform(1.4, 2.6)
            rho = np.random.uniform(-0.85, -0.55)
            theta = np.random.uniform(0.45, 0.75)
            eta = np.random.uniform(0.01, 0.1)
            sigma0 = np.sqrt(-np.log(np.random.uniform(0.99, 0.8521)))

            modelListTraining.append(
                models_algorithms.down_and_out_barrier_option_heston(kappa, eta, theta, rho, sigma0, barrier, strike, maturity, stock_value,
                                                        interest, dividend_yield))

        for i, model in enumerate(modelListTraining):
            valuesDOBPTraining[i] = model.monte_Carlo()
            for j, parameter in enumerate(model.get_parameters()):
                parametersModelsTraining.iat[i, j] = parameter

        return valuesDOBPTraining, parametersModelsTraining

    def test_data_heston_down_and_out(amountTest):

        modelListTest = []
        valuesDOBPTest = pd.DataFrame(index=range(1), columns=range(amountTest))
        parametersModelsTest = pd.DataFrame(index=range(amountTest), columns=range(11))

        for x in range(amountTest):
            # generate pseudo-random numbers for the Training set
            stock_value = 1
            strike = np.random.uniform(0.5, 1.5) * stock_value
            barrier = np.random.uniform(0.55, 0.99) * stock_value
            maturity = np.random.uniform(11 / 12, 1)
            interest = np.random.uniform(0.015, 0.025)
            dividend_yield = np.random.uniform(0, 0.05)

            # heston

            kappa = np.random.uniform(1.5, 2.5)
            rho = np.random.uniform(-0.8, -0.5)
            theta = np.random.uniform(0.4, 0.7)
            eta = np.random.uniform(0.02, 0.16)
            sigma0 = np.sqrt(-np.log(np.random.uniform(0.9802,  0.8521)))

            modelListTest.append(
                models_algorithms.down_and_out_barrier_option_heston(kappa, eta, theta, rho, sigma0, barrier, strike, maturity, stock_value,
                                                        interest, dividend_yield))

        for i, model in enumerate(modelListTest):
            for j, parameter in enumerate(model.get_parameters()):
                parametersModelsTest.iat[i, j] = parameter

        startPredictingOutSampleTimerFFT = timer()
        for i, model in enumerate(modelListTest):
            valuesDOBPTest[i] = model.monte_Carlo()
        endPredictingOutSampleTimerFFT = timer()

        print('Timer of predicting out sample FFT ' + str(
            endPredictingOutSampleTimerFFT - startPredictingOutSampleTimerFFT))

        return valuesDOBPTest, parametersModelsTest

class data_generators_american:

    def training_data_american(amountTraining,type):

        modelListTraining = []
        valuesDOBPTraining = pd.DataFrame(index=range(1), columns=range(amountTraining))
        parametersModelsTraining = pd.DataFrame(index=range(amountTraining), columns=range(6))

        for x in range(amountTraining):

            # generate pseudo-random numbers for the Training set

            stock_value = 1
            strike = np.random.uniform(0.4, 1.6) * stock_value
            maturity = np.random.uniform(11 / 12, 1)
            interest = np.random.uniform(0.015, 0.025)
            dividend_yield = np.random.uniform(0, 0.05)
            sigma = np.sqrt(-np.log(np.random.uniform(0.5769, 0.9512)))

            modelListTraining.append(
                models_algorithms.american_option(sigma, strike, maturity, stock_value,
                                                        interest, dividend_yield))

        for i, model in enumerate(modelListTraining):
            if type == 'american_call':
                valuesDOBPTraining[i] = model.binomial_tree_pricing(0)
            if type == 'american_put':
                valuesDOBPTraining[i] = model.binomial_tree_pricing(1)
            for j, parameter in enumerate(model.get_parameters()):
                parametersModelsTraining.iat[i, j] = parameter

        return valuesDOBPTraining, parametersModelsTraining

    def test_data_american(amountTest,type):

        modelListTest = []
        valuesDOBPTest = pd.DataFrame(index=range(1), columns=range(amountTest))
        parametersModelsTest = pd.DataFrame(index=range(amountTest), columns=range(6))

        for x in range(amountTest):
            # generate pseudo-random numbers for the Training set
            stock_value = 1
            strike = np.random.uniform(0.5, 1.5) * stock_value
            maturity = np.random.uniform(11 / 12, 1)
            interest = np.random.uniform(0.015, 0.025)
            dividend_yield = np.random.uniform(0, 0.05)
            sigma = np.sqrt(-np.log(np.random.uniform(0.6065, 0.9048)))

            modelListTest.append(
                models_algorithms.american_option(sigma, strike, maturity, stock_value,
                                                        interest, dividend_yield))

        for i, model in enumerate(modelListTest):
            for j, parameter in enumerate(model.get_parameters()):
                parametersModelsTest.iat[i, j] = parameter

        startPredictingOutSampleTimerFFT = timer()
        for i, model in enumerate(modelListTest):
            if type == 'american_call':
                valuesDOBPTest[i] = model.binomial_tree_pricing(0)
            if type == 'american_put':
                valuesDOBPTest[i] = model.binomial_tree_pricing(1)
        endPredictingOutSampleTimerFFT = timer()

        print('Timer of predicting out sample FFT ' + str(
            endPredictingOutSampleTimerFFT - startPredictingOutSampleTimerFFT))

        return valuesDOBPTest, parametersModelsTest