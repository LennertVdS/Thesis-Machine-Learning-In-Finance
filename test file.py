import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.interpolate import interp1d

import multiprocessing as mp
from multiprocessing import freeze_support
from pathos.multiprocessing import ProcessingPool as Pool





def howmany_within_range(m):
    import numpy as np
    import pandas as pd
    kappa = 1
    eta = 0.1
    theta = 0.5
    rho = -0.55
    sigma0 = 0.14
    H = 0.2
    K = 1.5
    T = 1
    S0 = 1
    r = 0.05
    q = 0.05

    #time steps
    dt= 1/250

    #matrix dimension
    n = int(round(T/dt))

    S = pd.DataFrame(0, index=range(m), columns=range(n + 1))
    v = pd.DataFrame(0, index=range(m), columns=range(n + 1))

    eps = np.random.randn(m, n)
    epsS = np.random.randn(m, n)

    eps1 = eps
    eps2 = eps * rho + np.sqrt(1 - rho**2) * epsS
    S.iloc[:, 0] = S0
    v.iloc[:, 0] = sigma0 ** 2

    for j in range(1,n + 1,1):
        S.iloc[:, j] = S.iloc[:,j-1]*(1 + (r - q) * dt + np.sqrt(v.iloc[:,j-1])*np.sqrt(dt) * eps1[:,j-1])
        v.iloc[:, j] = abs(v.iloc[:, j-1]+ (kappa * (eta - v.iloc[:, j-1]) - (theta ** 2) / 4)*dt +
                           theta* np.sqrt(v.iloc[:,j-1])*np.sqrt(dt) * eps2[:,j-1]+theta ** 2 * dt * (eps2[:,j-1] ** 2) / 4)

    DOBP_path = np.maximum(((S.min(axis =1) - H)/ (np.abs(S.min(axis =1) - H))),0)  * np.maximum((K - S.iloc[:,n]) * np.exp(-r*T), 0)

    DOBP = DOBP_path.mean()

    return DOBP



if __name__ == '__main__':
    freeze_support()
    a = np.mean((Pool().map(howmany_within_range, [2500,2500,2500,2500])))
    print(a)





