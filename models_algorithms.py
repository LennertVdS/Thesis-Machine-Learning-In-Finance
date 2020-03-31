import numpy as np
import pandas as pd
import math
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import datetime



"""
In this file, we calculate the prices of options 
"""


class model:

    # initialise the constants of the model
    def __init__(self, strike, stock_value, maturity, interest, dividend_yield):
        self.strike = strike
        self.stock_value = stock_value
        self.maturity = maturity
        self.interest = interest
        self.dividend_yield = dividend_yield


class vanilla_option_heston(model):

    # initialise the constants of the model
    # inherit from model
    def __init__(self, kappa, eta, theta, rho, sigma0, strike, maturity, stock_value, interest, dividend_yield):
        super().__init__(strike, stock_value, maturity, interest, dividend_yield)
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        self.eta = eta
        self.sigma0 = sigma0

    # getter
    def get_parameters(self):
        return self.kappa, self.eta, self.theta, self.rho, self.sigma0, self.strike, self.maturity, self.stock_value, \
               self.interest, self.dividend_yield

    # pre : value u to be fitted in the characteristic
    # post : the calculation of the characteristic of the heston model
    def heston_characteristic(self, u):

        kappa, eta, theta, rho, sigma0, K, t, S0, r, q = self.get_parameters()
        d = np.sqrt((rho*theta*u*np.complex(0, 1)-kappa)**2-theta**2*(-np.complex(0, 1)*u-u**2))
        g = (kappa-rho*theta*u*np.complex(0, 1)-d)/(kappa-rho*theta*u*np.complex(0, 1)+d)

        p1 = np.complex(0,1)*u*(np.log(S0)+(r-q)*t)
        p2 = eta*kappa*theta**(-2)*((kappa-rho*theta*u*np.complex(0,1)-d)*t-2*np.log((1-g*np.exp(-d*t))/(1-g)))
        p3 = sigma0**2*theta**(-2)*(kappa-rho*theta*u*np.complex(0,1) - d)*(1 - np.exp(-d*t))/(1-g*np.exp(-d*t))

        y = np.exp(p1)*np.exp(p2)*np.exp(p3)

        return y

    def heston_carr_madan(self, type):

        kappa, eta, theta, rho, sigma0, K, T, S0, r, q = self.get_parameters()

        # type = 0 --> Call option
        # type = 1 --> Put option
        # define parameters
        N = 4096
        alpha = 1.5
        eta_grid = 0.25
        lmbda = 2*math.pi/(N*eta_grid)
        b = lmbda*N/2

        # define grid of log-strikes
        k = np.arange(-b, b, lmbda)

        # compute Rho
        v = np.arange(0, N*eta_grid,eta_grid)
        u = v-(alpha+1)*np.complex(0, 1)
        Rho = np.exp(-r*T)*self.heston_characteristic(u)/(alpha**2+alpha-v**2+np.complex(0, 1)*(2*alpha+1)*v)

        # Use Simpson's rule for good accuracy
        simpson_1 = (1/3)
        simpson = ((3 + (-1)**np.arange(2, N+1,1))/3)
        simpson_int = np.append(simpson_1, simpson)

        # The calculation of the FFT
        a = fft(Rho*np.exp(np.complex(0,1)*v*b)*eta_grid*simpson_int, N).real

        # Calculation and interpolation of Calls
        callPrices = (1/math.pi)*np.exp(-alpha*k)*a
        KK = np.exp(k)
        y = interp1d(KK,callPrices,"cubic")(K)

        # Modification for puts
        if type == 1:
            y = y + K*np.exp(-r*T)- np.exp(-q*T)*S0

        return float(y)

class vanilla_option_VG(model):

    # initialise the constants of the model
    # inherit from model
    def __init__(self, nu, theta, sigma, strike, maturity, stock_value, interest, dividend_yield):
        super().__init__(strike, stock_value, maturity, interest, dividend_yield)
        self.nu = nu
        self.theta = theta
        self.sigma = sigma

    # getter
    def get_parameters(self):
        return self.nu, self.theta, self.sigma, self.strike, self.maturity, self.stock_value, \
               self.interest, self.dividend_yield

    # pre : value u to be fitted in the characteristic
    # post : the calculation of the characteristic of the heston model
    def vg_characteristic(self, u):

        nu, theta, sigma, K, t, S0, r, q = self.get_parameters()

        C = 1/nu
        G = 2/((((theta**2) * (nu**2) )+ (2*(sigma**2) * nu))**(0.5) - (theta*nu))
       # G = ((((theta**2) * (nu**2) / 4 )+ ((sigma**2) * nu /2))**(0.5) - (theta*nu/2))**(-1)
        M = 2 / ((((theta ** 2) * (nu ** 2)) + (2 * (sigma ** 2) * nu)) ** (0.5) + (theta * nu))
       #M = ((((theta ** 2) * (nu ** 2) / 4) + ((sigma ** 2) * nu / 2))**(0.5) + (theta * nu / 2)) ** (-1)

        omega = nu**(-1) * np.log(1-0.5*(sigma**2)*nu - theta*nu)
        y = np.exp(np.complex(0,1) * u * np.log(S0) + (r-q+omega)*t) * ((G*M)/(G*M + (M-G)*np.complex(0,1)*u + u**2))**(C*t)
        return y

    def vg_carr_madan(self, type):

        nu, theta, sigma, K, T, S0, r, q = self.get_parameters()

        # type = 0 --> Call option
        # type = 1 --> Put option
        # define parameters
        N = 4096
        alpha = 1.5
        eta_grid = 0.25
        lmbda = 2*math.pi/(N*eta_grid)
        b = lmbda*N/2

        # define grid of log-strikes
        k = np.arange(-b, b, lmbda)

        # compute Rho
        v = np.arange(0, N*eta_grid,eta_grid)
        u = v-(alpha+1)*np.complex(0, 1)
        Rho = np.exp(-r*T)*self.vg_characteristic(u)/(alpha**2+alpha-v**2+np.complex(0, 1)*(2*alpha+1)*v)

        # Use Simpson's rule for good accuracy
        simpson_1 = (1/3)
        simpson = ((3 + (-1)**np.arange(2, N+1,1))/3)
        simpson_int = np.append(simpson_1, simpson)

        # The calculation of the FFT
        a = fft(Rho*np.exp(np.complex(0,1)*v*b)*eta_grid*simpson_int, N).real

        # Calculation and interpolation of Calls
        callPrices = (1/math.pi)*np.exp(-alpha*k)*a
        KK = np.exp(k)
        y = interp1d(KK,callPrices,"cubic")(K)

        # Modification for puts
        if type == 1:
            y = y + K*np.exp(-r*T)- np.exp(-q*T)*S0

        return float(y)

class american_option(model):

    # initialise the constants of the model
    # inherit from model
    def __init__(self, sigma, strike, maturity, stock_value, interest, dividend_yield):
        super().__init__(strike, stock_value, maturity, interest, dividend_yield)
        self.sigma = sigma

    def get_parameters(self):
        return self.sigma, self.strike, self.maturity, self.stock_value, self.interest, self.dividend_yield

    def binomial_tree_pricing(self,  type, dt):

        print(datetime.datetime.now())

        sigma, K, T, S0, r, q = self.get_parameters()

        #matrix dimension
        n = int(round(T/dt))

        #going up
        u = np.exp(np.sqrt(sigma*dt))

        #and down
        d = np.exp(-np.sqrt(sigma*dt))

        # risk - neutral probability
        p = (np.exp((r-q)*dt) - d) / (u - d)

        # discount factor
        df = np.exp((-r-q) * dt);

        #create empty matrices
        S = pd.DataFrame(0,index=range(n+1), columns=range(n+1))
        A = pd.DataFrame(0,index=range(n+1), columns=range(n+1))

        # fill the last column with stock prices at maturity
        for i in range(1,n+2,1):
            S.iloc[i-1, n] = S0 * (u**(n-i+1)) * d**(i-1)

        # calculate payoffs at maturity
        if type == 0:
            A.iloc[:, n] = np.maximum(S.iloc[:,n]-K, 0)  # call
        else:
            A.iloc[:, n] = np.maximum(K - S.iloc[:, n ], 0)  # put

        for j in range(n, 0, -1):
            for i in range(1,j+1,1):
                A.iloc[i-1, j-1] = df * (p * A.iloc[i-1,j] + (1-p) * A.iloc[i,j])
                S.iloc[i-1, j-1] = S0 * (u**(j-i) * (d**(i-1))) # intermediate stock prices
                if type == 0:
                    A.iloc[i-1, j-1] = np.maximum(A.iloc[i-1, j-1], S.iloc[i-1, j-1] - K)  # call
                else:
                    A.iloc[i-1, j-1] = np.maximum(A.iloc[i-1, j-1], K - S.iloc[i-1, j-1])  # put

        return(A.iloc[0, 0])



class down_and_out_barrier_option_heston(model):

    # initialise the constants of the model
    # inherit from model
    def __init__(self, kappa, eta, theta, rho, sigma0, barrier, strike, maturity, stock_value, interest, dividend_yield):
        super().__init__(strike, stock_value, maturity, interest, dividend_yield)
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        self.eta = eta
        self.sigma0 = sigma0
        self.barrier = barrier

    # getter
    def get_parameters(self):
        return self.kappa, self.eta, self.theta, self.rho, self.sigma0, self.barrier, self.strike, self.maturity, self.stock_value, \
               self.interest, self.dividend_yield


    def monte_Carlo(self, m, dt):

        kappa, eta, theta, rho, sigma0, H, K, T, S0, r, q = self.get_parameters()
        print(datetime.datetime.now())

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

# class down_and_out_barrier_option_vg(model):
#
#     # initialise the constants of the model
#     # inherit from model
#     def __init__(self, nu, theta, sigma, barrier, strike, maturity, stock_value, interest, dividend_yield):
#         super().__init__(strike, stock_value, maturity, interest, dividend_yield)
#         self.nu = nu
#         self.theta = theta
#         self.sigma = sigma
#         self.barrier = barrier
#
#     # getter
#     def get_parameters(self):
#         return self.nu, self.theta, self.sigma, self.barrier, self.strike, self.maturity, self.stock_value, \
#                self.interest, self.dividend_yield
#
#     def monte_Carlo(self,m, dt):
#
#         nu, theta, sigma, H, K, T, S0, r, q = self.get_parameters()
#
#         # paths
#         #m = 10000
#
#         #time steps
#         #dt= 1/250
#
#         #matrix dimension
#         n = int(round(T/dt))
#
#         omega = nu ** (-1) * np.log(1 - 0.5 * (sigma ** 2) * nu - theta * nu)
#
#         eps = np.random.randn(m, n)
#         g = np.random.gamma(dt / nu, nu,(m,n))
#         vg = pd.DataFrame(0, index=range(m), columns=range(n + 1))
#
#         vg.iloc[0,0] = 0
#
#         for s in range(n):
#
#             vg.iloc[:,s+1] = vg.iloc[:,s] + theta*g[:,s] + sigma*np.sqrt(g[:,s]) * eps[:,s]
#
#         S = S0*np.exp((r-q + omega) * dt +vg)
#
#         DOBP_path = np.maximum(((S.min(axis =1) - H)/ (np.abs(S.min(axis =1) - H))),0)  * np.maximum((K - S.iloc[:,n]) * np.exp(-r*T), 0)
#
#         DOBP = DOBP_path.mean()
#
#         return DOBP



#test= down_and_out_barrier_option_vg(0.1,-0.1,0.3,5,100,1,50,0.01,0.01)
#print(test.vg_carr_madan(0))
#print(test.monte_Carlo())

    # def monte_Carlo_pricing(self):
    #     if __name__ == '__main__':
    #         freeze_support()
    #         DOBP = np.mean((Pool().map(self.monte_Carlo, [12500, 12500, 12500, 12500])))
    #         return DOBP