import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# This code executes a basic polynomial regression

class basicregression:

    def __init__(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def prediction(self,X_test):

        pred = self.pol_reg.predict(self.poly.fit_transform(X_test))
        return pred

    def fitting(self):

        self.poly = PolynomialFeatures(degree=2)
        X_poly = self.poly.fit_transform(self.X_train)
        self.pol_reg = LinearRegression()
        self.pol_reg.fit(X_poly, self.y_train)