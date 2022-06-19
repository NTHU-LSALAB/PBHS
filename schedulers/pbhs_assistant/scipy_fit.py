import sys
import numpy as np
import math
from scipy.optimize import curve_fit

def logrithm(x, a, b):
    
    return a * np.log(x) + b



def scipy_fit(observed, n_epochs, acc):
   
    x_data = np.arange(1, observed+1)
    y_data = acc[:observed]
    
    popt, pcov = curve_fit(logrithm, x_data, y_data)
    # print (popt)

    prediction = logrithm(n_epochs, popt[0], popt[1])

    return [prediction, 1]
