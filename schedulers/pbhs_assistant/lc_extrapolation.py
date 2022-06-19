import sys

import numpy as np
import time
import matplotlib.pyplot as plt
from pybnn.lc_extrapolation.learning_curves import MCMCCurveModelCombination

def extrapolate(observed, n_epochs, acc):
   
    t_idx = np.arange(1, observed+1)

    model = MCMCCurveModelCombination(n_epochs + 1,
                                    nwalkers=50,
                                    nsamples=800,
                                    burn_in=500,
                                    recency_weighting=False,
                                    soft_monotonicity_constraint=False,
                                    monotonicity_constraint=True,
                                    initial_model_weight_ml_estimate=True)
    st = time.time()
    if (model.fit(t_idx, acc[:observed]) == False):
        return [0.001, 0]

    
    # print("Training time: %.2f" % (time.time() - st))
    # st = time.time()
    # p_greater = model.posterior_prob_x_greater_than(n_epochs + 1, .5)
    

    # m = np.zeros([n_epochs])
    # s = np.zeros([n_epochs])

    # for i in range(n_epochs):
    #     p = model.predictive_distribution(i+1)
    #     m[i] = np.mean(p)
    #     s[i] = np.std(p)
    result = model.predictive_distribution(n_epochs)
    mean_mcmc = np.mean(result)
    std_mcmc = np.std(result)
    print("Prediction time: %.2f" % (time.time() - st))
    return [mean_mcmc, std_mcmc]

