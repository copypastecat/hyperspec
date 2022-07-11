import numpy as np

class parameter_estimator:

    def __init__(self,method="ls") -> None:
        self.method = method

    def estimate_parameters(self, substances_data, observed_data):
        if(self.method == "ls"):
            return self.ls_est(substances_data,observed_data)

    def ls_est(self, X, Y):
        #compute least squares estimate
        #@X: n_substances x n_frequencies array of true specrum values at sampling points (i.e. mixing matrix in LMM),
        #@Y: n_frequencies vector of observations 
       return np.linalg.lstsq(X,Y,rcond=None)
