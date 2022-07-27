import numpy as np
import cvxpy as cp

class parameter_estimator:

    def __init__(self,method="ls") -> None:
        self.method = method

    def estimate_parameters(self, substances_data, observed_data):
        if(self.method == "ls"):
            return self.ls_est(substances_data,observed_data)
        if(self.method == "cls"):
            return self.const_ls_est(substances_data,observed_data)
        else:
            print("Only least squares ('ls') or constrained least squares ('cls') are available as estimation methods so far")

    def ls_est(self, X, Y):
        #compute least squares estimate
        #@X: n_substances x n_frequencies array of true specrum values at sampling points (i.e. mixing matrix in LMM),
        #@Y: n_frequencies vector of observations 
       return np.linalg.lstsq(X,Y,rcond=None)

    def const_ls_est(self, X, Y,verbose=False):
        #compute constrained least squares estimate using convex optimization
        #@X: n_substances x n_frequencies array of true specrum values at sampling points (i.e. mixing matrix in LMM),
        #@Y: n_frequencies vector of observations 
        theta = cp.Variable(len(X[1]))
        cost = cp.sum_squares(X @ theta - Y)
        constraints = [cp.sum(theta) == 1, theta >= 0, theta <= 1]
        prob = cp.Problem(cp.Minimize(cost),constraints=constraints)
        prob.solve(verbose=verbose)

        return theta.value

