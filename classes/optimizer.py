
from math import *
from jsonschema import RefResolutionError
from classes.substance import substance
import numpy as np
import scipy.optimize as opt


class optimizer:

    def __init__(self, substances, sensor, n_sim_freqs, light_source = None) -> None:
        self.substances = substances
        self.light_source = light_source
        self.n_sim_freqs = n_sim_freqs
        self.sensor = sensor

    def find_freqs(self,N,criterion):
        initial_guess = np.random.choice(self.n_sim_freqs,N)
        if(criterion == "D"):
            metric = self.D_optimality_criterion
        elif(criterion == "A"):
            metric = self.A_optimality_criterion
        elif(criterion == "E"):
            metric = self.E_optimality_criterion
        else:
            print("criterion not valid (A,E or D available), choosing D-optimality!")
            metric = self.D_optimality_criterion

        def optimization_target(fs):
            return(-metric(self.calculate_FIM(fs,interpolate=True)) - np.abs(np.sum(np.diff(fs))))
        
        bound = (0,self.n_sim_freqs-1)
        bounds = []
        for i in range(N):
            bounds.append(bound)
        solution = opt.dual_annealing(optimization_target,bounds=bounds, x0=initial_guess)
        #solution = opt.brute(optimization_target,ranges=bounds)
        
        #return solution
        return solution.x, solution.fun

    
    def find_freqs_brute(self,N,criterion,verbose=False):
        if(criterion == "D"):
            metric = self.D_optimality_criterion
        elif(criterion == "A"):
            metric = self.A_optimality_criterion
        elif(criterion == "E"):
            metric = self.E_optimality_criterion
        else:
            print("criterion not valid (A,E or D available), choosing D-optimality!")
            metric = self.D_optimality_criterion

        def optimization_target(fs):
            return(-metric(self.calculate_FIM(fs)))# - np.abs(np.sum(np.diff(fs))))
        
        bound = (0,self.n_sim_freqs-1)
        bounds = []
        for i in range(N):
            bounds.append(bound)
        print(bounds)
        solution = opt.brute(optimization_target,ranges=bounds, Ns=self.n_sim_freqs,full_output=verbose)
        
        return solution

    def calculate_FIM(self, sampling_frequencies, interpolate=False):
        #calculate the Fisher Information Matrix for a Gaussian data model assuming linear spectral mixing with 
        # abundance non-negativity constraint (ANC) and abundance sum constraint (ASC) 
        S = len(self.substances)
        variance = self.sensor.variances
        FIM = np.zeros((S-1,S-1))
        for i in range(S-1):
            for j in range(S-1):
                for k in sampling_frequencies:
                    #to handle continuous frequencies: linearily interpolate between sampling points
                    #enables use of continuous optimization methods without rounding...
                    if(interpolate):
                       upper_k = ceil(k)
                       lower_k = floor(k)
                       residual = k - lower_k
                       phi_i = residual*self.substances[i].radiation_pattern[lower_k] + (1-residual)*self.substances[i].radiation_pattern[upper_k]
                       phi_j = residual*self.substances[j].radiation_pattern[lower_k] + (1-residual)*self.substances[j].radiation_pattern[upper_k]
                       phi_s = residual*self.substances[-1].radiation_pattern[lower_k] + (1-residual)*self.substances[-1].radiation_pattern[upper_k]
                       var_k = residual*variance[lower_k] + (1-residual)*variance[upper_k]
                    else:
                        phi_i = self.substances[i].radiation_pattern[int(k)]
                        phi_j = self.substances[j].radiation_pattern[int(k)]
                        phi_s = self.substances[-1].radiation_pattern[int(k)]
                        var_k = variance[int(k)] 
                    FIM[i,j] = FIM[i,j] + (1/var_k)*(phi_i*phi_j - phi_s*(phi_i + phi_j) + (phi_s)**2)
                
        return FIM

    def D_optimality_criterion(self,FIM):
        return np.linalg.det(FIM)

    def E_optimality_criterion(self,FIM):
        return max(np.linalg.eig(FIM)[0])

    def A_optimality_criterion(self,FIM):
        return np.trace(FIM)
