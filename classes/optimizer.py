
from math import log

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
            return(-metric(self.calculate_FIM(np.round(fs).astype(int))))
        
        bound = (0,N)
        bounds = []
        for i in range(N):
            bounds.append(bound)
        solution = opt.dual_annealing(optimization_target,bounds=bounds, no_local_search=True, x0=initial_guess)
        #solution = opt.brute(optimization_target,ranges=bounds)
        
        #return solution
        return solution.x, solution.fun


    def calculate_FIM(self, sampling_frequencies):
        S = len(self.substances)
        variance = self.sensor.variances
        FIM = np.zeros((S,S))
        for i in range(S):
            for j in range(S):
                for k in sampling_frequencies:
                    FIM[i,j] = FIM[i,j] + (1/variance[k])*self.substances[i].radiation_pattern[k]*self.substances[j].radiation_pattern[k]
                    #print("elements at i=",i,"j=",j," :",(1/variance[k])*self.substances[i].radiation_pattern[k]*self.substances[j].radiation_pattern[k])
                
        return FIM

    def D_optimality_criterion(self,FIM):
        return np.linalg.det(FIM)

    def E_optimality_criterion(self,FIM):
        return max(np.linalg.eig(FIM)[0])

    def A_optimality_criterion(self,FIM):
        return np.trace(FIM)
