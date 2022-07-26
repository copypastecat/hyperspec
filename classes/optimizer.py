
from math import *
from jsonschema import RefResolutionError
from matplotlib.pyplot import sci
from classes.substance import substance
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import scipy.integrate as integrate


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
            return(-metric(self.calculate_FIM(fs,interpolate=True)))# - np.abs(np.sum(np.diff(fs))))
        
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
        solution = opt.brute(optimization_target,ranges=bounds, Ns=self.n_sim_freqs,full_output=verbose, finish=None)
        
        return solution

    def find_freqs_brute_GBD(self,N,verbose=False):
        #find optimal solution using generalized Bhattacharyya distance
        if(N > 3):
            print("Too many sampling frequencies for numerical integration of BC!")
            return None
        
        def optimization_target(fs):
            return(-self.generalized_Bhattacharyya_distance(fs))
        
        bound = (0,self.n_sim_freqs-1)
        bounds = []
        for i in range(N):
            bounds.append(bound)
        print(bounds)
        solution = opt.brute(optimization_target,ranges=bounds, Ns=self.n_sim_freqs,full_output=verbose)

        return solution

    def find_freqs_minokwski_approx(self,N,verbose=False):
        #find approximation of optimal frequencies using det(A+B) >= det(A) + det(B)
        #arising from Minkowski's inequality (approximate D-optimal design)
        I_f = []
        for n in range(self.n_sim_freqs):
            I = self.calculate_FIM([n])
            I_f.append(np.linalg.det(I))
            #print("FIM: ", I, "with determinant ", np.linalg.det(I))
        
        I_f = np.array(I_f)
        max_ind = np.argpartition(I_f, -N)[-N:]

        return max_ind


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

    def generalized_Bhattacharyya_distance(self, sampling_frequencies):
        if(len(sampling_frequencies) > 3): 
            print("too many dimensions for numerical integration, try less sampling frquencies")
            return None

        pdfs = []
        n_classes = len(self.substances)
        sampling_frequencies = sampling_frequencies.astype(int)
        for s in self.substances:
            pdf = stats.multivariate_normal(s.radiation_pattern[sampling_frequencies], self.sensor.variances[sampling_frequencies])
            pdfs.append(pdf)
        if(len(sampling_frequencies) == 1):
            f = lambda x: np.prod([pdf.pdf(x) for pdf in pdfs])
            BC = integrate.quad(f,-np.inf,np.inf)
        if(len(sampling_frequencies) == 2):
            f = lambda x1, x2: np.prod([pdf.pdf([x1,x2]) for pdf in pdfs])
            BC = integrate.dblquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
        if(len(sampling_frequencies) == 3):
            f = lambda x1, x2, x3: np.prod([pdf.pdf([x1,x2,x3]) for pdf in pdfs])
            BC = integrate.tplquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, lambda x,y: -np.inf, lambda x,y: np.inf)

        return -log(BC[0])


