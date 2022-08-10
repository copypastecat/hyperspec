
from math import *
from tkinter import N
from classes.substance import substance
import numpy as np
import scipy.optimize as opt


class optimizer:

    def __init__(self, substances, sensor, n_sim_freqs, light_source = None) -> None:
        self.substances = substances
        self.light_source = light_source
        self.n_sim_freqs = n_sim_freqs
        self.sensor = sensor
        self.DFBB_lower = -np.inf
        self.DFBB_vbest = np.zeros(n_sim_freqs)
        S = len(substances)
        self.FIMs = np.zeros((S-1,S-1,n_sim_freqs))
        self.FIMs_already_calculated = False
        #pre-calculate single-frequency FIMs
        self.calculate_Iks()

    def calculate_Iks(self):
        if(not self.FIMs_already_calculated):
            for i in range(self.n_sim_freqs):
                self.FIMs[:,:,i] = self.calculate_FIM([i])


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
        #print(bounds)
        solution = opt.brute(optimization_target,ranges=bounds, Ns=self.n_sim_freqs,full_output=verbose, finish=None)
        
        return solution

    def find_freqs_brute_GBD(self,N,verbose=False):
        #find optimal solution using generalized Bhattacharyya distance
        
        def optimization_target(fs):
            return(-self.generalized_Bhattacharyya_distance(fs.astype(int)))
        
        bound = (0,self.n_sim_freqs-1)
        bounds = []
        for i in range(N):
            bounds.append(bound)
        #print(bounds)
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

    def find_freqs_DFBB(self,N,verbose=False):
        # wrapper for recursive BB method

        # start recursion form BB-trees root (empty sets for I_0, I_1):
        self.recursive_DFBB([],[],N)
        indicies = np.where(self.DFBB_vbest == 1)[0]

        return indicies, np.linalg.det(np.sum(self.FIMs[:,:,indicies],axis=2))

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
        # calculates generalized Bhattacharyya-Distance for S classes, using a WGN model for N sampling frequencies
        # the WGN modelling assumption leads to equal covariance matricies for all classes. Each class-mean corresponds
        # to the value of the reflectance spectrum of the substance associated with the class.
        S = len(self.substances)
        N = len(sampling_frequencies)
        mus = np.zeros((S,N))
        for i, substance in enumerate(self.substances):
           mus[i,:] = substance.radiation_pattern[sampling_frequencies].T
        mus = mus.T
        
        A = np.matrix(self.sensor.variances[sampling_frequencies] * np.eye(N)) #assuming equal cov-matr. for all classes
        theta_temp = np.matrix((1/S)*np.sum(np.linalg.inv(A) * mus,axis=1))
        theta = A * theta_temp
        GBD = (1/2) * (-(theta.T*np.linalg.inv(A)*theta) + ((1/S)*np.sum(mus.T * np.linalg.inv(A) * mus))) #log term disappears due to equal covs

        return float(GBD)


    def recursive_DFBB(self,I_1,I_0,N):
        # depth-first branch and bound det-FIM optimization method implemented after Ucinski, Patan: 
        # D-optimal design of a monitoring network for parameter estimation of distributed systems (2007). 
        # Used helper functions are defined below
        if((len(I_0) > self.n_sim_freqs-N) or len(I_1) > N):
            return 
        if(self.singularity_test(I_0,I_1,N)):
            return
        v_relaxed, det_FIM = self.relaxed_solution(I_0,I_1,N)
        #print("Relaxed solution: ", v_relaxed)
        if(det_FIM < self.DFBB_lower):
            #prune
            return
        elif(self.integral_test(v_relaxed)):
            #bound
            self.DFBB_vbest = v_relaxed
            self.DFBB_lower = det_FIM
        else:
            #branch
            i_star = self.index_branch(v_relaxed)
            self.recursive_DFBB(I_1=(I_1+[i_star]),I_0=I_0,N=N)
            self.recursive_DFBB(I_1=I_1,I_0=(I_0+[i_star]),N=N)

    def greedy_minkowski(self,N):
        #approximate D-optimal design using Minkowskis Inequality
        #and a Greedy strategy to drastically reduce the tree search complexity
        S = len(self.substances)

        self.calculate_Iks()
        M_sofar = np.zeros((S-1,S-1))
        indicies = []
        for k in range(N):
            index = -1
            max_det = 0
            max_trace = 0
            maxtrace_index = -1
            prev_rank = 0
            max_rank_index = -1
            max_rank_increase = 0
            for i in range(len(self.FIMs[0,0,:])):
                det = np.linalg.det(M_sofar + self.FIMs[:,:,i])
                trace = np.trace(M_sofar  + self.FIMs[:,:,i])
                rank_increase = np.linalg.matrix_rank(M_sofar + self.FIMs[:,:,i]) - prev_rank
                if((det > max_det) and i not in indicies):
                    index = i
                    max_det = det
                if((trace > max_trace) and i not in indicies):
                    maxtrace_index = i
                    max_trace = trace
                if((rank_increase > max_rank_increase) and i not in indicies):
                    max_rank_index = i
                    max_rank_increase = rank_increase
            if(index == -1):
                if(max_rank_index == -1):
                    index = maxtrace_index
                else:
                    index = max_rank_index
            if(index == -1):
                print("all measures are zero for k =", k, "picking random sampling point")
                index = np.random.choice(self.n_sim_freqs)
            indicies.append(index) #add new frequency to selected sampling points
            M_sofar = M_sofar + self.FIMs[:,:,index] #add FIM to FIM-sum

        return np.array(indicies), np.linalg.det(M_sofar)


        
#-----------------BRANCH-AND-BOUND INTERNAL HELPER FUNCTIONS-----------------------------------

    def singularity_test(self,I_0,I_1,N):
        rel_ind = [value for value in range(self.n_sim_freqs) if (value not in I_0) and (value not in I_1)] #indicies to relax
        r = N - len(I_1)
        q = len(rel_ind)
        test_matrix = np.sum(self.FIMs[:,:,I_1],axis=2) + (r/q)*np.sum(self.FIMs[:,:,rel_ind],axis=2)

        return (np.linalg.det(test_matrix) == 0)

    def relaxed_solution(self,I_0,I_1,N):
        #find optimal solution to relaxed problem
        #calculate optimization constants:
        w_ind = [value for value in range(self.n_sim_freqs) if (value not in I_0) and (value not in I_1)]
        q = len(w_ind)
        r = N - len(I_1)
        c = self.FIMs[:,:,w_ind]
        A = np.vstack([np.ones(q),np.eye(q),-np.eye(q)])
        b = np.hstack([r,np.ones(q),np.zeros(q)]).T

        def target(x):
            M = np.sum(self.FIMs[:,:,I_1],axis=2) + np.sum(x*self.FIMs[:,:,w_ind],axis=2)
            return -np.log(np.linalg.det(M))

        #define and solve LP
        start = (r/q)*np.ones(q)
        constraints = opt.LinearConstraint(A=A,ub=b,lb=-inf)
        solution = opt.minimize(target, x0=start, method='SLSQP',constraints=constraints)

        #construct v_relaxed
        v_relaxed = np.zeros(self.n_sim_freqs)
        v_relaxed[I_1] = 1
        v_relaxed[w_ind] = solution.x
        det_FIM = solution.fun

        return v_relaxed, det_FIM #np.linalg.det(np.sum(self.FIMs[:,:,I_1] + np.sum(x.value*self.FIMs[:,:,w_ind],axis=2),axis=2))

    def integral_test(self, v):
        arr_1 = v[v!=0]
        arr_notint = arr_1[arr_1!=1]
        return (len(arr_notint) == 0)

    def index_branch(self, v):
        return np.argmin(abs(v - 0.5))

