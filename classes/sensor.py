import numpy as np

class sensor:

    def __init__(self,f_min,f_max,n_samplingfreqs, variances = None, bias = 0) -> None:
        if(variances is None):
            self.variances = np.ones(n_samplingfreqs)
        else:
            self.variances = variances*np.ones(n_samplingfreqs)
        self.Nf = n_samplingfreqs
        self.bias = bias
        self.f_min = f_min
        self.f_max = f_max
        self.delta = (self.f_max -  self.f_min)/self.Nf

    def sample(self, spectrum, samplingpoints):
        #implements WGN measurment model
        samples = np.zeros(len(samplingpoints))
        for s in range(len(samples)):
            samples[s] = spectrum[samplingpoints[s]] + np.random.normal(self.bias,self.variances[samplingpoints[s]])

        return(samples)

    def freq_to_index(self,freqs):
        if(freqs.min() < self.f_min or freqs.max() > self.f_max):
            print("cannot convert frequency to index: out of range")
            return
        index = (freqs - self.f_min)/self.delta
        trunc_index = np.round(index)
        approx_freqs  = trunc_index*self.delta + self.f_min
        print("approximating frequencies as: ", approx_freqs, "truncation error: ", (freqs- approx_freqs))

        return trunc_index

    def index_to_freq(self,indices):
        
        return np.array(indices)*self.delta + self.f_min



        