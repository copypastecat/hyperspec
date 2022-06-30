import numpy as np

class sensor:

    def __init__(self,n_samplingfreqs, variances = None, bias = 0) -> None:
        if(variances is None):
            self.variances = np.ones(n_samplingfreqs)
        else:
            self.variances = variances*np.ones(n_samplingfreqs)
        self.Nf = n_samplingfreqs
        self.bias = bias

    def sample(self, spectrum, samplingpoints):
        #implements WGN measurment model
        samples = np.zeros(len(samplingpoints))
        for s in range(len(samples)):
            samples[s] = spectrum[samplingpoints[s]] + np.random.normal(self.bias,self.variances[samplingpoints[s]])

        return(samples)