import numpy as np
import pandas as pd
from scipy import signal
import os

class data_handler:

    def __init__(self, datapath, sensors, wavelength_files, nsamplingpoints) -> None:
        self.datapath = datapath #path to dataset
        self.sensors = sensors #what sensors are used for the dataset (for range/res info)
        self.nsamplingpoints = nsamplingpoints #sampling granularity in the simulation
        self.sensorwls = {} #to fill below
        for i, sensor in enumerate(sensors):
            path = os.path.join(self.datapath,wavelength_files[i])
            self.sensorwls[sensor] = np.array(pd.read_csv(path)) #put the actual  sampling frequencies of the sensors into a dict

    def get_spectrum(self, substance_name, sensor_name):
        freqs = self.sensorwls[sensor_name] #freq vals for the specific sensor used to collect the data
        path = os.path.join(self.datapath, substance_name) 
        spec = np.array(pd.read_csv(path))

        return freqs, spec

    def set_datapath(self, newpath):
        #default setter
        self.datapath = newpath

    def equalize_spectra(self, specs, sensor_names):
        #if different sensors are used to collect spectra, these must be casted to the same interval and resolution 
        #in order to be usable in the simulation
        #side-effect: possibility to precisely set the simulations frequency resolution through resampling
        max = 1e400
        min = 0
        for sensor in sensor_names:
            this_min = self.sensorwls[sensor].min()
            this_max = self.sensorwls[sensor].max()
            if(this_min > min):
                min = this_min
            if(this_max < max):
                max = this_max
        #min_range = np.linspace(min,max,self.nsamplingpoints)
        specs_res = []
        for i, spec in enumerate(specs):
            min_index = np.argwhere(self.sensorwls[sensor_names[i]] > min)[0,0]
            max_index = np.argwhere(self.sensorwls[sensor_names[i]] < max)[-1,0]
            #print(min_index)
            #print(max_index)
            spec = spec[int(min_index):int(max_index)]
            spec_res = signal.resample(spec, self.nsamplingpoints)
            #print(spec_res)
            specs_res.append(spec_res)

        return np.linspace(min,max,self.nsamplingpoints), specs_res