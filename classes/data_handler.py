import numpy as np
import pandas as pd
import os

class data_handler:

    def __init__(self, datapath, sensors, wavelength_files) -> None:
        self.datapath = datapath
        self.sensors = sensors
        self.sensorwls = {}
        for i, sensor in enumerate(sensors):
            path = os.path.join(self.datapath,wavelength_files[i])
            self.sensorwls[sensor] = np.array(pd.read_csv(path))

    def get_spectrum(self, substance_name, sensor_name):
        freqs = self.sensorwls[sensor_name]
        path = os.path.join(self.datapath, substance_name)
        spec = np.array(pd.read_csv(path))

        return freqs, spec

    def set_datapath(self, newpath):
        self.datapath = newpath
