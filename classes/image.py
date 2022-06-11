#class for hyperspectral image at simulation frequency (oversampled)
from math import floor
from .substance import substance
from .light_source import light_source
import numpy as np

class image:

    def __init__(self, size, nbins, substances, light_source) -> None:
        self.size = size
        self.nbins = nbins
        self.substances = substances
        self.light_source = light_source

    def generate_image(self, approx_share):
        state_matrix = np.zeros(self.size)
        center_x = floor(abs(np.random.rand(1))*self.size[0])
        center_y = floor(abs(np.random.rand(1))*self.size[1])
        lifespan = round((approx_share*min(self.size)))
        shape = (abs(np.random.normal(1)),abs(np.random.normal(1)))
        print((center_x,center_y))
        
        #generate cluster around center point
        m = 0
        while(lifespan > 0 ):#and center_x + m < self.size[0] and center_x - m > 0): 
            n = 0
            lifespan_y = lifespan - m
            while(lifespan_y > 0 ):#and center_y + n < self.size[1] and center_y - n > 0):
                if(center_x + m < self.size[0]):
                    if(center_y + n < self.size[1]):
                        state_matrix[center_x+m,center_y+n] = 1
                if(center_x + m < self.size[0]):
                    state_matrix[center_x+m,center_y-n] = 1
                if(center_y + n < self.size[1]):
                    state_matrix[center_x-m,center_y+n] = 1
                state_matrix[center_x-m,center_y-n] = 1
                lifespan_y = lifespan_y - 0.1*shape[0]*n*np.random.rand(1)
                n = n + 1
            m = m + 1
            lifespan = lifespan - 0.1*shape[1]*m*np.random.rand(1)

        return state_matrix

    def display():
        pass

    def n_closest(x,n,d=1):
       return x[n[0]-d:n[0]+d+1,n[1]-d:n[1]+d+1]