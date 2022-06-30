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

    def generate_image(self, approx_share, nclusters):
        #create a underlying state-matrix with one state for each substance to simulate
        state_matrix = np.zeros(self.size)
        for i in range(1,nclusters+1):
           #create a cluster for each substance
           state_matrix = self.make_cluster(state_matrix,approx_share,class_val=(i%len(self.substances)))

        himage = np.zeros((state_matrix.shape[0],state_matrix.shape[1],self.nbins))
        #for each pixel, create a spectral dimension with the corresponding spectral characteristic of the state:
        for x,y in np.ndindex(state_matrix.shape):
            himage[x,y,:] = self.substances[int(state_matrix[x,y])].calculate_radiation(self.light_source)
    
        return state_matrix, himage

    def display():
        pass

    def n_closest(x,n,d=1):
       #not used for now
       return x[n[0]-d:n[0]+d+1,n[1]-d:n[1]+d+1]

    def make_cluster(self,state_matrix, approx_share, class_val=1):
        center_x = floor(abs(np.random.rand(1))*self.size[0])
        center_y = floor(abs(np.random.rand(1))*self.size[1])
        lifespan = round((approx_share*min(self.size)))
        shape = (abs(np.random.normal(1)),abs(np.random.normal(1)))
        print((center_x,center_y))
        
        #generate random ellipsiod-like cluster around center point
        m = 0
        while(lifespan > 0 ):#and center_x + m < self.size[0] and center_x - m > 0): 
            n = 0
            lifespan_y = lifespan - m
            while(lifespan_y > 0 ):#and center_y + n < self.size[1] and center_y - n > 0):
                if(center_x + m < self.size[0]):
                    if(center_y + n < self.size[1]):
                        state_matrix[center_x+m,center_y+n] = class_val
                if(center_x + m < self.size[0]):
                    state_matrix[center_x+m,center_y-n] = class_val
                if(center_y + n < self.size[1]):
                    state_matrix[center_x-m,center_y+n] = class_val
                state_matrix[center_x-m,center_y-n] = class_val
                lifespan_y = lifespan_y - 0.1*shape[0]*n*np.random.rand(1)
                n = n + 1
            m = m + 1
            lifespan = lifespan - 0.1*shape[1]*m*np.random.rand(1)

        return state_matrix