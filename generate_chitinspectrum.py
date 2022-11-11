import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

#parameters
lambda_min = 0.25
lambda_max = 0.9
peak_location = 0.41
width = 0.04
n_sim_freqs = 20000
scale_factor = 0.05
bias = 0.05

x = np.linspace(lambda_min,lambda_max,n_sim_freqs)
y = scale_factor*norm(loc=peak_location,scale=width).pdf(x=x) + bias

df = pd.DataFrame(y)
df.to_csv("usgs_selected/chitin_10layer_75mm.txt",sep='\n',index=None,header=False)
xs = pd.DataFrame(x)
xs.to_csv("usgs_selected/wavelengths_chitin.txt",sep='\n',index=None,header=False)

'''
plt.plot(x,y)
plt.ylim(0,0.6)
plt.xlim(0.25,0.9)
plt.show()
'''

