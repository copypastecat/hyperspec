#dummy main file for running simulations
from classes import image
from classes import substance
from classes import data_handler
import numpy as np
import matplotlib.pyplot as plt

from classes.light_source import light_source

n_sim_freqs = 200

abs_conc = np.log(np.arange(n_sim_freqs)+1)
concrete = substance.substance("concrete", abs_conc)

abs_water = np.ones(n_sim_freqs)
water = substance.substance("water", abs_water)

data_handler = data_handler.data_handler("../datasets/usgs_selected",["BECK","ASD"],["splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt","splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"])
freqs_water, spec_water = data_handler.get_spectrum("splib07a_Seawater_Coast_Chl_SW1_BECKa_AREF.txt","BECK")

rad_sun = np.ones(n_sim_freqs) + 0.04*np.random.rand(n_sim_freqs)
sun = light_source("sun", rad_sun)

this_image = image.image((128,128), n_sim_freqs, [concrete, water], sun)

state_matrix, himage = this_image.generate_image(approx_share=1,nclusters=2)

#plt.figure()
#plt.imshow(state_matrix)

#plt.figure()
#plt.imshow(himage[:,1,:].T)
#plt.show()