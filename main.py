#dummy main file for running simulations
from classes import image
from classes import substance
import numpy as np
import matplotlib.pyplot as plt

from classes.light_source import light_source

n_sim_freqs = 200

abs_conc = np.log(np.arange(n_sim_freqs)+1)
concrete = substance.substance("concrete", abs_conc)

abs_water = np.ones(n_sim_freqs)
water = substance.substance("water", abs_water)

rad_sun = np.ones(n_sim_freqs) + 0.04*np.random.rand(n_sim_freqs)
sun = light_source("sun", rad_sun)

this_image = image.image((1028,1028), n_sim_freqs, [concrete, water], sun)

state_matrix = this_image.generate_image(1)

plt.imshow(state_matrix)
plt.show()

