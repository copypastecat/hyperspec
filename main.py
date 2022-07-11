#dummy main file for running simulations
from classes import image
from classes.substance import substance
from classes import data_handler
import numpy as np
import matplotlib.pyplot as plt
from classes.sensor import sensor
from classes.light_source import light_source
from classes.parameter_estimator import parameter_estimator
from classes.optimizer import optimizer

n_sim_freqs = 20


#read in spectral data from dataset
data_handler = data_handler.data_handler("../datasets/usgs_selected",["BECK","ASD"],["splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt","splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"],n_sim_freqs)
freqs_water, spec_water = data_handler.get_spectrum("splib07a_Seawater_Coast_Chl_SW1_BECKa_AREF.txt","BECK")
freqs_sand, spec_sand = data_handler.get_spectrum("splib07a_Sand_DWO-3-DEL2ar1_no_oil_ASDFRa_AREF.txt","ASD")
#plt.figure()
#plt.plot(freqs_water,spec_water)
#plt.plot(freqs_sand,spec_sand)

#cut and resample the spectra to be defined over the same support set
new_freqs, new_specs = data_handler.equalize_spectra([spec_water,spec_sand],["BECK","ASD"])
water = substance("water",new_freqs, new_specs[0])
sand = substance("sand",new_freqs,new_specs[1])

#define sensor-models
this_ideal_sensor =  sensor(n_sim_freqs,variances=0.00,bias = 0)
this_sensor = sensor(n_sim_freqs, variances=0.01, bias=0)

#sample from spectra using ideal sensor models (for ground truth)
sampling_freqs = [19,1,4]
sampled_vals_water = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=sampling_freqs)
sampled_vals_sand = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=sampling_freqs)

#create optimizer
this_optimizer = optimizer(substances=[water,sand],sensor=this_sensor,n_sim_freqs=n_sim_freqs)
print((this_optimizer.calculate_FIM(sampling_frequencies=sampling_freqs)))
solution = this_optimizer.find_freqs_brute(3,"D")
print(solution)
opt_freqs = solution[0]

#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=opt_freqs)))
#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=[opt_freqs[0]-2,opt_freqs[1]-2,opt_freqs[2]+2])))

#create hypothetical mixture  of ground components
mixture = 0.9*new_specs[0] + 0.1*new_specs[1]

#sample from mixture with noisy sensor
sampled_observation = this_sensor.sample(mixture,samplingpoints=[10,3,4])

plt.figure()
plt.plot(new_specs[0])
plt.plot(new_specs[1])
plt.show()

#use the least-squares estimator to estimate the mixture coefficients
this_estimator = parameter_estimator()
estimated_mixture_coeffs = this_estimator.estimate_parameters(np.array([sampled_vals_water,sampled_vals_sand]).T,sampled_observation)
#print(estimated_mixture_coeffs)

####
#image generation (skipped for now...)
####

#rad_sun = np.ones(n_sim_freqs) + 0.04*np.random.rand(n_sim_freqs)
#sun = light_source("sun", rad_sun)

#this_image = image.image((128,128), n_sim_freqs, [concrete, water], sun)

#state_matrix, himage = this_image.generate_image(approx_share=1,nclusters=2)

#plt.figure()
#plt.imshow(state_matrix)

#plt.figure()
#plt.imshow(himage[:,1,:].T)
#plt.show()