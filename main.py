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

n_sim_freqs = 50
N = 3


#read in spectral data from dataset
data_handler = data_handler.data_handler("../datasets/usgs_selected",["BECK","ASD"],["splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt","splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"],n_sim_freqs)
freqs_water, spec_water = data_handler.get_spectrum("splib07a_Seawater_Coast_Chl_SW1_BECKa_AREF.txt","BECK")
freqs_sand, spec_sand = data_handler.get_spectrum("splib07a_Sand_DWO-3-DEL2ar1_no_oil_ASDFRa_AREF.txt","ASD")
freqs_concrete, spec_concrete = data_handler.get_spectrum("s07_AV14_Concrete_WTC01-37A_ASDFRa_AREF.txt","ASD")
#plt.figure()
#plt.plot(freqs_water,spec_water)
#plt.plot(freqs_sand,spec_sand)

#cut and resample the spectra to be defined over the same support set
new_freqs, new_specs = data_handler.equalize_spectra([spec_water,spec_sand,spec_concrete],["BECK","ASD","ASD"])
water = substance("water",new_freqs, new_specs[0])
sand = substance("sand",new_freqs,new_specs[1])
concrete = substance("concrete",new_freqs,new_specs[2])
#"null-hypothesis"
nothing = substance("nothing", new_freqs, np.zeros(len(new_freqs)))

#define sensor-models
this_ideal_sensor =  sensor(n_sim_freqs,variances=0.00,bias = 0)
this_sensor = sensor(n_sim_freqs, variances=0.001, bias=0)

standard_sampling_freqs = np.linspace(0,n_sim_freqs-1,num=N)
#sample from spectra using ideal sensor models (for ground truth)
sampling_freqs = [19,1,4]
sampled_vals_water = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=standard_sampling_freqs.astype(int))
sampled_vals_sand = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=standard_sampling_freqs.astype(int))
sampled_vals_concrete = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=standard_sampling_freqs.astype(int))

#create optimizer
this_optimizer = optimizer(substances=[water,sand,concrete,nothing],sensor=this_sensor,n_sim_freqs=n_sim_freqs)
solution = this_optimizer.find_freqs_brute(N=N,criterion="D")
print(solution)
opt_freqs = solution[0]
sampled_vals_water_opt = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=solution.astype(int))
sampled_vals_sand_opt = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=solution.astype(int))
sampled_vals_concrete_opt = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=solution.astype(int))

#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=opt_freqs)))
#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=[opt_freqs[0]-2,opt_freqs[1]-2,opt_freqs[2]+2])))

#create hypothetical mixture  of ground components
coeffs = [0.3,0.3,0.4]
mixture = coeffs[0]*new_specs[0] + coeffs[1]*new_specs[1] + coeffs[2]*new_specs[2]

#sample from mixture with noisy sensor, compute statistics
this_estimator = parameter_estimator()
avg_n = 50
MSEs_opt=[]
MSEs_std = []
for m in range(avg_n):
    n_obs = np.arange(1,200)
    MSE_opt = []
    MSE_std = []
    for n in n_obs:
       observations_opt = []
       observations_std = []
       for i in range(n):
           observations_opt.append(this_sensor.sample(mixture,samplingpoints=solution.astype(int)))
           observations_std.append(this_sensor.sample(mixture,standard_sampling_freqs.astype(int)))
       opt_arr = np.array(observations_opt)
       standard_arr = np.array(observations_std)
       mean_opt = opt_arr.mean(axis=0)
       mean_standard = standard_arr.mean(axis=0)
       est_coeffs_opt = this_estimator.estimate_parameters(np.array([sampled_vals_water_opt,sampled_vals_sand_opt,sampled_vals_concrete_opt]).T,mean_opt)[0]
       est_coeffs_std = this_estimator.estimate_parameters(np.array([sampled_vals_water,sampled_vals_sand,sampled_vals_concrete]).T,mean_standard)[0]
       sqerr_opt = (coeffs - est_coeffs_opt)**2
       sqerr_std = (coeffs - est_coeffs_std)**2
       MSE_opt.append(sqerr_opt.mean())
       MSE_std.append(sqerr_std.mean())
    MSEs_opt.append(MSE_opt)
    MSEs_std.append(MSE_std)

#compute average estimation error:
MSEs_opt = np.array(MSEs_opt)
MSEs_std = np.array(MSEs_std)
plt.plot(MSEs_opt.mean(axis=0))
plt.plot(MSEs_std.mean(axis=0))
plt.legend(["MSE optimal sampling", "MSE standard sampling"])
plt.show()

    


#plt.figure()
#plt.plot(new_freqs, new_specs[0],"-bD",markevery=solution.astype(int))
#plt.plot(new_freqs, new_specs[1],"-gD",markevery=solution.astype(int))
#plt.plot(new_freqs, new_specs[2],"-kD",markevery=solution.astype(int))
#plt.legend(["water","sand","concrete"])
#plt.show()

#use the least-squares estimator to estimate the mixture coefficients

#estimated_mixture_coeffs = this_estimator.estimate_parameters(np.array([sampled_vals_water,sampled_vals_sand]).T,sampled_observation)
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