#dummy main file for running simulations
from classes.image import image
from classes.substance import substance
from classes.data_handler import data_handler
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from classes.sensor import sensor
from classes.light_source import light_source
from classes.parameter_estimator import parameter_estimator
from classes.optimizer import optimizer
import sys
import pandas as pd
import time

n_sim_freqs = 130
N = 5


#read in spectral data from dataset
this_data_handler = data_handler("./usgs_selected",["BECK","ASD","chitin1"],["splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt","splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt","wavelengths_chitin_2.txt"],n_sim_freqs)
freqs_water, spec_water = this_data_handler.get_spectrum("splib07a_Seawater_Coast_Chl_SW1_BECKa_AREF.txt","BECK")
freqs_sand, spec_sand = this_data_handler.get_spectrum("splib07a_Sand_DWO-3-DEL2ar1_no_oil_ASDFRa_AREF.txt","ASD")
freqs_concrete, spec_concrete = this_data_handler.get_spectrum("s07_AV14_Concrete_WTC01-37A_ASDFRa_AREF.txt","ASD")
freqs_chitin, spec_chitin = this_data_handler.get_spectrum("chitin_10layer_75mm_water_2.txt","chitin1")

#plt.figure()
#plt.plot(freqs_water,spec_water)
#plt.plot(freqs_sand,spec_sand)

#cut and resample the spectra to be defined over the same support set
new_freqs, new_specs = this_data_handler.equalize_spectra([spec_water,spec_sand,spec_concrete,spec_chitin],["BECK","ASD","ASD","chitin1"])
water = substance("water",new_freqs, new_specs[0])
sand = substance("sand",new_freqs,new_specs[1])
concrete = substance("concrete",new_freqs,new_specs[2])
chitin = substance("chitin",new_freqs,new_specs[3])
#"null-hypothesis"
nothing = substance("nothing", new_freqs, np.zeros(len(new_freqs)))

#define sensor-models, optimizer
this_sensor = sensor(new_freqs[0],new_freqs[-1], n_sim_freqs, variances=0.1, bias=0)
this_ideal_sensor =  sensor(new_freqs[0],new_freqs[-1],n_sim_freqs,variances=0.00,bias = 0)
this_optimizer = optimizer(substances=[water,sand,concrete,chitin,nothing],sensor=this_sensor,n_sim_freqs=n_sim_freqs)
standard_sampling_freqs = np.array([17,30,48]) #RGB for n_sim_freqs = 60, min(ADS,BECK) range
t_Dopt_start = time.time()
#solution = this_optimizer.find_freqs_brute(N=N,criterion="D")
solution = np.array([43,56,76]) #RGB sampling points
mikasense_freqs = this_sensor.freq_to_index(np.array([0.475,0.56,0.668,0.717,0.842]))
#mikasense_freqs = np.array([43,56,76,93,118]) #sampling points of MikaSense RedEdge-MX
t_Dopt_stop = time.time()
t_mink_start = time.time()
approx_sampling_freqs = this_optimizer.find_freqs_minokwski_approx(N=N)
t_mink_stop = time.time()
t_BB_start = time.time()
BB_solution = this_optimizer.find_freqs_DFBB(N=N)
t_BB_stop = time.time()
t_greedy_start = time.time()
greedy_solution = this_optimizer.greedy_minkowski(N=N)
t_greedy_stop = time.time()
#sample from spectra using ideal sensor models (for ground truth)
sampled_vals_water = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))
sampled_vals_sand = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))
sampled_vals_concrete = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))
sampled_vals_chitin = this_ideal_sensor.sample(chitin.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))

#print(approx_solution)
#opt_freqs = approx_solution[0]
sampled_vals_water_opt = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=solution.astype(int))
sampled_vals_sand_opt = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=solution.astype(int))
sampled_vals_concrete_opt = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=solution.astype(int))
sampled_vals_chitin_opt = this_ideal_sensor.sample(chitin.radiation_pattern,samplingpoints=solution.astype(int))

sampled_vals_water_approx = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))
sampled_vals_sand_approx = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))
sampled_vals_concrete_approx = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))
sampled_vals_chitin_approx = this_ideal_sensor.sample(chitin.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))

#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=opt_freqs)))
#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=[opt_freqs[0]-2,opt_freqs[1]-2,opt_freqs[2]+2])))

#create hypothetical mixture  of ground components
coeffs = [0.25,0.25,0.25,0.25]
mixture = coeffs[0]*new_specs[0] + coeffs[1]*new_specs[1] + coeffs[2]*new_specs[2] + coeffs[3]*new_specs[3]
#'''
#sample from mixture with noisy sensor, compute statistics
#print(new_freqs)
this_estimator = parameter_estimator(method="cls")
avg_n = 5000
MSEs_opt=[]
MSEs_std = []
MSEs_approx = []
for m in range(avg_n):
    vars = np.flip(np.arange(0.000,4,0.5))
    MSE_opt = []
    MSE_std = []
    MSE_approx = []
    for var in vars:
       this_sensor = sensor(new_freqs[0],new_freqs[-1],n_sim_freqs, variances=var, bias=0)
       observations_opt = []
       observations_std = []
       observations_approx = []
       for i in range(1):
           observations_opt.append(this_sensor.sample(mixture,samplingpoints=solution.astype(int)))
           observations_std.append(this_sensor.sample(mixture,mikasense_freqs.astype(int)))
           observations_approx.append(this_sensor.sample(mixture,greedy_solution[0].astype(int)))
       opt_arr = np.array(observations_opt)
       standard_arr = np.array(observations_std)
       approx_arr = np.array(observations_approx)
       mean_opt = opt_arr.mean(axis=0)
       mean_standard = standard_arr.mean(axis=0)
       mean_approx = approx_arr.mean(axis=0)
       est_coeffs_opt = this_estimator.estimate_parameters(np.array([sampled_vals_water_opt,sampled_vals_sand_opt,sampled_vals_concrete_opt,sampled_vals_chitin_opt]).T,mean_opt)[0]
       est_coeffs_std = this_estimator.estimate_parameters(np.array([sampled_vals_water,sampled_vals_sand,sampled_vals_concrete,sampled_vals_chitin]).T,mean_standard)[0]
       est_coeffs_approx = this_estimator.estimate_parameters(np.array([sampled_vals_water_approx,sampled_vals_sand_approx,sampled_vals_concrete_approx,sampled_vals_chitin_approx]).T,mean_approx)[0]
       sqerr_opt = (coeffs - est_coeffs_opt)**2
       sqerr_std = (coeffs - est_coeffs_std)**2
       sqerr_approx = (coeffs - est_coeffs_approx)**2
       MSE_opt.append(sqerr_opt.mean())
       MSE_std.append(sqerr_std.mean())
       MSE_approx.append(sqerr_approx.mean())
    MSEs_opt.append(MSE_opt)
    MSEs_std.append(MSE_std)
    MSEs_approx.append(MSE_approx)
    progress = (m/avg_n) * 100
    sys.stdout.write("\r{0}>".format(m))
    sys.stdout.flush()

#compute average estimation error:
MSEs_opt = np.array(MSEs_opt)
MSEs_std = np.array(MSEs_std)
MSEs_approx = np.array(MSEs_approx)
df_data = np.array([vars,MSEs_opt.mean(axis=0),MSEs_approx.mean(axis=0),MSEs_std.mean(axis=0)]).T
df = pd.DataFrame(df_data,columns=["x","RGB","Greedy","Mikasense"])
df.to_csv("MSEdata_greedy_equalmix.csv")
plt.plot(vars,MSEs_opt.mean(axis=0))
plt.plot(vars,MSEs_std.mean(axis=0))
plt.plot(vars,MSEs_approx.mean(axis=0))
plt.xlabel("$\sigma^2$")
plt.ylabel("$MSE_{avg}$")
plt.legend(["RGB", "Mikasense", "greedy"])
plt.title("Average MSE of estimated coefficients using CLS \n under increasing noise power (concrete, water, sand, chitin)")
plt.savefig("avg_CLS_err_GreedyvsMikavsRGB_equalmix_5000iter.pdf")
plt.show()
#'''

    

'''
plt.figure()
plt.plot(new_freqs, new_specs[0],"-bD",markevery=mikasense_freqs.astype(int))
plt.plot(new_freqs, new_specs[1],"-gD",markevery=mikasense_freqs.astype(int))
plt.plot(new_freqs, new_specs[2],"-kD",markevery=mikasense_freqs.astype(int))
plt.plot(new_freqs, new_specs[3],"-rD",markevery=mikasense_freqs.astype(int))
plt.xlabel("$\lambda (\mu m)$")
plt.ylabel("reflection")
plt.legend(["water","sand","concrete","layered chitin"])
plt.show()
'''

'''
spec_arr = np.array(new_specs)
print(spec_arr[:,:,0])
sp_df = pd.DataFrame(spec_arr[:,:,0].T,columns=["water","sand","concrete","chitin"],index=new_freqs)
print(sp_df.tail())
sp_df.to_csv("specs_N3.csv")
'''

print("Time for D-optimal calculation: ", t_Dopt_stop - t_Dopt_start)
print("Time for calculation of Minkowski approximation: ", t_mink_stop - t_mink_start)
print("Time for calculation of DFBB solution: ", t_BB_stop - t_BB_start)
print("Time for calculationn of Greedy solution: ", t_greedy_stop - t_greedy_start)
print("D_opt solution: ", solution, np.linalg.det(this_optimizer.calculate_FIM(solution)))
print("Minkowski solution: ", approx_sampling_freqs)
print("Greedy solution: ", greedy_solution)

'''
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
'''