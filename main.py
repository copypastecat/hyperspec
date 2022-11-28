#main file for running simulations
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
from os.path import join

n_sim_freqs = 130
N = 5


#read in spectral data from dataset
this_data_handler = data_handler("./usgs_selected",["BECK","ASD","chitin1"],["splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt","splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt","wavelengths_chitin.txt"],n_sim_freqs)
freqs_water, spec_water = this_data_handler.get_spectrum("splib07a_Seawater_Coast_Chl_SW1_BECKa_AREF.txt","BECK")
freqs_sand, spec_sand = this_data_handler.get_spectrum("splib07a_Sand_DWO-3-DEL2ar1_no_oil_ASDFRa_AREF.txt","ASD")
freqs_concrete, spec_concrete = this_data_handler.get_spectrum("s07_AV14_Concrete_WTC01-37A_ASDFRa_AREF.txt","ASD")
freqs_chitin, spec_chitin = this_data_handler.get_spectrum("chitin_10layer_75mm.txt","chitin1")

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
this_wateroptimizer = optimizer(substances=[water,chitin,nothing],sensor=this_sensor,n_sim_freqs=n_sim_freqs)
rgb_sampling_freqs = this_sensor.freq_to_index(np.array([0.465,0.532,0.630])) #RGB
t_Dopt_start = time.time()
#solution = this_optimizer.find_freqs_brute(N=N,criterion="D")
mikasense_freqs = this_sensor.freq_to_index(np.array([0.475,0.56,0.668,0.717,0.842]))
#approx_sampling_freqs = this_optimizer.find_freqs_minokwski_approx(N=N)
t_mink_stop = time.time()
t_BB_start = time.time()
BB_solution = this_optimizer.find_freqs_DFBB(N=N)
t_BB_stop = time.time()
t_greedy_start = time.time()
greedy_solution = this_optimizer.greedy_minkowski(N=N)
t_greedy_stop = time.time()
#sample from spectra using ideal sensor models (for ground truth)
sampled_vals_water_mikasense = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))
sampled_vals_sand_mikasense = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))
sampled_vals_concrete_mikasense = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))
sampled_vals_chitin_mikasense = this_ideal_sensor.sample(chitin.radiation_pattern,samplingpoints=mikasense_freqs.astype(int))

#print(approx_solution)
#opt_freqs = approx_solution[0]
sampled_vals_water_rgb = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=rgb_sampling_freqs.astype(int))
sampled_vals_sand_rgb = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=rgb_sampling_freqs.astype(int))
sampled_vals_concrete_rgb = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=rgb_sampling_freqs.astype(int))
sampled_vals_chitin_rgb = this_ideal_sensor.sample(chitin.radiation_pattern,samplingpoints=rgb_sampling_freqs.astype(int))

sampled_vals_water_greedy = this_ideal_sensor.sample(water.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))
sampled_vals_sand_greedy = this_ideal_sensor.sample(sand.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))
sampled_vals_concrete_greedy = this_ideal_sensor.sample(concrete.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))
sampled_vals_chitin_greedy = this_ideal_sensor.sample(chitin.radiation_pattern,samplingpoints=greedy_solution[0].astype(int))

#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=opt_freqs)))
#print(np.linalg.det(this_optimizer.calculate_FIM(sampling_frequencies=[opt_freqs[0]-2,opt_freqs[1]-2,opt_freqs[2]+2])))

#create hypothetical mixture  of ground components
coeffs = [0.25,0.25,0.25,0.25]
mixture = coeffs[0]*new_specs[0] + coeffs[1]*new_specs[1] + coeffs[2]*new_specs[2] + coeffs[3]*new_specs[3]
#'''
#sample from mixture with noisy sensor, compute statistics
#print(new_freqs)
this_estimator = parameter_estimator(method="cls")
avg_n = 10000 #ToDo: now that everything is debugged, check # iterations needed for reaosonable MSE variance!
MSEs_rgb=[]
MSEs_mikasense = []
MSEs_greedy = []
for m in range(avg_n):
    vars = np.flip(np.arange(0.000,4,0.03125*4))
    MSE_rgb = []
    MSE_mikasense = []
    MSE_greedy = []
    for var in vars:
       this_sensor = sensor(new_freqs[0],new_freqs[-1],n_sim_freqs, variances=var, bias=0)
       observations_rgb = []
       observations_mikasense = []
       observations_greedy = []
       for i in range(1):
           observations_rgb.append(this_sensor.sample(mixture,samplingpoints=rgb_sampling_freqs.astype(int)))
           observations_mikasense.append(this_sensor.sample(mixture,mikasense_freqs.astype(int)))
           observations_greedy.append(this_sensor.sample(mixture,greedy_solution[0].astype(int)))
       rgb_arr = np.array(observations_rgb)
       mikasense_arr = np.array(observations_mikasense)
       greedy_arr = np.array(observations_greedy)
       mean_rgb = rgb_arr.mean(axis=0)
       mean_mikasense = mikasense_arr.mean(axis=0)
       mean_approx = greedy_arr.mean(axis=0)
       est_coeffs_rgb = this_estimator.estimate_parameters(np.array([sampled_vals_water_rgb,sampled_vals_sand_rgb,sampled_vals_concrete_rgb,sampled_vals_chitin_rgb]).T,mean_rgb)
       est_coeffs_mikasense = this_estimator.estimate_parameters(np.array([sampled_vals_water_mikasense,sampled_vals_sand_mikasense,sampled_vals_concrete_mikasense,sampled_vals_chitin_mikasense]).T,mean_mikasense)
       est_coeffs_approx = this_estimator.estimate_parameters(np.array([sampled_vals_water_greedy,sampled_vals_sand_greedy,sampled_vals_concrete_greedy,sampled_vals_chitin_greedy]).T,mean_approx)
       sqerr_rgb = (coeffs - est_coeffs_rgb)**2
       sqerr_mikasense = (coeffs - est_coeffs_mikasense)**2
       sqerr_approx = (coeffs - est_coeffs_approx)**2
       MSE_rgb.append(sqerr_rgb.mean())
       MSE_mikasense.append(sqerr_mikasense.mean())
       MSE_greedy.append(sqerr_approx.mean())
    MSEs_rgb.append(MSE_rgb)
    MSEs_mikasense.append(MSE_mikasense)
    MSEs_greedy.append(MSE_greedy)
    progress = (m/avg_n) * 100
    sys.stdout.write("\r{0}>".format(m))
    sys.stdout.flush()

print("estimated coefficients RGB: ", est_coeffs_rgb)
print("estimated coefficients Mikasense: ", est_coeffs_mikasense)
print("estimated coefficients Greedy: ", est_coeffs_approx)

#compute average estimation error:
MSEs_rgb = np.array(MSEs_rgb)
MSEs_mikasense = np.array(MSEs_mikasense)
MSEs_greedy = np.array(MSEs_greedy)
df_data = np.array([vars,MSEs_rgb.mean(axis=0),MSEs_greedy.mean(axis=0),MSEs_mikasense.mean(axis=0)]).T
df = pd.DataFrame(df_data,columns=["x","RGB","Greedy","Mikasense"])

#'''
#save results:
path = "./sim/mixture_equal"
df.to_csv(join(path,"msedata.csv"))
#write parameters to text file:
with open(join(path,"parameters.txt"),'w') as f:
    f.write("Mixture coefficients: ")
    f.write(str(coeffs))
    f.write('\n')
    f.write("N runs: ")
    f.write(str(avg_n))
    f.write('\n')
    f.write("RGB wavelengths: ")
    f.write(str(this_sensor.index_to_freq(rgb_sampling_freqs)))
    f.write('\n')
    f.write("Mikasense wavelengths: ")
    f.write(str(this_sensor.index_to_freq(mikasense_freqs)))
    f.write('\n')
    f.write("Greedy-DETMAX wavelengths: ")
    f.write(str(this_sensor.index_to_freq(greedy_solution[0])))
    f.write('\n')
#'''

plt.plot(vars,MSEs_rgb.mean(axis=0))
plt.plot(vars,MSEs_mikasense.mean(axis=0))
plt.plot(vars,MSEs_greedy.mean(axis=0))
plt.xlabel("$\sigma^2$")
plt.ylabel("$MSE_{avg}$")
plt.legend(["RGB", "Mikasense", "greedy"])
plt.title("Average MSE of estimated coefficients using CLS \n under increasing noise power (concrete, water, sand, chitin)")
plt.savefig("avg_CLS_err_GreedyvsMikavsRGB_90water10chitin_10000iter.pdf")
plt.show()
#'''

    

#'''
plt.figure()
plt.plot(new_freqs, new_specs[0],"-bD",markevery=greedy_solution[0].astype(int))
plt.plot(new_freqs, new_specs[1],"-gD",markevery=greedy_solution[0].astype(int))
plt.plot(new_freqs, new_specs[2],"-kD",markevery=greedy_solution[0].astype(int))
plt.plot(new_freqs, new_specs[3],"-rD",markevery=greedy_solution[0].astype(int))
plt.xlabel("$\lambda (\mu m)$")
plt.ylabel("reflection")
plt.legend(["water","sand","concrete","layered chitin"])
plt.show()
#'''

'''
spec_arr = np.array(new_specs)
print(spec_arr[:,:,0])
sp_df = pd.DataFrame(spec_arr[:,:,0].T,columns=["water","sand","concrete","chitin"],index=new_freqs)
print(sp_df.tail())
sp_df.to_csv("specs_N5_longer.csv")
'''

'''
print("Time for calculationn of Greedy solution: ", t_greedy_stop - t_greedy_start)
print("RGB solution: ", rgb_sampling_freqs, np.linalg.det(this_optimizer.calculate_FIM(rgb_sampling_freqs)))
print("Mikasense solution: ", mikasense_freqs)
print("Greedy solution: ", greedy_solution)
'''
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