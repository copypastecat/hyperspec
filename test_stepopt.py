from classes.substance import substance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from classes.parameter_estimator import parameter_estimator
from classes.optimizer import optimizer
from classes.sensor import sensor

freqs = np.arange(0,3)
s1_spec = np.zeros(10)
s1_spec[0:1] = 1
s2_spec = np.zeros(10)
s2_spec[1:2] = 1
s3_spec = np.zeros(10)
s3_spec[2:3] = 1
#if one wants to add the hypothesis: "non of the 3 substances is present"
s4_spec = np.zeros(10)

s1 = substance("s1", freqs=freqs, radiation_pattern=s1_spec)
s2 = substance("s2", freqs=freqs, radiation_pattern=s2_spec)
s3 = substance("s3", freqs=freqs, radiation_pattern=s3_spec)
s4 = substance("s4", freqs=freqs,  radiation_pattern=s4_spec)

this_sensor = sensor(len(freqs),variances=1)

opt = optimizer(substances=[s1,s2,s3],sensor=this_sensor,n_sim_freqs=len(freqs))

optimal_solution = opt.find_freqs_brute(3, "D",verbose=True)
optimal_solution_BD = opt.find_freqs_brute_GBD(3,verbose=True)
approx_solution_minkowski = opt.find_freqs_minokwski_approx(3)
BB_solution = opt.find_freqs_DFBB(N=3)
greedy_solution = opt.greedy_minkowski(N=3)
grid = optimal_solution[2]
fvals = optimal_solution[3]


'''
X = freqs
Y = freqs
X, Y = np.meshgrid(X,Y)
fig, ax = fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X,Y,-fvals,cmap=cm.coolwarm)
plt.show()
'''

#plot result
#plt.plot(freqs, s1_spec + s2_spec + s3_spec, '-kD', markevery=approx_solution_minkowski[0].astype(int))
#plt.show()
#'''
print(optimal_solution[0:2])
print(optimal_solution_BD[0:2])
print(BB_solution)
print(greedy_solution)
#print(np.unique(fvals))
#'''

#print(opt.calculate_FIM([0,0,0]))
#print(opt.calculate_FIM([0,0,2]))
#print(opt.calculate_FIM([0,1,2])) 