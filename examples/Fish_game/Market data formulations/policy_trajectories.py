import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from System_behavior_formulations.two_harvesters_two_policies_shared_info_threshold_simple import fish_game
from fallback_bargaining import fallback_bargaining
import ast
plt.style.use('ggplot')

# load data
regret = np.loadtxt("sharedinfo_regret.csv", skiprows=1, delimiter=',')
satisficing = np.loadtxt("sharedinfo_satisficing.csv", skiprows=1, delimiter=',')
objectives = pd.read_csv('sharedinfo_threshold.csv', header=0,  sep=',', usecols=[1,2,3,4]).values
decisionvariables = pd.read_csv('sharedinfo_threshold.csv', header=0,  sep=',', usecols=[0])

# set criteria direction of preference
objective_direction = [1,1,-1,-1] 
satisficing_direction = [1,1,1,1]
regret_direction = [-1,-1,-1,-1]

objectivescompromize=fallback_bargaining(objectives, objective_direction)
satisficingcompromize=fallback_bargaining(satisficing, satisficing_direction)
regretcompromize=fallback_bargaining(regret, regret_direction)

objectivesNPVa=np.argmax(objectives[:,0])
satisficingNPVa=np.argmax(satisficing[:,0])
regretNPVa=np.argmin(regret[:,0])

solutions_to_highlight = [objectivesNPVa, satisficingNPVa, regretNPVa,
                          objectivescompromize, satisficingcompromize, regretcompromize]

solutions_names = ['Max. NPV_a', 'Max. NPV_a satisficing', 'Min. NPV_a regret',
                   'Objectives compromise', 'Satisficing compromise', 'Regret compromise']

solutions_decision_variables = [ast.literal_eval(decisionvariables.at[s,'vars']) for s in solutions_to_highlight]

colors = ['red', 'blue','green','yellow','orange','purple']

cmap = plt.cm.get_cmap("plasma")

nb_points = 100
y_iso = [np.linspace(0, 300, nb_points)]

def isoclines(y):
    a = 0.005 # rate at which the prey is available to the predator
    b = 0.5 # prey growth rate
    c = 0.5 # rate with which consumed prey is converted to predator abundance
    d = 0.1 # predator death rate
    h = 0.1 # handling time (time each predator needs to consume the caught prey)
    K = 2000 # prey carrying capacity given its environmental conditions
    m = 0.7 # predator interference parameter
    return ([(y**m*d)/(a*(c-h*d)),
             K*b/(2*b)-y**m/(2*a*h)+K*np.sqrt((a*h*b+y**m*b/K)**2-4*a**2*h*b*y/K)/(2*a*h*b)])
    
iso1= np.zeros([nb_points]) # y isocline
iso2= np.zeros([nb_points]) # x isocline
iso1, iso2 = isoclines(y_iso)

ncols = 3
nrows = 2

fig =  plt.figure(figsize=(18,9))
for l in range(nrows*ncols):
    ax = fig.add_subplot(nrows,ncols,l+1)
    ax.plot(iso1,y_iso, c='gray')
    ax.plot(iso2,y_iso, c='gray')
    
    for n in range(len(highprofitharv[0,:,:])): # Loop through initial conditions
        ax.set_prop_cycle(cycler('color', [cmap(1.*highprofitharv[l,n,2][i]) for i in range(tSteps)]))
        for i in range(tSteps):
            line1 = ax.plot(highprofitharv[l,n,0][i:i+2], highprofitharv[l,n,1][i:i+2], linewidth=2, linestyle='--',label='Most robust in NPV')
        ax.set_prop_cycle(cycler('color', [cmap(1.*robustharv[l,n,2][i]) for i in range(tSteps)]))
        for i in range(tSteps):
            line2 = ax.plot(robustharv[l,n,0][i:i+2], robustharv[l,n,1][i:i+2], linewidth=2, linestyle='-',label='Most robust in all criteria')
#            ax.set_prop_cycle(cycler('color', [cmap(1.*noharv[ncols*k+l,n,2][i]) for i in range(tSteps)]))
#            for i in range(tSteps):
#                line3 = ax.plot(noharv[ncols*k+l,n,0][i:i+2], noharv[ncols*k+l,n,1][i:i+2], linewidth=2, linestyle='-',label='No harvest')
        endpoint1 = ax.scatter(highprofitharv[l,n,0][100], highprofitharv[l,n,1][100], c='darkgoldenrod', s=20)
        endpoint2 = ax.scatter(robustharv[l,n,0][100], robustharv[l,n,1][100], c='gold', s=20)
#            endpoint3 = ax.scatter(noharv[ncols*k+l,n,0][100], noharv[ncols*k+l,n,1][100], c='black', s=20)
        collapse_thres = ax.axvline(x=worlds[l][5]*0.5*0.2, linestyle=':', c='crimson')
        overfishing_thres = ax.axvline(x=worlds[l][5]*0.5*0.5, linestyle=':', c='purple')
    ax.set_xlabel("Prey")
#        ax.set_ylim(0,305)
#        ax.set_xlim(0,2500)
    if l==0:
        ax.set_ylabel("Predator")        
#        if l==2:       
#            ax.legend([endpoint1, endpoint2, pop_thres],['Most robust in NPV equilibrium point','Most robust in all criteria equilibrium point','Population threshold'], loc = 'lower right')
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([0.0,1.0])
fig.subplots_adjust(bottom = 0.2)
cbar_ax = fig.add_axes([0.1, 0.06, 0.8, 0.06])
cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cb.ax.set_xlabel("Ratio of prey harvested")
plt.show()
plt.savefig("policy_trajectories.png")
plt.savefig("policy_trajectories.svg")


