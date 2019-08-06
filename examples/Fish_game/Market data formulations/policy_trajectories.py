import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from System_behavior_formulations.two_harvesters_two_policies_shared_info_threshold_simple import fish_game
from fallback_bargaining import fallback_bargaining
import ast
import math

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

cmap1 = plt.cm.get_cmap("plasma")
cmap2 = plt.cm.get_cmap("plasma")

tSteps = 100

ncols = 3
nrows = 4

fig =  plt.figure(figsize=(18,9))
maxpreyharvest=0
maxpredatorharvest=0
for l in range(len(solutions_to_highlight)):
    ax1 = fig.add_subplot(nrows,ncols,l+1+int(math.floor(l/ncols))*ncols)
    ax2 = fig.add_subplot(nrows,ncols,l+1+ncols+int(math.floor(l/ncols))*ncols)
#    ax.plot(iso1,y_iso, c='gray')
#    ax.plot(iso2,y_iso, c='gray')
    systemseries = fish_game(solutions_decision_variables[l])
    ax1.set_prop_cycle(cycler('color', [cmap1(0.0005*systemseries[6][i]) for i in range(tSteps)]))
    for i in range(tSteps):
        line1 = ax1.plot(range(tSteps)[i:i+2], systemseries[4][i:i+2], linewidth=1, linestyle='--')
    ax2.set_prop_cycle(cycler('color', [cmap2(0.004*systemseries[7][i]) for i in range(tSteps)]))
    for i in range(tSteps):
        line2 = ax2.plot(range(tSteps)[i:i+2], systemseries[5][i:i+2], linewidth=1, linestyle='-')
#    collapse_thres = ax.axhline(y=K*0.5*0.2, linestyle=':', c='crimson')
#    overfishing_thres = ax.axhline(y=K*0.5*0.5, linestyle=':', c='purple')    
    ax1.set_xlim(0,100)
    ax2.set_xlim(0,100)
    ax1.set_ylim(200,2000)
    ax2.set_ylim(0,255)
    if l%ncols==0:
        ax1.set_ylabel("Prey population\ndensity", fontsize=10)
        ax2.set_ylabel("Predator population\ndensity", fontsize=10)
    ax1.set_title(solutions_names[l],fontsize=12)
    ax2.set_title('NPVa: '+"{0:.2f}".format(systemseries[0])+' NPVb: '+"{0:.2f}".format(systemseries[1])+\
                  ' Mean prey def.: '+"{0:.2f}".format(systemseries[2])+' Mean pred. def.: '+"{0:.2f}".format(systemseries[3]),
                  fontsize=8)
    ax1.set_xticks([x*20 for x in range(6)])
    ax2.set_xticks([x*20 for x in range(6)])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    if l+1+ncols+int(math.floor(l/ncols))*ncols>9:
        ax2.set_xticklabels([str(x*20) for x in range(6)])
        ax2.set_xlabel("Time")
    if max(systemseries[6])>maxpreyharvest:
        maxpreyharvest=max(systemseries[6])
    if max(systemseries[7])>maxpredatorharvest:
        maxpredatorharvest=max(systemseries[7])
    plt.setp(ax1.spines.values(), color=colors[l])
    plt.setp([ax1.get_xticklines(), ax1.get_yticklines()], color=colors[l])
    plt.setp(ax2.spines.values(), color=colors[l])
    plt.setp([ax2.get_xticklines(), ax2.get_yticklines()], color=colors[l])
#plt.figlegend([line1, line2],['Prey trajectory','Predator trajectory'], loc = 'lower right')
sm1 = plt.cm.ScalarMappable(cmap=cmap1)
sm1.set_array([0.0,maxpreyharvest])
fig.subplots_adjust(bottom = 0.2)
cbar_ax1 = fig.add_axes([0.1, 0.06, 0.25, 0.03])
cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation="horizontal")
cb1.ax.set_xlabel("Units of prey harvested")
sm2 = plt.cm.ScalarMappable(cmap=cmap2)
sm2.set_array([0.0,maxpredatorharvest])
cbar_ax2 = fig.add_axes([0.4, 0.06, 0.3, 0.03])
cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
cb2.ax.set_xlabel("Units of predator harvested")
plt.show()



