from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from fallback_bargaining import fallback_bargaining

# load data
regret = np.loadtxt("sharedinfo_regret.csv", skiprows=1, delimiter=',')
satisficing = np.loadtxt("sharedinfo_satisficing.csv", skiprows=1, delimiter=',')
objectives = pd.read_csv('sharedinfo_threshold.csv', header=0,  sep=',', usecols=[1,2,3,4]).values

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

objs_labels = ['NPV_a', 
               'NPV_b', 
               'PreyDeficit', 
               'PredatorDeficit']

solutions_to_highlight = [objectivesNPVa, satisficingNPVa, regretNPVa,
                          objectivescompromize, satisficingcompromize, regretcompromize]
solutions_names = ['Max. NPV_a', 'Max. NPV_a satisficing', 'Min. NPV_a regret',
                   'Objectives compromise', 'Satisficing compromise', 'Regret compromise']
colors = ['red', 'blue','green','yellow','orange','purple']

def parallel_coordinate_plot(performance, criteria_directions,space):
    numberofusers = len(criteria_directions)
    numberofsolutions = len(performance[:,0])
    for u in range(numberofusers):
        performance[:,u] = performance[:,u]*criteria_directions[u]
    # Normalization across performance criteria
    mins = performance.min(axis=0)
    maxs = performance.max(axis=0)

    normalized_performance = performance.copy()
    
    for i in range(numberofusers):
        normalized_performance[:,i] = (performance[:,i] - mins[i]) / (maxs[i] - mins[i])
    
    fig = plt.figure(figsize=(18,9)) # create the figure
    ax = fig.add_subplot(1, 1, 1)    # make axes to plot on
    xs = range(numberofusers)
    # Plot all solutions
    for i in range(numberofsolutions):
        ys = normalized_performance[i,:]
        ax.plot(xs, ys, c='#cfcfcf', linewidth=2)
    
    # Highlight specific solutions
    for s in range(len(solutions_to_highlight)):
        ys = normalized_performance[solutions_to_highlight[s],:]
        ax.plot(xs, ys, c=colors[s], linewidth=2, label=solutions_names[s])                
    
    ax.set_ylabel("Preference ->", size= 12)
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([])
    ax.set_xticklabels(objs_labels)
    ax.legend(loc='upper right')
    ax.set_title('Solutions in the '+space+ ' space')

parallel_coordinate_plot(objectives, objective_direction, 'objective')
parallel_coordinate_plot(satisficing, satisficing_direction, 'satisficing')
parallel_coordinate_plot(regret, regret_direction, 'regret')

