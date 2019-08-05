import numpy as np
import pandas as pd

# load data
regret = np.loadtxt("sharedinfo_regret.csv", skiprows=1, delimiter=',')
satisficing = np.loadtxt("sharedinfo_satisficing.csv", skiprows=1, delimiter=',')
objectives = pd.read_csv('sharedinfo_threshold.csv', header=0,  sep=',', usecols=[1,2,3,4]).values

objective_direction = [1,1,-1,-1] # 1 for objectives to be maximimized, -1 for objectives to be minimized
satisficing_direction = [1,1,1,1]
regret_direction = [-1,-1,-1,-1]

def fallback_bargaining(performance, criteria_direction):
    totalsolutions = len(performance[:,0])
    numberofusers = len(criteria_direction)
    solutions_rankings = np.zeros([totalsolutions, numberofusers])

    # sort solutions for each user, largest to smallest
    for u in range(numberofusers):
        solutions_rankings[:,u] = np.argsort(performance[:,u]*criteria_direction[u])[::-1]

    # initialize proposals and compromise solution array
    compromise = []
    proposals = [[] for i in range(numberofusers)]
    
    i=0
    # perform fallback bargaining
    while len(compromise) <1:
        for u in range(numberofusers):
            proposals[u].append(solutions_rankings[i,u])  
    	# check to see if there is a common solution across users
        compromise = list(set.intersection(*[set(ranking) for ranking in proposals]))
        if len(compromise) > 0:
            print ("Compromise is solution: " + str(int(compromise[0])) +"\n")
            print ("Option number: " + str(i) + "\n")
        i+=1
    return(int(compromise[0]))
    
objectivescompromize=fallback_bargaining(objectives, objective_direction)
satisficingcompromize=fallback_bargaining(satisficing, satisficing_direction)
regretcompromize=fallback_bargaining(regret, regret_direction)