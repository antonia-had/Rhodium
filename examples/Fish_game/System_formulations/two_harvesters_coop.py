import numpy as np
import itertools
import sys
sys.path.append('../')
from Harvesting_formulations.gaussian_OG import hrvSTR

N = 100 # Number of realizations of environmental stochasticity

tSteps = 100 # no. of timesteps to run the fish game on

# Define problem to be solved
def fish_game(vars, # contains all C, R, W for RBF policy
              a = 0.005, # rate at which the prey is available to the predator
              b = 0.5, # prey growth rate
              c = 0.5, # rate with which consumed prey is converted to predator abundance
              d = 0.1, # predator death rate
              h = 0.1, # handling time (time each predator needs to consume the caught prey)
              K = 2000, # prey carrying capacity given its environmental conditions
              m = 0.7, # predator interference parameter
              sigmaX = 0.004, # variance of stochastic noise in prey population
              sigmaY = 0.004): # variance of stochastic noise of predator population

    x = np.zeros(tSteps+1) # Create prey population array
    y = np.zeros(tSteps+1) # Create predator population array
    z = np.zeros(tSteps+1) # Create harvest array

    # Create array to store harvest for all realizations
    harvest = np.zeros([N,tSteps+1])
    # Create array to store effort for all realizations
    effort = np.zeros([N,tSteps+1])
    # Create array to store prey for all realizations
    prey = np.zeros([N,tSteps+1])
    # Create array to store predator for all realizations
    predator = np.zeros([N,tSteps+1])
    
    # Create array to store metrics per realization
    NPV = np.zeros(N)
    cons_low_harv = np.zeros(N)
    harv_1st_pc = np.zeros(N)    
    
    # Create array with environmental stochasticity for prey
    epsilon_prey = np.random.normal(0.0, sigmaX, N)
    
    # Create array with environmental stochasticity for predator
    epsilon_predator = np.random.normal(0.0, sigmaY, N)

    # Go through N possible realizations
    for i in range(N):
        # Initialize populations and values
        x[0] = prey[i,0] = K
        y[0] = predator[i,0] = 250
        z[0] = effort[i,0] = hrvSTR([x[0]], vars, [[0, K]], [[0, 1]])
        NPVharvest = harvest[i,0] = effort[i,0]*x[0]        
        # Go through all timesteps for prey, predator, and harvest
        for t in range(tSteps):
            if x[t] > 0 and y[t] > 0:
                x[t+1] = (x[t] + b*x[t]*(1-x[t]/K) - (a*x[t]*y[t])/(np.power(y[t],m)+a*h*x[t]) - z[t]*x[t])* np.exp(epsilon_prey[i]) # Prey growth equation
                y[t+1] = (y[t] + c*a*x[t]*y[t]/(np.power(y[t],m)+a*h*x[t]) - d*y[t]) *np.exp(epsilon_predator[i]) # Predator growth equation
                if t <= tSteps-1:
                    input_ranges = [[0, K]] # Prey pop. range to use for normalization
                    output_ranges = [[0, 1]] # Range to de-normalize harvest to
                    z[t+1] = hrvSTR([x[t]], vars, input_ranges, output_ranges)
            prey[i,t+1] = x[t+1]
            predator[i,t+1] = y[t+1]
            effort[i,t+1] = z[t+1]
            harvest[i,t+1] = z[t+1]*x[t+1]
            NPVharvest = NPVharvest + harvest[i,t+1]*(1+0.05)**(-(t+1))
        NPV[i] = NPVharvest
        low_hrv = [harvest[i,j]<prey[i,j]/20 for j in range(len(harvest[i,:]))] # Returns a list of True values when there's harvest below 5%
        count = [ sum( 1 for _ in group ) for key, group in itertools.groupby( low_hrv ) if key ] # Counts groups of True values in a row
        if count: # Checks if theres at least one count (if not, np.max won't work on empty list)
            cons_low_harv[i] = np.max(count)  # Finds the largest number of consecutive low harvests
        else:
            cons_low_harv[i] = 0
        harv_1st_pc[i] = np.percentile(harvest[i,:],1)
    
    return (np.mean(NPV), # Mean NPV for all realizations
            np.mean((K-prey)/K), # Mean prey deficit
            np.mean(cons_low_harv), # Mean worst case of consecutive low harvest across realizations
            np.mean(harv_1st_pc), # 5th percentile of all harvests
            np.mean((predator < 1).sum(axis=1))) 