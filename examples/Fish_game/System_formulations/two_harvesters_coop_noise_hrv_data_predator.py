import numpy as np
import itertools
import sys
sys.path.append('../')
from Harvesting_formulations.two_coop import hrvSTR

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
    z_a = np.zeros(tSteps+1) # Create harvest array
    z_b = np.zeros(tSteps+1)

    # Create array to store harvest for all realizations
    harvest_a = np.zeros([N,tSteps+1])
    harvest_b = np.zeros([N,tSteps+1])
    # Create array to store prey for all realizations
    prey = np.zeros([N,tSteps+1])
    # Create array to store predator for all realizations
    predator = np.zeros([N,tSteps+1])
    
    # Create array to store metrics per realization
    NPV_a = np.zeros(N)
    NPV_b = np.zeros(N)
    cons_low_harv = np.zeros(N)
    harv_1st_pc = np.zeros(N)    
    
    # Create array with environmental stochasticity for prey
    epsilon_prey = np.random.normal(0.0, sigmaX, N)
    
    # Create array with environmental stochasticity for predator
    epsilon_predator = np.random.normal(0.0, sigmaY, N)
    
    sensor_noise_prey = np.exp(np.random.normal(0.0, 0.5, N))
    sensor_noise_predator = np.exp(np.random.normal(0.0, 0.5, N))
    
    #Set policy input and output ranges
    input_ranges = [[0, K*sensor_noise_prey.max()], [0, 500*sensor_noise_predator.max()], [0, 500]] # Prey pop. range to use for normalization
    output_ranges = [[0, 1], [0, 1]] # Range to de-normalize harvest to

    # Go through N possible realizations
    for i in range(N):
        # Initialize populations and values
        x[0] = prey[i,0] = K
        y[0] = predator[i,0] = 250
        z_a[0], z_b[0] = hrvSTR([x[0]*sensor_noise_prey[0], y[0]*sensor_noise_predator[0], 0], vars, input_ranges, output_ranges)
        NPVharvest_a = harvest_a[i,0] = z_a[0]*x[0]   
        NPVharvest_b = harvest_b[i,0] = z_b[0]*y[0] 
        # Go through all timesteps for prey, predator, and harvest
        for t in range(tSteps):
            if x[t] > 0 and y[t] > 0:
                x[t+1] = (x[t] + b*x[t]*(1-x[t]/K) - (a*x[t]*y[t])/(np.power(y[t],m)+a*h*x[t]) - z_a[t]*x[t])* np.exp(epsilon_prey[i]) # Prey growth equation
                y[t+1] = (y[t] + c*a*x[t]*y[t]/(np.power(y[t],m)+a*h*x[t]) - d*y[t] - z_b[t]*y[t]) *np.exp(epsilon_predator[i]) # Predator growth equation
                if t <= tSteps-1:
                    z_a[t+1], z_b[t+1] = hrvSTR([x[t]*sensor_noise_prey[t], y[t]*sensor_noise_predator[t], z_b[t]*y[t]], vars, input_ranges, output_ranges)
            prey[i,t+1] = x[t+1]
            predator[i,t+1] = y[t+1]
            harvest_a[i,t+1] = z_a[t+1]*x[t+1]
            harvest_b[i,t+1] = z_b[t+1]*y[t+1]
            NPVharvest_a = NPVharvest_a + harvest_a[i,t+1]*(1+0.05)**(-(t+1))
            NPVharvest_b = NPVharvest_b + harvest_b[i,t+1]*(1+0.05)**(-(t+1))
        NPV_a[i] = NPVharvest_a
        NPV_b[i] = NPVharvest_b
        low_hrv = [harvest_a[i,j]<prey[i,j]/20 for j in range(len(harvest_a[i,:]))] # Returns a list of True values when there's harvest below 5%
        count = [ sum( 1 for _ in group ) for key, group in itertools.groupby( low_hrv ) if key ] # Counts groups of True values in a row
        if count: # Checks if theres at least one count (if not, np.max won't work on empty list)
            cons_low_harv[i] = np.max(count)  # Finds the largest number of consecutive low harvests
        else:
            cons_low_harv[i] = 0
        harv_1st_pc[i] = np.percentile(harvest_a[i,:],1)
    return (np.mean(NPV_a), # Mean NPV for all realizations
            np.mean(NPV_b), # Mean NPV for all realizations
            np.mean((K-prey)/K), # Mean prey deficit
            #np.mean(cons_low_harv), # Mean worst case of consecutive low harvest across realizations
            #np.mean(harv_1st_pc), # 5th percentile of all harvests
            np.mean((250-predator)/250))#(predator < 1).sum(axis=1))) 