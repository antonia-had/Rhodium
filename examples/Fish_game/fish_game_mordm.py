import numpy as np
import itertools
from rhodium import * 
from j3 import J3


nRBF = 2 # no. of RBFs to use
nIn = 1 # no. of inputs (depending on selected strategy)
nOut = 1 # no. of outputs (depending on selected strategy)

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
    
    #Set policy input and output ranges
    input_ranges = [[0, K]] # Prey pop. range to use for normalization
    output_ranges = [[0, 1]] # Range to de-normalize harvest to

    # Go through N possible realizations
    for i in range(N):
        # Initialize populations and values
        x[0] = prey[i,0] = K
        y[0] = predator[i,0] = 250
        z[0] = effort[i,0] = hrvSTR([x[0]], vars, input_ranges, output_ranges)
        NPVharvest = harvest[i,0] = effort[i,0]*x[0]        
        # Go through all timesteps for prey, predator, and harvest
        for t in range(tSteps):
            if x[t] > 0 and y[t] > 0:
                x[t+1] = (x[t] + b*x[t]*(1-x[t]/K) - (a*x[t]*y[t])/(np.power(y[t],m)+a*h*x[t]) - z[t]*x[t])* np.exp(epsilon_prey[i]) # Prey growth equation
                y[t+1] = (y[t] + c*a*x[t]*y[t]/(np.power(y[t],m)+a*h*x[t]) - d*y[t]) *np.exp(epsilon_predator[i]) # Predator growth equation
                if t <= tSteps-1:
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
            np.mean((predator < 1).sum(axis=1))) # Mean number of predator extinction days per realization

# Calculate outputs (u) corresponding to each sample of inputs
# u is a 2-D matrix with nOut columns (1 for each output)
# and as many rows as there are samples of inputs
def hrvSTR(Inputs, vars, input_ranges, output_ranges):
    # Rearrange decision variables into C, R, and W arrays
    # C and R are nIn x nRBF and W is nOut x nRBF
    # Decision variables are arranged in 'vars' as nRBF consecutive
    # sets of {nIn pairs of {C, R} followed by nOut Ws}
    # E.g. for nRBF = 2, nIn = 3 and nOut = 4:
    # C, R, C, R, C, R, W, W, W, W, C, R, C, R, C, R, W, W, W, W
    C = np.zeros([nIn,nRBF])
    R = np.zeros([nIn,nRBF])
    W = np.zeros([nOut,nRBF])
    for n in range(nRBF):
        for m in range(nIn):
            C[m,n] = vars[(2*nIn+nOut)*n + 2*m]
            R[m,n] = vars[(2*nIn+nOut)*n + 2*m + 1]
        for k in range(nOut):
            W[k,n] = vars[(2*nIn+nOut)*n + 2*nIn + k]

    # Normalize weights to sum to 1 across the RBFs (each row of W should sum to 1)
    totals = np.sum(W,1)
    for k in range(nOut):
        if totals[k] > 0:
            W[k,:] = W[k,:]/totals[k]
    # Normalize inputs
    norm_in = np.zeros(nIn)
    for m in range (nIn):
        norm_in[m] = (Inputs[m]-input_ranges[m][0])/(input_ranges[m][1]-input_ranges[m][0])
    # Create array to store outputs
    u = np.zeros(nOut)
    # Calculate RBFs
    for k in range(nOut):
        for n in range(nRBF):
            BF = 0
            for m in range(nIn):
                if R[m,n] > 10**-6: # set so as to avoid division by 0
                    BF = BF + ((norm_in[m]-C[m,n])/R[m,n])**2
                else:
                    BF = BF + ((norm_in[m]-C[m,n])/(10**-6))**2
            u[k] = u[k] + W[k,n]*np.exp(-BF)
    # De-normalize outputs
    norm_u = np.zeros(nOut)
    for k in range(nOut):
        norm_u[k] = output_ranges[k][0] + u[k]*(output_ranges[k][1]-output_ranges[k][0])
    return norm_u

# =============================================================================
# Implementation in Rhodium
# =============================================================================

model = Model(fish_game)

model.parameters = [Parameter("vars"),
                    Parameter("a"),
                    Parameter("b"),
                    Parameter("c"),
                    Parameter("d"),
                    Parameter("h"),
                    Parameter("K"),
                    Parameter("m"),
                    Parameter("sigmaX"),
                    Parameter("sigmaY")]

model.responses = [Response("NPV", Response.MAXIMIZE),
                   Response("PreyDeficit", Response.MINIMIZE),
                   Response("ConsLowHarvest", Response.MINIMIZE),
                   Response("WorstHarvest", Response.MAXIMIZE),
                   Response("PredatorExtinction", Response.INFO)]

model.constraints = [Constraint("PredatorExtinction < 1")]

model.uncertainties = [UniformUncertainty("a", 0.002, 2),
                       UniformUncertainty("b", 0.005, 1),
                       UniformUncertainty("c", 0.2, 1),
                       UniformUncertainty("d", 0.05, 0.2),
                       UniformUncertainty("h", 0.001, 1),
                       UniformUncertainty("K", 100, 5000),
                       UniformUncertainty("m", 0.1, 1.5),
                       UniformUncertainty("sigmaX", 0.001, 0.01),
                       UniformUncertainty("sigmay", 0.001, 0.01)]

model.levers = [RealLever("vars", 0.0, 1.0, length = 6)]

output = optimize(model, "NSGAII", 1000)
#fig = parallel_coordinates(model, output, colormap="Blues", c= "NPV", target="top")


#SOWs = sample_lhs(model, 1000)
policy = output.find_max("NPV")
#results = evaluate(model, update(SOWs, policy))

# Visualize using J3
#J3(output.as_dataframe(list(model.responses.keys())))
J3(output.as_dataframe(['NPV', 'PreyDeficit', 'ConsLowHarvest', 'WorstHarvest']))

result = sa(model, "NPV", policy=policy, method="sobol", nsamples=1000)
#
#classification = results.apply("'Survival' if PredatorExtinction < 1 else 'Extinction'")
#p = Prim(results, classification, include=model.uncertainties.keys(), coi="Survival")
#box = p.find_box()
#fig = box.show_tradeoff()
#
#c = Cart(results, classification, include=model.uncertainties.keys(), min_samples_leaf=50)
#c.show_tree()