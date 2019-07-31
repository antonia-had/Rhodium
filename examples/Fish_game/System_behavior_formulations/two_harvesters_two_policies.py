import numpy as np

N = 100 # Number of realizations of environmental stochasticity
tSteps = 100 # no. of timesteps to run the fish game on

def generate_policy(input_ranges, output_ranges, vars):
    #Generate policy from vars
    nRBF = 2 # no. of RBFs to use
    nIn = len(input_ranges) # no. of inputs (depending on selected strategy)
    nOut = len(output_ranges) # no. of outputs (depending on selected strategy)
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
            
    policy=[input_ranges, output_ranges, nIn, nOut, nRBF, C, R, W]
    
    return(policy)

def hrvSTR(Inputs, policy):
    input_ranges, output_ranges, nIn, nOut, nRBF, C, R, W = policy
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
    x = np.zeros([N,tSteps]) # Create prey population array
    y = np.zeros([N,tSteps]) # Create predator population array
    z_a = np.zeros(tSteps) # Create harvest array
    z_b = np.zeros(tSteps)

    # Create array to store harvest for all realizations
    harvest_a = np.zeros([N,tSteps])
    harvest_b = np.zeros([N,tSteps])
    
    # Create array to store metrics per realization
    NPV_a = np.zeros(N)
    NPV_b = np.zeros(N)    
    
    # Create array with environmental stochasticity for prey
    epsilon_prey = np.random.normal(0.0, sigmaX, N)
    
    # Create array with environmental stochasticity for predator
    epsilon_predator = np.random.normal(0.0, sigmaY, N)
    
    #Set policy input and output ranges
    input_ranges = [[0, K]] # Prey pop. range to use for normalization
    output_ranges = [[0, 1]] # Range to de-normalize harvest to
    
    preypolicy = generate_policy(input_ranges, output_ranges, vars[:6])
    predatorpolicy = generate_policy(input_ranges, output_ranges, vars[6:])
    
    # Initialize populations and values
    x[:,0] = K
    y[:,0] = 250
    harvest_a[:,0] = 0
    harvest_b[:,0] = 0

    # Go through N possible realizations
    for i in range(N):
        # Initialize harvest NPV
        NPVharvest_a = 0
        NPVharvest_b = 0
        # Go through all timesteps for prey, predator, and harvest
        for t in range(tSteps-1):
            if x[i,t] > 0 and y[i,t] > 0:
                x[i,t+1] = (x[i,t] + b*x[i,t]*(1-x[i,t]/K) - (a*x[i,t]*y[i,t])/(np.power(y[i,t],m)+a*h*x[i,t]) - z_a[t]*x[i,t])* np.exp(epsilon_prey[i]) # Prey growth equation
                y[i,t+1] = (y[i,t] + c*a*x[i,t]*y[i,t]/(np.power(y[i,t],m)+a*h*x[i,t]) - d*y[i,t] - z_b[t]*y[i,t]) *np.exp(epsilon_predator[i]) # Predator growth equation
                z_a[t+1]= hrvSTR([z_a[t]*x[i,t]], preypolicy)
                z_b[t+1]= hrvSTR([z_b[t]*y[i,t]], predatorpolicy)
            harvest_a[i,t+1] = z_a[t+1]*x[i,t+1]
            harvest_b[i,t+1] = z_b[t+1]*y[i,t+1]
            NPVharvest_a += harvest_a[i,t+1]*(1+0.05)**(-(t+1))
            NPVharvest_b += harvest_b[i,t+1]*(1+0.05)**(-(t+1))
        NPV_a[i] = NPVharvest_a
        NPV_b[i] = NPVharvest_b
    
    return (np.mean(NPV_a), # Mean NPV for all realizations
            np.mean(NPV_b), # Mean NPV for all realizations
            np.mean((K-x)/K).clip(0,1), # Mean prey deficit
            np.mean((250-y)/250).clip(0,1))