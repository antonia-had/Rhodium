import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

output_sharedinfo_threshold = pd.read_csv('sharedinfo_threshold.csv', header=0,  sep=',', usecols=[1,2,3,4]).values
output_noinfo_threshold = pd.read_csv('noinfo_threshold.csv', header=0,  sep=',', usecols=[1,2,3,4]).values
output_sharedinfo_threshold[:,2:4]=1-output_sharedinfo_threshold[:,2:4]
output_noinfo_threshold[:,2:4]=1-output_noinfo_threshold[:,2:4]


# Create a mask with brushing conditions
mask_sharedinfo = [True if output_sharedinfo_threshold[i,0]>=1750 and \
                   output_sharedinfo_threshold[i,1]>=120 and \
                   output_sharedinfo_threshold[i,2]>=0.3 and \
                   output_sharedinfo_threshold[i,3]>=0.5 else False for i in range(len(output_sharedinfo_threshold[:,1]))]
#output_sharedinfo_threshold = output_sharedinfo_threshold[mask_sharedinfo,:] 

mask_noinfo = [True if output_noinfo_threshold[i,0]>=1750 and \
                   output_noinfo_threshold[i,1]>=120 and \
                   output_noinfo_threshold[i,2]>=0.3 and \
                   output_noinfo_threshold[i,3]>=0.5 else False for i in range(len(output_noinfo_threshold[:,1]))]
#output_noinfo_threshold = output_noinfo_threshold[mask_noinfo,:]

objs_labels = ['NPV_a', 
               'NPV_b', 
               'PreyDeficit', 
               'PredatorDeficit']

# Normalization across objectives
mins = np.zeros([2,len(output_sharedinfo_threshold[0,:])])
maxs = np.zeros([2,len(output_sharedinfo_threshold[0,:])])
mins[0] = output_sharedinfo_threshold.min(axis=0)
maxs[0] = output_sharedinfo_threshold.max(axis=0)
mins[1] = output_noinfo_threshold.min(axis=0)
maxs[1] = output_noinfo_threshold.max(axis=0)

norm_sharedinfo = output_sharedinfo_threshold.copy()
norm_noinfo = output_noinfo_threshold.copy()

for i in range(len(output_sharedinfo_threshold[0,:])):
    norm_sharedinfo[:,i] = (output_sharedinfo_threshold[:,i] - np.min(mins[:,i])) / (np.max(maxs[:,i]) - np.min(mins[:,i]))
    norm_noinfo[:,i] = (output_noinfo_threshold[:,i] - np.min(mins[:,i])) / (np.max(maxs[:,i]) - np.min(mins[:,i]))
    
norm_sharedinfo_mask = norm_sharedinfo[mask_sharedinfo,:]
norm_noinfo_mask = norm_noinfo[mask_noinfo,:]

fig = plt.figure(figsize=(18,9)) # create the figure
ax = fig.add_subplot(1, 1, 1)    # make axes to plot on
xs = range(len(norm_sharedinfo[0,:]))
## Plot all solutions
for i in range(len(norm_sharedinfo[:,0])):
    ys = norm_sharedinfo[i,:]
    ax.plot(xs, ys, c='#b8cde3', linewidth=2)
            
for i in range(len(norm_noinfo[:,0])):
    ys = norm_noinfo[i,:]
    ax.plot(xs, ys, c='#bd9fa0', linewidth=2)
            
for i in range(len(norm_sharedinfo_mask)):
    ys = norm_sharedinfo_mask[i]
    ax.plot(xs, ys, c='#527aa3', linewidth=2)
            
for i in range(len(norm_noinfo_mask)):
    ys = norm_noinfo_mask[i]
    ax.plot(xs, ys, c='#d66065', linewidth=2)
            
maxNPV_shared = np.argmax(norm_sharedinfo[:,0])
maxNPV_noinfo = np.argmax(norm_noinfo[:,0])

maxNPV_shared_masked = np.argmax(norm_sharedinfo_mask[:,0])
maxNPV_noinfo_masked = np.argmax(norm_noinfo_mask[:,0])

ax.plot(xs, norm_sharedinfo[maxNPV_shared,:], c='#004c99', linewidth=4, linestyle = '--')
ax.plot(xs, norm_noinfo[maxNPV_noinfo,:], c='#a10007', linewidth=4, linestyle = '--')
ax.plot(xs, norm_sharedinfo_mask[maxNPV_shared_masked,:], c='#004c99', linewidth=4)
ax.plot(xs, norm_noinfo_mask[maxNPV_noinfo_masked,:], c='#a10007', linewidth=4)        

ax.set_ylabel("Preference ->", size= 12)
ax.set_xticks([0,1,2,3])
ax.set_yticks([])
ax.set_xticklabels(objs_labels)
plt.savefig('parallelaxis.png')


noinfo_robustness = np.loadtxt("noinfo_robustness.csv", skiprows=1, delimiter=',')
sharedinfo_robustness = np.loadtxt("sharedinfo_robustness.csv", skiprows=1, delimiter=',')

robustness_sharedinfo_mask = sharedinfo_robustness[mask_sharedinfo,:]
robustness_noinfo_mask = noinfo_robustness[mask_noinfo,:]

criteria = ['NPV_a>=1200', 
            'NPV_b>=120', 
            'PreyDeficit<=0.5',
            'PredatorDeficit<=0.5',
            'NPV_a>=1200 and Prey Def.<=0.5',
            'NPV_b>=120 and Predator Def.<=0.5',
            'NPV_a>=1200 and NPV_b>=120','All criteria']

fig = plt.figure(figsize=(18,9)) # create the figure
ax = fig.add_subplot(1, 1, 1)    # make axes to plot on
xs = range(len(noinfo_robustness[0,:]))
## Plot all solutions
for i in range(len(sharedinfo_robustness[:,0])):
    ys = sharedinfo_robustness[i,:]
    ax.plot(xs, ys, c='#b8cde3', linewidth=2)
            
for i in range(len(noinfo_robustness[:,0])):
    ys = noinfo_robustness[i,:]
    ax.plot(xs, ys, c='#bd9fa0', linewidth=2)
            
for i in range(len(robustness_sharedinfo_mask)):
    ys = robustness_sharedinfo_mask[i]
    ax.plot(xs, ys, c='#527aa3', linewidth=2)
            
for i in range(len(robustness_noinfo_mask)):
    ys = robustness_noinfo_mask[i]
    ax.plot(xs, ys, c='#d66065', linewidth=2)

ax.plot(xs, sharedinfo_robustness[maxNPV_shared,:], c='#004c99', linewidth=4, linestyle = '--')
ax.plot(xs, noinfo_robustness[maxNPV_noinfo,:], c='#a10007', linewidth=4, linestyle = '--')
ax.plot(xs, robustness_sharedinfo_mask[maxNPV_shared_masked,:], c='#004c99', linewidth=4)
ax.plot(xs, robustness_noinfo_mask[maxNPV_noinfo_masked,:], c='#a10007', linewidth=4)      

ax.set_ylabel("Preference ->", size= 12)
ax.set_xticks([0,1,2,3,4,5,6,7])
ax.set_yticks([])
ax.set_xticklabels(criteria)
plt.savefig('parallelaxis_robustness.png')
