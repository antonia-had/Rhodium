import numpy as np
import pandas as pd

# load data
data = np.loadtxt('satisficing_by_utility.csv', delimiter=',', skiprows=1)

# solution numbers for indexing
sol_nums = data[:,0]

# sort solutions by satisficing for each utility, largest to smallest
owasa_sorted = np.argsort(data[:,1])[::-1]
durham_sorted = np.argsort(data[:,2])[::-1]
cary_sorted = np.argsort(data[:,3])[::-1]
raleigh_sorted = np.argsort(data[:,4])[::-1]

# rank solutions for each utility
owasa_ranking = sol_nums[owasa_sorted]
durham_ranking = sol_nums[durham_sorted]
cary_ranking = sol_nums[cary_sorted]
raleigh_ranking = sol_nums[raleigh_sorted]

# initialize proposals, OWASA and Cary each have 
# the first ~50 options of 100% robustness, so add them
owasa_proposed = owasa_ranking[0:52]
durham_proposed = np.array([])
cary_proposed = cary_ranking[0:48]
raleigh_proposed = np.array([])
compromise = np.array([])

# add all solutions greater than 99% to proposed


i=0
# perform fallback bargaining
while len(compromise) <1:
	# get the ith preference of each utility
	owasa_proposed = np.append(owasa_proposed, owasa_ranking[i])
	durham_proposed = np.append(durham_proposed, durham_ranking[i])
	cary_proposed = np.append(cary_proposed, cary_ranking[i])
	raleigh_proposed = np.append(raleigh_proposed, raleigh_ranking[i])

	# check to see if there is a common solution across utilities
	o_d_intersect = np.intersect1d(owasa_proposed, durham_proposed)
	if len(o_d_intersect) > 0:
		o_d_c_intersect = np.intersect1d(o_d_intersect, cary_proposed)
		if len(o_d_c_intersect) > 0:
			compromise = np.intersect1d(o_d_c_intersect, raleigh_proposed)
			if len(compromise) > 0:
				print "Compromise is solution: " + str(int(compromise[0])) +"\n"
				print "Option number: " + str(i) + "\n"
	i+=1

print owasa_ranking[33]
print durham_ranking[33]
print cary_ranking[33]
print raleigh_ranking[33]





