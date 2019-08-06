import sys
sys.path.append('../')
from System_behavior_formulations.two_harvesters_two_policies_shared_info_threshold import fish_game
from rhodium import * 
from j3 import J3
import json
import numpy as np
from platypus import wrappers
import ast

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
                    Parameter("sigmaY"),
                    Parameter("preylimit"),
                    Parameter("predatorlimit")]

model.responses = [Response("NPV_a", Response.MAXIMIZE),
                   Response("NPV_b", Response.MAXIMIZE),
                   Response("PreyDeficit", Response.MINIMIZE),
                   Response("PredatorDeficit", Response.MINIMIZE)]

model.constraints = []#Constraint("PredatorExtinction < 1")]

model.uncertainties = [UniformUncertainty("a", 0.002, 0.05),
                       UniformUncertainty("b", 0.01, 1),
                       UniformUncertainty("c", 0.2, 1),
                       UniformUncertainty("d", 0.05, 0.2),
                       UniformUncertainty("h", 0.01, 1),
                       UniformUncertainty("K", 1000, 4000),
                       UniformUncertainty("m", 0.1, 1.5),
                       UniformUncertainty("sigmaX", 0.001, 0.01),
                       UniformUncertainty("sigmay", 0.001, 0.01)]

model.levers = [RealLever("vars", 0.0, 1.0, length = 20)]

#output = optimize(model, "BorgMOEA", 10000, module="platypus.wrappers", epsilons=[10, 10, 0.01, 0.01])
#output.save('sharedinfo_threshold.csv')
#SOWs = sample_lhs(model, 1000)
SOWs = load("SOWS.csv")[1]
#
#if __name__ == "__main__":
#    # Use a Process Pool evaluator, which will work on Python 3+\n",
#    with ProcessPoolEvaluator(4) as evaluator:
#            RhodiumConfig.default_evaluator = evaluator
#            reevaluation = [evaluate(model, update(SOWs, policy)) for policy in output]
#            
#for i in range(len(reevaluation)):
#    reevaluation[i].save("./Revaluation/shared_info_threshold_evaluation_"+str(i)+".csv")
#    
#    
output_sharedinfo_threshold = load('sharedinfo_threshold.csv')[1]
J3(output_sharedinfo_threshold.as_dataframe(list(model.responses.keys())))

#def regret(model, results, baseline, percentile=90):
#    quantiles = []
#    for response in model.responses:
#        if response.dir == Response.MAXIMIZE:
#            if not baseline[response.name]==0:
#                values = [abs((result[response.name] - baseline[response.name]) / baseline[response.name]) if result[response.name]<baseline[response.name] else 0 for result in results]
#                quantiles.append(np.percentile(values, percentile))
#            else:
#                 quantiles.append(0)
#        if response.dir == Response.MINIMIZE:
#            if not baseline[response.name]==0:
#                values = [abs((result[response.name] - baseline[response.name]) / baseline[response.name]) if result[response.name]>baseline[response.name] else 0 for result in results]
#                quantiles.append(np.percentile(values, percentile))
#            else:
#                 quantiles.append(0)
#    return (quantiles)
#
#regret_metric = DataSet()
#keys = range(len(output))
#names = [response.name for response in model.responses]
#for i in keys:
#    regret_metric.append(OrderedDict(zip(names, regret(model, reevaluation[i], output[i], percentile=90))))
#regret_metric.save("sharedinfo_regret.csv")    
#
#def satisficing(model, results):
#    percentages = np.zeros(4)
#    percentages[0] = np.mean([1 if result[model.responses[0].name]>=1200 else 0 for result in results])*100
#    percentages[1] = np.mean([1 if result[model.responses[1].name]>=120 else 0 for result in results])*100
#    percentages[2] = np.mean([1 if result[model.responses[2].name]<=0.5 else 0 for result in results])*100
#    percentages[3] = np.mean([1 if result[model.responses[3].name]<=0.5 else 0 for result in results])*100
#    return (percentages)
#
#satisficing_metric = DataSet()
#keys = range(len(output))
#names = [response.name for response in model.responses]
#for i in keys:
#    satisficing_metric.append(OrderedDict(zip(names, satisficing(model, reevaluation[i]))))
#satisficing_metric.save("sharedinfo_satisficing.csv")
