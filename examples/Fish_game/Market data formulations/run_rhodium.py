import sys
sys.path.append('../')
from System_behavior_formulations.two_harvesters_two_policies_threshold import fish_game
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
                    Parameter("K"),
                    Parameter("m")]

model.responses = [Response("NPV_a", Response.MAXIMIZE),
                   Response("NPV_b", Response.MAXIMIZE),
                   Response("PreyDeficit", Response.MINIMIZE),
                   Response("PredatorDeficit", Response.MINIMIZE)]

model.constraints = []#Constraint("PredatorExtinction < 1")]

model.uncertainties = [UniformUncertainty("a", 0.002, 0.02),
                       UniformUncertainty("b", 0.025, 0.075),
                       UniformUncertainty("K", 1500, 3000),
                       UniformUncertainty("m", 0.5, 1.2)]

model.levers = [RealLever("vars", 0.0, 1.0, length = 12)]

output = optimize(model, "BorgMOEA", 10000, module="platypus.wrappers", epsilons=[10, 10, 0.01, 0.01])

#policy = output.find_max("NPV_a")

output.save("noinfo_threshold.csv")

SOWs = sample_lhs(model, 500)
SOWs.save("SOWS.csv")

if __name__ == "__main__":
    # Use a Process Pool evaluator, which will work on Python 3+\n",
    with ProcessPoolEvaluator(4) as evaluator:
            RhodiumConfig.default_evaluator = evaluator
            reevaluation = [evaluate(model, update(SOWs, policy)) for policy in output]

#output_sharedinfo_threshold = load('sharedinfo_threshold.csv')[1]
#output_noinfo_threshold = load('noinfo_threshold.csv')[1]
#
#for i in range(len(output_noinfo_threshold)):
#    output_noinfo_threshold[i]['strategy']=0
#for i in range(len(output_sharedinfo_threshold)):
#    output_sharedinfo_threshold[i]['strategy']=1
#
#merged = DataSet(output_noinfo_threshold+output_sharedinfo_threshold)
#J3(merged.as_dataframe(list(model.responses.keys())+['strategy']))
#
##SOWs = sample_lhs(model, 1000)
##SOWs.save("SOWS.csv")
##
#policy_noinfo = OrderedDict(output_noinfo_threshold.find_max("NPV_a"))
##
##policy_noinfo['vars']=ast.literal_eval(policy_noinfo['vars'])
##
#reevaluation_noinfo = evaluate(model, update(SOWs, policy))
#reevaluation_noinfo.save("reevaluation_noinfo.csv")
#
#
#reevaluation_shared = load("reevaluation_shared.csv")[1]
#
#for i in range(len(reevaluation_noinfo)):
#    reevaluation_noinfo[i]['strategy']=0
#for i in range(len(reevaluation_shared)):
#    reevaluation_shared[i]['strategy']=1
#
#merged = DataSet(reevaluation_noinfo+reevaluation_shared)
#J3(merged.as_dataframe(list(model.responses.keys())+['strategy']))
#
#J3(reevaluation_noinfo.as_dataframe(list(model.responses.keys())))

#fig1 = parallel_coordinates(model, output, colormap="Blues", c= "NPV_a", target="top")
##
J3(output.as_dataframe(list(model.responses.keys())))


#reevaluation = [evaluate(model, update(SOWs, policy)) for policy in output]
#with open("harvest_data_shared_info_reevaluation.txt", "w") as f:
#    json.dump(reevaluation, f) 
#
#def regret(model, results, baseline, percentile=90):
#    quantiles = []
#    for response in model.responses:
#        if response.dir == Response.MINIMIZE or response.dir == Response.MAXIMIZE:
#            if not baseline[response.name]==0:
#                values = [abs((result[response.name] - baseline[response.name]) / baseline[response.name]) for result in results]
#                quantiles.append(np.percentile(values, percentile))
#            else:
#                 quantiles.append(0)
#    return (quantiles)
#
#def satisficing(model, results):
#    percentages = np.zeros(4)
#    percentages[0] = np.mean([1 if result[model.responses[0].name]>=2500 else 0 for result in results])*100
#    percentages[1] = np.mean([1 if result[model.responses[1].name]>=250 else 0 for result in results])*100
#    percentages[2] = np.mean([1 if result[model.responses[2].name]<=0.3 else 0 for result in results])*100
#    percentages[3] = np.mean([1 if result[model.responses[3].name]<=0.3 else 0 for result in results])*100
#    return (percentages)
##
#regret_metric = DataSet()
#keys = range(len(output))
#names = [response.name for response in model.responses]
#for i in keys:
#    regret_metric.append(OrderedDict(zip(names, regret(model, reevaluation[i], output[i], percentile=90))))
#    
#fig2 = parallel_coordinates(model, regret_metric, colormap="Blues", c= "NPV_a", target="bottom")
#
#satisficing_metric = DataSet()
#keys = range(len(output))
#names = [response.name for response in model.responses]
#for i in keys:
#    satisficing_metric.append(OrderedDict(zip(names, satisficing(model, reevaluation_noinfo[i]))))
#    
#fig3 = parallel_coordinates(model, reevaluation_noinfo, colormap="Blues", c= "NPV_a", target="top")
         
#policy = output.find_max("NPV_b")
#results = evaluate(model, update(SOWs, policy))
#fig2 = parallel_coordinates(model, results, colormap="Blues", c= "NPV_a", target="top")
##
#result = sa(model, "NPV", policy=policy, method="sobol", nsamples=1000)
#
#classification = results.apply("'Survival' if PredatorExtinction < 1 else 'Extinction'")
#p = Prim(results, classification, include=model.uncertainties.keys(), coi="Survival")
#box = p.find_box()
#fig = box.show_tradeoff()
#
#c = Cart(results, classification, include=model.uncertainties.keys(), min_samples_leaf=50)
#c.show_tree()