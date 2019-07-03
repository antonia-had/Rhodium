import sys
sys.path.append('../')
from System_formulations.two_harvesters_coop_noise_hrv_data import fish_game
from rhodium import * 
from j3 import J3
import json

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

model.responses = [Response("NPV_a", Response.MAXIMIZE),
                   Response("NPV_b", Response.MAXIMIZE),
                   Response("PreyDeficit", Response.MINIMIZE),
#                   Response("ConsLowHarvest", Response.MINIMIZE),
#                   Response("WorstHarvest", Response.MAXIMIZE),
                   Response("PredatorExtinction", Response.MINIMIZE)]

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

output = optimize(model, "NSGAII", 10000)
with open("harvest_data.txt", "w") as f:
    json.dump(output, f)

fig1 = parallel_coordinates(model, output, colormap="Blues", c= "NPV_a", target="top")
#
#J3(output.as_dataframe(list(model.responses.keys())))
#
#SOWs = sample_lhs(model, 1000)
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