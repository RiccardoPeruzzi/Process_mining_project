import pm4py

log = pm4py.read_xes("../Dataset/HospitalBilling.xes")

net, im, fm = pm4py.discover_petri_net_alpha(log) # Apply the algorithm.

parameters = {pm4py.visualization.petri_net.visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
gviz = pm4py.visualization.petri_net.visualizer.apply(net, im, fm, parameters=parameters)

pm4py.visualization.petri_net.visualizer.save(gviz, "alpha.png")

fitness = pm4py.algo.evaluation.replay_fitness.algorithm.apply(log, net, im, fm)
print("Fitness in alpha miner:", fitness)