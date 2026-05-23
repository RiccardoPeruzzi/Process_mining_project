import os
from pathlib import Path
import pm4py
import pandas as pd
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuParams

# Create output directory for images
OUTPUT_DIR = "../../Results"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("-------- LOADING LOG --------")

log = pm4py.read_xes("../Dataset/HospitalBilling.xes")

print("\n-------- ALPHA MINER --------")

net_alpha, im_alpha, fm_alpha = pm4py.discover_petri_net_alpha(log)

# Save Petri net image
parameters = {
    pm4py.visualization.petri_net.visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"
}
gviz_alpha = pm4py.visualization.petri_net.visualizer.apply(
    net_alpha, im_alpha, fm_alpha, parameters=parameters
)
pm4py.visualization.petri_net.visualizer.save(gviz_alpha, os.path.join(OUTPUT_DIR, "alpha.png"))

# Fitness (token-based replay)
fitness_alpha = replay_fitness.apply(
    log, net_alpha, im_alpha, fm_alpha,
    variant=replay_fitness.Variants.TOKEN_BASED
)

# Precision
precision_alpha = precision_evaluator.apply(
    log, net_alpha, im_alpha, fm_alpha,
    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
)

# Generalization
generalization_alpha = generalization_evaluator.apply(
    log, net_alpha, im_alpha, fm_alpha
)

# Simplicity
simplicity_alpha = simplicity_evaluator.apply(net_alpha)

print("\n-------- HEURISTICS MINER --------")

heu_net = pm4py.discover_heuristics_net(log)
net_heu, im_heu, fm_heu = pm4py.convert_to_petri_net(heu_net)

# Save Petri net image
gviz_heu = pm4py.visualization.petri_net.visualizer.apply(
    net_heu, im_heu, fm_heu, parameters=parameters
)
pm4py.visualization.petri_net.visualizer.save(gviz_heu, os.path.join(OUTPUT_DIR, "heuristic.png"))

# Save heuristics net image
gviz_hn = hn_visualizer.apply(heu_net)
hn_visualizer.save(gviz_hn, os.path.join(OUTPUT_DIR, "heuristic_net.png"))

# Fitness
fitness_heu = replay_fitness.apply(
    log, net_heu, im_heu, fm_heu,
    variant=replay_fitness.Variants.TOKEN_BASED
)

# Precision
precision_heu = precision_evaluator.apply(
    log, net_heu, im_heu, fm_heu,
    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
)

# Generalization
generalization_heu = generalization_evaluator.apply(
    log, net_heu, im_heu, fm_heu
)

# Simplicity
simplicity_heu = simplicity_evaluator.apply(net_heu)

print("\n-------- HEURISTICS MINER (with noise filtering thresholds) --------")

heu_net_thresh = heuristics_miner.apply_heu(log, parameters={
    HeuParams.DEPENDENCY_THRESH:    0.8,
    HeuParams.MIN_ACT_COUNT:        500,   # min occurrences for an activity
    HeuParams.MIN_DFG_OCCURRENCES:  500,   # min occurrences for an edge
})

net_heu_t, im_heu_t, fm_heu_t = pm4py.convert_to_petri_net(heu_net_thresh)

# Save heuristics net image with thresholds
gviz_hn_t = hn_visualizer.apply(heu_net_thresh)
hn_visualizer.save(gviz_hn_t, os.path.join(OUTPUT_DIR, "heuristic_net_threshold.png"))

# Save Petri net image with thresholds
gviz_heu_t = pm4py.visualization.petri_net.visualizer.apply(
    net_heu_t, im_heu_t, fm_heu_t, parameters=parameters
)
pm4py.visualization.petri_net.visualizer.save(gviz_heu_t, os.path.join(OUTPUT_DIR, "heuristic_threshold.png"))

# Fitness
fitness_heu_t = replay_fitness.apply(
    log, net_heu_t, im_heu_t, fm_heu_t,
    variant=replay_fitness.Variants.TOKEN_BASED
)

# Precision
precision_heu_t = precision_evaluator.apply(
    log, net_heu_t, im_heu_t, fm_heu_t,
    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
)

# Generalization
generalization_heu_t = generalization_evaluator.apply(
    log, net_heu_t, im_heu_t, fm_heu_t
)

# Simplicity
simplicity_heu_t = simplicity_evaluator.apply(net_heu_t)

print("\n-------- SUMMARY --------")

summary = pd.DataFrame({
    "Metric": ["Fitness (log)", "Precision", "Generalization", "Simplicity"],
    "Alpha Miner": [
        fitness_alpha["log_fitness"],
        precision_alpha,
        generalization_alpha,
        simplicity_alpha,
    ],
    "Heuristics Miner": [
        fitness_heu["log_fitness"],
        precision_heu,
        generalization_heu,
        simplicity_heu,
    ],
    "Heuristics (threshold)": [
        fitness_heu_t["log_fitness"],
        precision_heu_t,
        generalization_heu_t,
        simplicity_heu_t,
    ],
})

print(summary.to_string(index=False))