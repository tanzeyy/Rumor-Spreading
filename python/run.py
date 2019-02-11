import os
import random
import sys
import time
from sys import platform as _platform

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

import infectionModels
import runExperiments
import utilities

if __name__ == "__main__":

    max_time = 20
    max_infection = 3 # 20 for spies, 3 for snapshot, -1 for determinisc
    trials = 1
    result_path = "results/"
    
    start = time.clock()
    
    gf = utilities.graphFetcher()
    for graph_name in gf:
        adjacency = utilities.build_adjacency(*gf.get(graph_name))
        dists = runExperiments.run_dataset(adjacency, trials, max_time, max_infection, graph_name)
        jordan_distances, rumor_distances, ml_distances = [], [], []
        for trial in dists:
            j, r, m = trial
            jordan_distances.extend(j[1:])
            rumor_distances.extend(r[1:])
            ml_distances.extend(m[1:])
        print("jordan error: ", np.mean(jordan_distances))
        print("rumor error: ", np.mean(rumor_distances))
        print("ml error: ", np.mean(ml_distances))

        path = os.path.join(result_path, graph_name)
        assert(utilities.check_path(path))
        np.save(os.path.join(path, "jordan_results.npy"), jordan_distances)
        np.save(os.path.join(path, "rumor_results.npy"), rumor_distances)
        np.save(os.path.join(path, "ml_results.npy"), ml_distances)
        print("Experiment took ", time.clock() - start, " seconds.")
        print("\n")
