import os
import random
import sys
import time
from sys import platform as _platform

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as io
from tqdm import tqdm

import estimation
import utilities


# Detect source from a diffusion graph
def detect_source(source, adjacency, who_infected):

    # Jordan-centrality estimate
    jordan_estimate = estimation.jordan_centrality(who_infected)
    jordan_dist = estimation.get_estimate_dist(source, jordan_estimate, adjacency)

    # Rumor centrality estimate
    rumor_estimate = estimation.rumor_centrality(who_infected)
    rumor_dist = estimation.get_estimate_dist(source, rumor_estimate, adjacency)

    # ML estimate
    ml_estimate, _ = estimation.max_likelihood(who_infected, adjacency)
    ml_dist = estimation.get_estimate_dist(source, ml_estimate, adjacency)

    results = (jordan_dist, rumor_dist, ml_dist)
    # results = (jordan_estimate, rumor_estimate, ml_estimate)
    return results

def compute_distance(source, node, graph):
    ctg_path = nx.astar_path(graph, source, node)
    dis = len(ctg_path) - 1
    return dis

if __name__ == "__main__":

    result_path = "results/"
    data_path = "../source_tracer/data/"
    
    start = time.clock()
    names = ['tvshow', 'as', 'fb', 'politician', 'gov']
    for graph_name in names:
        print("Graph:", graph_name)
        jordan_distances, rumor_distances, ml_distances = [], [], []
        gf = utilities.graphFetcher(graph_name=graph_name, data_path=data_path, flag='test', size=(20, 10000))

        pbar = tqdm(range(len(gf)))
        for _ in pbar:
            g, info = next(gf)
            adjacency, who_infected, source = utilities.build_adjacency(g, info)
            results = detect_source(source, adjacency, who_infected)
            j, r, m = results
            jordan_distances.append(j)
            rumor_distances.append(r)
            ml_distances.append(m)
            pbar.set_description('jordan error %d rumor error %d ml error %d' % (j, r, m))
        print("jordan avg error: %.2f" % np.mean(jordan_distances))
        print("rumor avg error: %.2f" % np.mean(rumor_distances))
        print("ml avg error: %.2f" % np.mean(ml_distances))

        path = os.path.join(result_path, graph_name)
        assert(utilities.check_path(path))
        np.save(os.path.join(path, "jordan_results.npy"), jordan_distances)
        np.save(os.path.join(path, "rumor_results.npy"), rumor_distances)
        np.save(os.path.join(path, "ml_results.npy"), ml_distances)
        print("Experiment took %.3f seconds." % (time.clock() - start))
        print("\n")
