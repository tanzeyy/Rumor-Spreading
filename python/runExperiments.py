# runExperiments
import random

import networkx as nx
import numpy as np
import tqdm

import infectionModels
import utilities
from diffusion_graph import DiffGraph

'''Runs a spreading algorithm over a real dataset. Either pramod's algo (deterministic) or message passing (ours)'''    
def run_dataset(adjacency, trials, max_time=20, max_infection=-1, graph_name='as'):
    '''Run dataset runs a spreading algorithm over a dataset. 
    Inputs:
    
          filename:               name of the file containing the data
          trials:                 number of trials to run
          max_time(opt):          the maximum number of timesteps to use
          max_infection(opt):     the maximum number of nodes a node can infect in any given timestep
    
    Outputs:
    
          dists:                  erro distances of jordan center, rumor center, ml estimator.
    
    NB: If max_infection is not set, then we'll run the deterministic algorihtm. Otherwise, it's the message passing one. '''
    # adjacency = buildGraph.buildDatasetGraph(filename, min_degree, max_num_nodes)

    num_nodes = len(adjacency)
    num_true_nodes = sum([len(item)>0 for item in adjacency])
    
    print('----Graph %s statistics:-----' % graph_name)
    degrees = [len(item) for item in adjacency]
    mean_degree = np.mean(degrees)
    num_threes = len([i for i in degrees if i == 3])
    print('the mean degree is',mean_degree)
    print('num true nodes',num_true_nodes)
    print('---------------------------\n')

    dists = []

    pbar = tqdm.tqdm(range(trials))
    for trial in pbar:
        # if trial % 20 == 0:
        while True:
            source = random.randint(0,num_nodes-1)
            if len(adjacency[source]) > 0:
                break
        num_infected, infection_pattern, who_infected, results = infectionModels.infect_nodes_adaptive_diff(source,adjacency,max_time,max_infection)
        # results = infectionModels.detect_source(source, adjacency, max_time, max_infection)
        # unpack the results
        jordan_distances, rumor_distances, ml_distances = results
        dists.append(results)
        pbar.set_description('Trial %d' % trial)
        # print('dists', ml_leaf_dists)
    return dists
    
'''Run a random tree'''    
def run_randtree(trials, max_time, max_infection, degrees_rv, method=0, known_degrees=[], additional_time = 0,
                 q = 0.5, spies=False, spy_probability = 0.0, diffusion=False, est_times=None, num_hops_pa = 0):
    ''' Run dataset runs a spreading algorithm over a dataset. 
    
    Arguments:    
        trials:                 number of trials to run
        max_time:               the maximum number of timesteps to use
        max_infection:          max number of nodes an infected node can infect in 1 timestep
        degrees_rv:             random tree degree random variable
        method:                 which approach to use: 0 = regular spreading to all neighbors, 
                                                       1 = alternative spreading (VS chosen proportional to degree of candidates) 
                                                       2 = pre-planned spreading
                                                       3 = Old adaptive diffusion (i.e. line algorithm)
                                                       4 = regular diffusion (symmetric in all directions)
        known_degrees           the degrees of nodes in the tree for likelihood computation
        additional_time         the number of additional estimates to make after the initial snapshot
        p                       probability with which diffusion passes the message in a timestep
        spy_probability         probability of a node being a spy
        est_times               times at which we estimate the source (must be <= max_time)
    
    Outputs:
    
          p:                      delay rate 
          avg_num_infected:           total number of nodes infected by the algorithm
    
    NB: If max_infection is not set, then we'll run the deterministic algorihtm. Otherwise, it's the message passing one.'''
    
    pd_ml = [0 for i in range(max_time)] # Maximum likelihood
    pd_spy = [0 for i in range(max_time)] # Maximum likelihood
    pd_lei = [0 for i in range(max_time)] # Maximum likelihood
    additional_pd_mean = [0 for i in range(additional_time)] # Pd from additional measurements
    pd_rc = [0 for i in range(max_time)] # Rumor centrality
    pd_jc = [0 for i in range(max_time)] # Jordan centrality
    pd_rand_leaf = [0 for i in range(max_time)]
    avg_num_infected = [0 for i in range(max_time)]
    avg_hop_distance = [0 for i in range(max_time)]
    avg_spy_hop_distance = [0 for i in range(max_time)]
    avg_lei_hop_distance = [0 for i in range(max_time)]
    
    for trial in range(trials):
        if trial % 500 == 0:
            print('\nTrial ',trial, ' / ',trials-1)
        source = 0
        if method == 0:      # Infect nodes with adaptive diffusion over an irregular tree (possibly with multiple snapshots)
            if spies:
                # WIth spies
                
                if not diffusion:   # adaptive diffusion
                    # Infect nodes
                    up_down_infector = spyInfectionModels.UpDownInfector(spy_probability,degrees_rv)
                    infection_details = up_down_infector.infect(source, max_time)
                    who_infected, num_infected = infection_details
                    
                    # Estimate the source
                    adversary = adversaries.UpDownAdversary(source, up_down_infector.spies_info, who_infected, degrees_rv)
                    results = adversary.get_estimates(max_time, est_times)
                    ml_correct, hop_distances = results
                    additional_pd = [0 for i in range(additional_time)]
            else:
                # snapshot, adaptive diffusion over a random tree
                ad_infector = snapshotInfectionModels.RandTreeADInfector(source, max_infection, max_time, degrees_rv)
                ad_adversary = adversaries.ADSnapshotAdversary(source, degrees_rv, d_o = max_infection + 1) # should this be max_infection + 1?
                for idx in range(max_time):
                    # Spread one more timestep
                    ad_infector.infect_one_timestep()
                    # Estimate the source
                    ad_adversary.update_data(ad_infector.who_infected, ad_infector.degrees)
                    ad_adversary.get_estimates(ad_infector.virtual_source)

                ml_correct = ad_adversary.ml_correct
                num_infected = ad_infector.tot_num_infected

            #     infection_details, ml_correct, additional_pd = infectionModels.infect_nodes_adaptive_irregular_tree(source, max_time, max_infection,
            #                                                                                      degrees_rv, additional_time = additional_time, alt=False)
            #     num_infected, infection_pattern, who_infected, additional_hops = infection_details
            #     hop_distances = []
            # # print(additional_hops)
            # additional_pd_mean = [i+j for (i,j) in zip(additional_pd, additional_pd_mean)]
        elif method == 1:    # Infect nodes with preferential attachment adaptive diffusion (PAAD) over random tree
            paad_infector = snapshotInfectionModels.RandTreePAADInfector(source, max_infection, max_time, degrees_rv, num_hops_pa=num_hops_pa)
            paad_adversary = adversaries.PAADSnapshotAdversary(source, degrees_rv, num_hops_pa=num_hops_pa)
            for idx in range(max_time):
                # Spread one more timestep
                paad_infector.infect_one_timestep()
                # Estimate the source
                paad_adversary.update_data(paad_infector.who_infected, paad_infector.degrees, paad_infector.local_neighborhood)
                paad_adversary.get_estimates(paad_infector.virtual_source)

            ml_correct = paad_adversary.ml_correct
            num_infected = paad_infector.tot_num_infected

            # infection_details, ml_correct = infectionModels.infect_nodes_adaptive_irregular_tree(source, max_time, max_infection, 
            #                                                                                      degrees_rv, alt = True)[:2]
            # num_infected, infection_pattern, who_infected = infection_details[:3]
            # print("OLD SPREADING", who_infected)
        elif method == 2:    # Infect nodes with adaptive diffusion over a pre-determined irregular tree
            infection_details, ml_correct, rand_leaf_correct, known_degrees = infectionModels.infect_nodes_adaptive_planned_irregular_tree(source, max_time, max_infection, degrees_rv, known_degrees)
            num_infected, infection_pattern, who_infected = infection_details
            pd_rand_leaf = [i+j for (i,j) in zip(pd_rand_leaf, rand_leaf_correct)]
        elif method == 3:   # Infect nodes with adaptive diffusion over a line
            infection_details, results = infectionModels.infect_nodes_line_adaptive(source, max_time, max_infection, degrees_rv)
            num_infected, who_infected = infection_details
            pd_jc = [i+j for (i,j) in zip(pd_jc, results[0])]
            pd_rc = [i+j for (i,j) in zip(pd_rc, results[1])]
            # We don't actually compute the ML estimate here because it's computationally challenging
            ml_correct = pd_ml 
        elif method == 4:   # infect nodes with regular diffusion
            infection_details, results = infectionModels.infect_nodes_diffusion_irregular_tree(source, max_time, degrees_rv, q, spy_probability,
                                                                                               est_times = est_times, diffusion=True)
            ml_correct, spy_correct, lei_correct = results
            num_infected, who_infected, hop_distances, spy_hop_distances, lei_hop_distances = infection_details
            avg_hop_distance  = [i+j for (i,j) in zip(avg_hop_distance, hop_distances)]
            avg_spy_hop_distance  = [i+j for (i,j) in zip(avg_spy_hop_distance, spy_hop_distances)]
            avg_lei_hop_distance  = [i+j for (i,j) in zip(avg_lei_hop_distance, lei_hop_distances)]
            pd_spy = [i+j for (i,j) in zip(pd_spy, spy_correct)]
            pd_lei = [i+j for (i,j) in zip(pd_lei, lei_correct)]
        # Infect nodes with adaptive diffusion over an irregular tree, alternative spreading
        # unpack the results
        pd_ml = [i+j for (i,j) in zip(pd_ml, ml_correct)]
        avg_num_infected = [i+j for (i,j) in zip(avg_num_infected, num_infected)]
        
    pd_ml = [float(i) / trials for i in pd_ml]
    pd_rand_leaf = [float(i) / trials for i in pd_rand_leaf]
    avg_num_infected = [ float(i) / trials for i in avg_num_infected]
    
    if method == 0:
        additional_pd_mean = [i/trials for i in additional_pd_mean]
        # results = (pd_ml, additional_pd_mean, hop_distances)
        results = (pd_ml, additional_pd_mean, avg_hop_distance)
    elif method == 1:
        results = (pd_ml)
    elif method == 2:
        results = (pd_ml, pd_rand_leaf)
    elif method == 3:
        pd_rc = [float(i) / trials for i in pd_rc]
        pd_jc = [float(i) / trials for i in pd_jc]
        results = (pd_rc, pd_jc)
    elif method == 4:
        pd_spy = [float(i) / trials for i in pd_spy]
        avg_spy_hop_distance = [float(i) / trials for i in avg_spy_hop_distance]
        pd_lei = [float(i) / trials for i in pd_lei]
        avg_lei_hop_distance = [float(i) / trials for i in avg_lei_hop_distance]
        avg_hop_distance = [float(i) / trials for i in avg_hop_distance]
        print('pd_ml: ', pd_ml, 'avg_hop_distance', avg_hop_distance)
        print('pd_spy: ', pd_spy, 'avg_spy_hop_distance', avg_spy_hop_distance)
        print('pd_lei: ', pd_lei, 'avg_lei_hop_distance', avg_lei_hop_distance)
        results = (pd_ml, avg_hop_distance, pd_spy, avg_spy_hop_distance, pd_lei, avg_lei_hop_distance)

    # return avg_num_infected, pd_ml, pd_rand_leaf
    return avg_num_infected, results
