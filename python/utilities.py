# utilities
import argparse
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

from diffusion_graph import DiffGraph


def parse_args(args):
    '''
    Parses the arguments used to call trial simulations.
    '''
    parser = argparse.ArgumentParser(description = "Run adaptive diffusion over trees.")
    parser.add_argument("-t", '--trials', help = "Number of trials to run (default: 1)", type = int, default = 1)
    parser.add_argument("-w", '--write_results', help = "Write results to file? (default: false)", action = 'store_true')
    parser.add_argument("-d", '--diffusion', help = "Spread using regular diffusion? (default: false)", action = 'store_true')
    parser.add_argument("-s", '--spy_probability', help = "Probability of a node being a spy (default: 0)", type = float, default = 0.0)
    parser.add_argument("-q", '--delay_parameter', help = "Probability of passing the message in a timestep (default: 0)", type = float, default = 0.5)
    parser.add_argument("-a", '--alt', help = "Use alternative spreading model? (default: 1)", type = int, default = 0)
    parser.add_argument("--db", nargs = '?', help = "Which database to use(fb=facebook, pg=power grid)", type = str, default = 'none')
    parser.add_argument("-r", '--run', help = "Which run number to save as", type = int, default = 0)
    args = parser.parse_args()

    # Num trials
    if args.trials:
        trials = int(args.trials)
    else:
        trials = 1
    # Write results
    if args.write_results:
        write_results = bool(args.write_results)
    else:
        write_results = False # Do not write results to file
    # Run number
    if args.run:
            run = args.run
    else:
        run = 0
    # Spy probability
    if args.spy_probability:
        spy_probability = float(args.spy_probability)
        spies = True
    else: 
        spy_probability = 0.0
        spies = False
    # Diffusion
    if args.diffusion:
        diffusion = bool(args.diffusion)
        # return {'trials':trials, 'write_results':write_results, 'diffusion':diffusion,
                # 'spy_probability':spy_probability,'run':run, 'delay_parameter':delay_parameter}
    else:
        diffusion = False
    if args.delay_parameter:
        delay_parameter = float(args.delay_parameter)
    else:
        delay_parameter = 0.5
    # Preferential attachment spreading
    if args.alt:
        alt = int(args.alt)
    else:
        alt = 0 # Use the alternative method of spreading virtual sources?
    if not (args.db == 'none'):
        database = args.db
        print("The parameters are:\nDataset = ", database,"\nTrials = ",trials,"\nwrite_results = ",write_results,"\nalt = ",alt,"\n")
        
        # return {'trials':trials, 'write_results':write_results, 'database':database, 'run':run, 'spy_probability':spy_probability} 
    else:
        database = None
        
    print("The parameters are:\nTrials = ",trials,"\nwrite_results = ",write_results,"\nalt = ",alt,"\n")
    return {'trials':trials, 'write_results':write_results, 'alt':alt,'spy_probability':spy_probability,'run':run, 
            'diffusion':diffusion, 'spies':spies, 'delay_parameter':delay_parameter, 'database':database}

def nCk(n, k):
    '''
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    '''
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
        
def compute_alpha(m,T,d):
    '''
    Compute the probability of keeping the virtual source
    Arguments:
          m:          distance from the virtual source to the true source, in hops
          T:          time
          d:          degree of the d-regular tree
    
    Outputs:
          alpha:      the probability of keeping the virtual source where it is
    '''
    

    if d > 2:
        return float(pow((d-1),(T-m+1))-1)/(pow(d-1,T+1)-1)
    elif d == 2:
        return float(T-m+1)/(T+1)

    # alpha1 = float(N(T,d)) / (N(T+1,d))
    # if m == 1:
    #     return alpha1
    # else:
    #     # alpha = alpha1 + compute_alpha(m-1,T,d)/(d-1) - 1/(d-1) 
    #     if d > 2:
    #         alpha = (float(1-alpha1)/(d-2))/pow(d-1,m-1) + float(alpha1*(d-1)-1)/(d-2)
    #     else:
    #         alpha = float(T-m) / T
    # return alpha

def N(T,d):
    '''
    Compute the number of graphs that can appear at time T in a d-regular graph
    Arguments:
          T:          time
          d:          degree of the d-regular tree
    
    Outputs:
          n           the number of nodes at time T
    '''
    
    if d > 2:
        n = float(d) / (d-2) * (pow(d-1,T)-1)
    else:
        n = 1 + 2*T
    return n
    
# def N_nodes(T,d):
#     '''
#     Compute the number of nodes that appear in a graph at time T in a d-regular graph
#     '''
#     return N(T,d) + 1

def print_adjacency(adj, true):
    for i in range(len(adj)):
        if len(adj[i]) > 0:
            pass
            # print(i, ': ', adj[i], ' (', true[i],')')
            
def update_spies_diffusion(candidates, spy_probability = 0.3):
    spies = []
    for candidate in candidates:
        if random.random() < spy_probability:
            spies.append(candidate)
    return spies




class graphFetcher():
    def __init__(self,
                 graph_name="small_world",
                 flag="train",
                 data_path="data",
                 size=(0, 200),
                 visible_rate=0):
        graph_path = "graphs/%s_%s.txt" % (graph_name, flag)
        self._data_path = os.path.join(data_path, graph_name)
        self._graphs = []
        self._vr = visible_rate
        with open(graph_path, 'r+') as f:
            graphs = f.read().splitlines()
        # Unpack info from file name
        for graph in graphs:
            _, source, time, num, total = np.array(
                graph.split('.')[0].split('-'))[[0, 2, 4, 6, 8]]
            info = np.array([source, time, num, total]).astype(np.int)
            if info[2] in range(*size):
                self._graphs.append((graph, info))

        random.shuffle(self._graphs)
        self._cur_idx = 0
        if len(self._graphs) == 0:
            raise("No graph meets the requirement.")

    def __iter__(self):
        for graph in self._graphs:
            g, info = self._load(graph)
            yield g, info

    def __next__(self):
        g, info = self._load(self._graphs[self._cur_idx % len(self._graphs)])
        self._cur_idx += 1
        return g, info

    def __len__(self):
        return len(self._graphs)

    def __repr__(self):
        _, info = self._load(self._graphs[self._cur_idx % len(self._graphs)])
        return str(info)

    def _load(self, graph):
        graph_path, info = graph
        g = nx.read_gpickle(os.path.join(
            self._data_path, graph_path))
        source, _, _, _ = info
        G = DiffGraph(g,
                      visible_rate=self._vr,
                      load_graph=True,
                      contagion_source=source,
                      obs_hop=1)
        return G, info

def check_path(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        return True
    except:
        return False
   
def build_adjacency(g, info):
    source, _, _, total = info
    adj = dict(g.graph.adjacency())
    infected_adj = dict(g.contagion_subgraph.adjacency())
    adjacency = [[] for _ in range(total)]
    who_infected = [[] for _ in range(total)]
    for k, v in adj.items():
        adjacency[k] += list(v.keys())
    for k, v in infected_adj.items():
        who_infected[k] += list(v.keys())
    return adjacency, who_infected, source


if __name__ == "__main__":
    gf = graphFetcher()
    g, info = gf.get('as')
    adj1 = build_adjacency(g, info)
    adj2 = build_adjacency(*gf.get('as'))
    assert(adj1 == adj2)