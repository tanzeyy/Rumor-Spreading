import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats


class DiffGraph():
    def __init__(self, G=nx.karate_club_graph(), 
                       obs_hop=1,
                       visible_rate=1.0,
                       propagation_method="belief",
                       load_graph=False,
                       contagion_source=None):
        """
        Class DiffGraph, generate a diffusion graph, 
            :param G: Networkx.Graph instance to initialize the diffusion graph, default is karate_club_graph
            :param visible_rate: Proportion of visible contagion time of the nodes, default is 1.0
            :param propagation_method: belief or prob, deciding the way susceptible nodes are infected
        """
        # self.pos = nx.spring_layout(G)
        self.graph = G
        self._invisible_rate = 1 - visible_rate
        self._obs_hop = obs_hop
        self.contagion_set = set()
        # Tree-like contagion subgraph
        self.contagion_subgraph = None
        if not load_graph:
            self._set_graph_state()
            self._set_contagion_states()
        else:
            self._set_load_graph()

        if contagion_source is None:
            contagion_source = random.choice(list(self.graph.nodes))
        self._reset_contagion_source(contagion_source)

        if propagation_method == "belief":
            self._get_involved = self._get_involved_belief
        elif propagation_method == "prob":
            self._get_involved = self._get_involved_prob

        self._current_subgraph = None

        # Set invisible nodes with probability self._invisible_rate
        for node_id in random.sample(self.graph.nodes, int(len(self.graph.nodes) * self._invisible_rate)):
            node = self.graph.node[node_id]
            node['visible'] = False

    def _set_graph_state(self):
        '''
        Set the nodes' attributes
        '''
        # Distribution of the states
        lower, upper = 0.2, 1
        mu, sigma = 0.6, 0.1
        dis = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # Ensure non-continous nodes id not wrong
        nodes_num = len(self.graph.nodes)
        id_to_idx = {v: k for k, v in enumerate(list(self.graph.nodes))}

        self._belief_rates = dis.rvs(nodes_num)
        # self._edge_weights = np.random.uniform(0.3, 0.8, size=[nodes_num, nodes_num])

        # Set nodes' inherent attributes
        for node_id in self.graph.nodes:
            node = self.graph.node[node_id]
            node.clear()
            node['rate'] = self._belief_rates[id_to_idx[node_id]]
            node['degree'] = self.graph.degree(node_id)

        for u, v in self.graph.edges:
            edge = self.graph[u][v]
            edge.clear()
            edge['weight'] = np.random.uniform(0.3, 0.8)#self._edge_weights[id_to_idx[u]][id_to_idx[v]]

    def _set_contagion_states(self):
        '''
        Set the contagion states of the nodes
        '''
        # Set nodes' attributes
        for node_id in self.graph.nodes:
            node = self.graph.node[node_id]
            node['state'] = 0
            node['infect_num'] = 0

            # Make sure these two attributes are in the last
            node['contagion_time'] = 0
            node['visible'] = True

        for u, v in self.graph.edges:
            edge = self.graph[u][v]
            edge['used_time'] = 0

    def _reset_contagion_source(self, source):
        self.source_node = source
        self.contagion_set.add(source)
        self.graph.nodes[self.source_node]['state'] = 1

    def reset_contagion_state(self, source=None):
        """
        Reset the source node and the states of all nodes of DiffGraph
        """
        if source == None:
            source = random.choice(list(self.graph.nodes))
        self._set_contagion_states()
        self.contagion_set.clear()
        self._reset_contagion_source(source)

    def save_graph(self, path="graphs/ego-network"):
        nx.write_gpickle(self.graph, path)

    def _update_node_state(self, node_id, mode='infect'):
        """
        Update node state according to its current state
            :param node_id: id of node to update
            :param mode='infect': infect or decay or count, count: how many nodes infected by it
        If mode is decay, only update of states of visible nodes
        """
        node = self.graph.nodes[node_id]
        if mode == 'infect':
            node['state'] = 1
        elif mode == 'decay':
            node['contagion_time'] += 1
        elif mode == 'count':
            node['infect_num'] += 1

    def _update_edge_state(self, u, v):
        """
        Update number of times the edge was used
        """
        self.graph[v][u]['used_time'] += 1

    def _contaminate_neighborhood(self, v):
        '''
        From the perspective of the infector
        '''
        for u in self.graph.neighbors(v):
            if u not in self.contagion_set and (v, u) in self.graph.edges() and self.graph[v][u]['used_time'] == 0:
                epislon = np.random.uniform(0.0, 1.0)
                if self.graph[v][u]['weight'] > epislon:
                    self._update_node_state(u, mode='infect')
                    self._update_edge_state(v, u)
                    self.contagion_set.add(u)

    # Get involved with probability independently with diffrent potential parent nod es
    def _get_involved_prob(self, v, involve_set):
        contagion_parent_candidates = list(
            set(self.graph.neighbors(v)) & self.contagion_set)
        if not contagion_parent_candidates:
            return
        belif_weights = {self.graph[u][v]['weight']
            : u for u in contagion_parent_candidates}
        w = max(belif_weights.keys())
        u = belif_weights[w]
        epislon = np.random.uniform(0.0, 1.0)
        if epislon < w:
            involve_set.add((v, u))

    # Get involved with belief degree of the susceptible node
    def _get_involved_belief(self, v, involve_set):
        contagion_parent_candidates = set(
            self.graph.neighbors(v)) & self.contagion_set
        if not contagion_parent_candidates:
            return
        belif_weights = {self.graph[u][v]['weight']
            : u for u in contagion_parent_candidates}
        w = max(belif_weights.keys())
        u = belif_weights[w]
        if self.graph.nodes[v]['rate'] < w:
            involve_set.add((v, u))

    def _contaminate_one_step(self):
        '''
        From the perspective of the reveiver
        '''
        for v in self.contagion_set:
            self._update_node_state(v, mode='decay')
        susceptible_set = set()
        for v in self.contagion_set:
            susceptible_set = set(self.graph.neighbors(v)) | susceptible_set
        susceptible_set = susceptible_set - self.contagion_set
        involve_set = set()
        for v in susceptible_set:
            self._get_involved(v, involve_set)
        for v, u in involve_set:
            self._update_node_state(v, mode='infect')
            self._update_edge_state(v, u)
            self.contagion_set.add(v)
        for v, u in involve_set:
            self._update_node_state(u, mode='count')

    def contaminate(self, steps=1):
        """
        Contagion the graph n steps
            :param steps: number of contagion steps, default is 1
        """
        for _ in range(steps):
            self._contaminate_one_step()
        self._set_contagion_subgraph()

    def _nodes_attrs(self, nodes=None):
        nodes_set = self.graph.nodes if nodes == None else nodes
        a = np.array([list(self.graph.nodes[i].values()) for i in nodes_set])
        # for idx, node in enumerate(nodes_set):
        #     if self.graph.nodes[node]['visible'] == False:
        #         a[:, -2][idx] = -1
        return a[:, :-1]

    def _subgraph_node_features(self):
        global_attrs = self._nodes_attrs(self._current_subgraph.nodes)

        def infect_degree(g):
            return {node_id: len([1 for v in g.neighbors(node_id) if g.nodes[v]['state'] == 1]) for node_id in g.nodes()}

        # local features of the current subgraph
        funcs = [nx.degree, infect_degree, nx.betweenness_centrality, nx.closeness_centrality]
        local_features = np.concatenate([np.array(list(i.values())).reshape((len(i), 1))
                                   for i in [dict(func(self._current_subgraph)) for func in funcs]], axis=1)
        
        return np.hstack((global_attrs, local_features))

    def combined_features(self, v=None):
        '''
        make sure v is not None
        '''
        if self._current_subgraph == None:
            self._current_subgraph = self.get_subgraph(v)
        # embeddings = self._asne_embed()
        ids = [[x] for x in self._current_subgraph.nodes()]
        features = self._subgraph_node_features()
        combined_featrues = np.concatenate((ids, features), axis=1)
        return combined_featrues

    def get_subgraph(self, v=None, hop=None):
        """
        Get the n-hop subgraph of node v as center, n = 1, 2, ..., depending on self._obs_hop
            :param v: node id of the center node
        """
        if v == None:
            v = self.source_node
        if hop == None:
            hop = self._obs_hop
        n_hop_tree = nx.dfs_tree(self.graph, v, hop)
        self._current_subgraph = nx.subgraph(self.graph, n_hop_tree)

        return self._current_subgraph

    def local_source(self, v=None):
        if v == None:
            return self.source_node
        subg = self.get_subgraph(v)
        contagion_time = np.array(
            [[self.graph.nodes[i]["contagion_time"], i] for i in subg.nodes])
        max_time = np.argmax(contagion_time[:, 0])
        local_s = contagion_time[max_time, :]
        return local_s[1]

    def _set_contagion_subgraph(self):
        edge_set = set()
        for edge in self.graph.edges:
            if self.graph.edges[edge]['used_time'] == 1:
                edge_set.add(edge)
        self.contagion_subgraph = nx.Graph(nx.edge_subgraph(self.graph, edge_set))
    
    def _set_contagion_set(self):
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            if node['state'] == 1:
                self.contagion_set.add(node_id)

    def _set_load_graph(self):
        self._set_contagion_set()
        self._set_contagion_subgraph()

    # TODO: Add embedding method
    def _deep_embed(self):
        pass

    # TODO: Need to be modified in the future
    def get_node_features(self, node):
        pass


if __name__ == "__main__":
    from utils import draw
    import pandas as pd
    np.set_printoptions(suppress=True)

    # edges = pd.read_csv('graphs/edges/0.edges', sep=' ', header=None)
    # g = nx.from_pandas_edgelist(edges, 0, 1)
    g = nx.karate_club_graph()
    G = DiffGraph(g, load_graph=False, visible_rate=0.5)
    # G.contaminate(10)
    pos = nx.spring_layout(G.graph)
    # plt.ion()
    # # ego = nx.ego_graph(G.graph, 236)
    # for i in range(10):
    #     draw(G.graph, [], pos)
    #     plt.show()  
    #     plt.pause(0.1)
    #     input("Please press enter")
    #     G.contaminate()
    # plt.ioff()
    G.contaminate(5)
    # print(G.graph.nodes[12])
    print(G.get_subgraph(1))
    print(G.combined_features())
    print(G.contagion_subgraph)
    pos = nx.spring_layout(g)
    # nx.draw(G.graph, pos=pos, with_labels=True)
    # plt.show()
    nx.draw(G.contagion_subgraph, pos=pos, with_labels=True)
    plt.show()  

