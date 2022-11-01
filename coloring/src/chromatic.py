import enum
from mimetypes import init
import random
from typing import Dict, NewType, Tuple, List
from collections import OrderedDict
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from copy import copy
from tqdm import tqdm

"""Datatype defining what a list of Edges is

Example:
    `[(0, 1), (1, 2), (2, 3)]` would be a valid Edges object
    `(0, 1), (1, 2), (2, 3)` would not be a valid Edges object
    `[(0, 1, 2), (2, 3, 4)]` would not be a valid Edges object
"""
Edges = NewType('Edges', List[Tuple[int, int]])

class Network():
    def __init__(self, edges_import: Edges) -> None:
        self.edges_import = edges_import
        self.graph = self._create_network()
        self.nodes = self.graph.nodes()
        self.edges = self.graph.edges()
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        self.n_neighbors = self._n_neighbors_array()
        # self.network_dict = self._create_network_dict()
        self.max_color_set = set([num for num in range(self.n_nodes)])
                
    def _create_network(self):
        G = nx.Graph()
        G.add_edges_from(self.edges_import)
        return G
    
    def network_dict(self) -> Dict:
        """Creates a dictionary that defines the network. 

        Returns:
            Dict: keys are nodes, values are lists of neighbors for the node
        """
        network_dict = {}
        for node in self.nodes:
            neighbors = self.get_neighbors_for_node(node)
            network_dict[node] = neighbors
        return network_dict
    
    def get_neighbors_for_node(self, node:int) -> List:
        assert node in self.graph.nodes()
        return [node for node in self.graph.neighbors(node)]
    
    def _n_neighbors_array(self) -> np.ndarray:
        """creates a two-column numpy array where the 0th column is the node-id
        and the 1st column is the count of neighbors that the 0th column's node has.

        Returns:
            np.ndarray: a two-column numpy array where the 0th column is the node-id 
            and the 1st column is the count of neighbors that the 0th column's node has.
        """
        # Order the nodes highest degree first
        n_neighbors = np.zeros(shape=(self.n_nodes, 2), dtype=int)
        for node in self.nodes:
            n_neighbors[node, 0] = node
            n_neighbors[node, 1] = len(self.get_neighbors_for_node(node))
        n_neighbors = n_neighbors[n_neighbors[:,1].argsort()[::-1]]
        return n_neighbors
    
    def draw_graph(self):
        nx.draw_spring(self.graph, with_labels=True)
        plt.show()
        
 
class NodeColoringAlgorithms:
    def __init__(self, network: Network) -> None:
        self.network = network
        self.node_coloring = {}
               
    def reset_node_coloring(self):
        self.node_coloring = {}
        return
    
    def greedy_color_node(self, node: int) -> int:
        """This implements a greedy node coloring algorithm using node . The number of colors
        applied to nodes in the graph approximates a minimum. A node-coloring 
        algorithm that achieves the optimal (minimum) number, k, of colors such that 
        any two connected nodes are not the same color is called a k-chromatic graph
        algorithm.
 
        Args:
            node (int): the identifier of a single node

        Returns:
            int: the color of the node using a greedy algorithm
        """
        # checking that the node is valid
        assert node in self.network.nodes
        
        # getting a list of neighgbors
        neighbor_coloring = []
        neighbors = self.network.get_neighbors_for_node(node=node)
        # finding the colors of the neighbors (if exist)
        for neighbor in neighbors:
            neighbor_color = self.node_coloring.get(neighbor)
            neighbor_coloring.append(neighbor_color)
        
        # choosing the smallest color that is not taken by one of the neighbors
        neighbor_coloring = set(neighbor_coloring)
        node_color = min(self.network.max_color_set - neighbor_coloring)
        
        # updating the node coloring dictionary
        self.node_coloring[node] = node_color
        
        return node_color

    
    def greedy_coloring(self):
        """ This implements a greedy-node coloring algorithm such that the number of colors
        applied to nodes in the graph approximates a lower bound. A node-coloring 
        algorithm that achieves the minimum number of colors, k, such that 
        any two connected nodes are not the same color is called a k-chromatic graph algorithm.
        See https://mathworld.wolfram.com/k-ChromaticGraph.html

        Returns:
            None
        """
        self.reset_node_coloring()
        for node in self.network.n_neighbors[:,0]:
            self.greedy_color_node(node)
        
            
        # update the node coloring dictionary so it is ordered
        self.node_coloring = OrderedDict(sorted(self.node_coloring.items()))
        return self.node_coloring 

    def rlf_sampling(self, n_searches: int) -> Dict:
        """This is a non-deterministic approach to the Recursive Largest First algorithm that
        leverages the `networkx.maximal_independent_set` method, which randomly selects a 
        subset of nodes in graph G such that no edges can be drawn between any of the selected nodes. 
        This implementation will execute this method `n_searches` number of times and will always
        select the subset of nodes that is the largest. 

        Args:
            n_searches (int): the number of maximal_independent_set samples to take

        Returns:
            Dict: A dictionary where the key is the node and the value is the color
        """
        self.reset_node_coloring

        temp_graph = copy(self.network.graph)
        color = 0

        pbar = tqdm(total = len(temp_graph.nodes))
        while len(temp_graph.nodes) > 0:
            adj_dict = temp_graph.adj
            # Finding initial node
            max_len = 0
            init_node =""
            for key in adj_dict.keys():
                cur_len = len(adj_dict.get(key))
                if cur_len >= max_len:
                    init_node = key
                    max_len = cur_len

            max_len = 0
            for iteration, _ in enumerate(range(n_searches)):
                indep_nodes = nx.maximal_independent_set(temp_graph, nodes=[init_node])
                if len(indep_nodes) > max_len:
                    max_independent_set = indep_nodes
                    max_len = len(indep_nodes)
            
            for node in max_independent_set:
                self.node_coloring[node] = color
            
            color += 1
            temp_graph.remove_nodes_from(max_independent_set) 
            pbar.update(len(max_independent_set)) 

        # update the node coloring dictionary so it is ordered
        self.node_coloring = OrderedDict(sorted(self.node_coloring.items()))
        return self.node_coloring 


        
    
    def rlf(self) -> Dict:
        """Implements the Recursive Largest First algorithm described here: 
        https://en.wikipedia.org/wiki/Recursive_largest_first_algorithm. Unfortunately, 
        the heuristic used in this approach to estimating the maximal independent set is 
        O(2^n)-ish so this is not a performant approach. Use `rlf_sampling` for a faster, 
        but non-deterministic, approach.

        Returns:
            Dict: A dictionary where the key is the node and the value is the color
        """
        self.reset_node_coloring()
        
        # The first vertex added is the vertex that has the largest number of neighbors
        temp_graph = copy(self.network)
        color = 0
        
        pbar = tqdm(total = len(temp_graph.nodes))
        while len(temp_graph.nodes) > 0:
            adj_dict = temp_graph.graph.adj
            # Finding initial node
            max_len = 0
            init_node =""
            for key in adj_dict.keys():
                cur_len = len(adj_dict.get(key))
                if cur_len >= max_len:
                    init_node = key
                    max_len = cur_len
            maximal_independent_set_iter = {init_node}
            maximal_independent_set_neigh = []
            keep_going=True
            
            while keep_going:
                maximal_independent_set_neigh = \
                        set(sum([temp_graph.get_neighbors_for_node(node)
                                 for node in maximal_independent_set_iter], []))
                
                # Subsequent vertices are chosen where (a) they are not currently adjacent to vertices in 
                # maximal_independent_set and (b) they have a maximal number of neighbors that are adjecent to vertices
                # in maximal_independent_set.
                            
                # (a) nodes not currently adjacent to nodes in maximal_independent_set
                nodes_not_adjacent_to_max_adj_set = []
                network_dict = temp_graph.network_dict()
                for node in network_dict.keys():
                    neighbors = set(network_dict.get(node))
                    # If intersecting set is more than 0 nodes, intersection is set to True
                    intersection = len(neighbors.intersection(maximal_independent_set_iter)) > 0
                    if node in maximal_independent_set_iter:
                        pass
                    elif not intersection:
                        nodes_not_adjacent_to_max_adj_set.append(node)
                
                # (b) For nodes from the previous step, find the number of neighbors that are adjacent to the 
                # maximal_independent_set
                n_adj_neighbors = {}
                for node in nodes_not_adjacent_to_max_adj_set:
                    neighbors = set(network_dict.get(node))
                    num_adjacent = len(neighbors.intersection(maximal_independent_set_neigh))
                    if n_adj_neighbors.get(num_adjacent) is None:
                        n_adj_neighbors[num_adjacent] = [node]
                    else:
                        n_adj_neighbors[num_adjacent].append(node)
                if len(n_adj_neighbors.keys()) == 0:
                    keep_going = False
                else:
                    max_n_adj_neighbors = max(n_adj_neighbors.keys())
                    best_nodes = n_adj_neighbors.get(max_n_adj_neighbors)    
                    
                    # c tie breaker
                    if len(best_nodes) > 1:
                        tie_breaker = np.inf
                        best_node = best_nodes[0]
                        for node in best_nodes:
                            neighbors = set([neighbor for neighbor in temp_graph.get_neighbors_for_node(node)])
                            neighbors_not_in_s = neighbors - maximal_independent_set_iter
                            if len(neighbors_not_in_s) < tie_breaker:
                                tie_breaker = len(neighbors_not_in_s)
                                best_node = node
                    elif len(best_nodes) == 1:
                        best_node = best_nodes[0]
                    
                    maximal_independent_set_iter.add(best_node)      
                    keep_going = (
                        len(nodes_not_adjacent_to_max_adj_set) > 0 
                        or len(n_adj_neighbors.keys) >0
                    )
            for node in maximal_independent_set_iter:
                self.node_coloring[node] = color
            
            color += 1
            temp_graph.graph.remove_nodes_from(maximal_independent_set_iter) 
            pbar.update(len(maximal_independent_set_iter)) 

        # update the node coloring dictionary so it is ordered
        self.node_coloring = OrderedDict(sorted(self.node_coloring.items()))
        return self.node_coloring 
        
        
        
        
        
        
            
        
    
    
                
            
            
      