from typing import NewType, Tuple, List
import networkx as nx


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
        self.node_coloring = {}
                
    def _create_network(self):
        G = nx.Graph()
        G.add_edges_from(self.edges_import)
        return G
        
    def get_neighbors_for_node(self, node:int):
        assert node in self.nodes
        return self.graph.neighbors(node)
    
    def greedy_color_node(self, node: int) -> int:
        """This method is a greedy algorithm that approximates the color of a 
        given node if the network is at its chromatic number.

        Args:
            node (int): the identifier of a single node

        Returns:
            int: the color of the node using a greedy algorithm
        """
        # checking that the node is valid
        assert node in self.nodes
        # upper bound of the number of colors based on number of nodes
        max_color_set = set([num for num in range(self.n_nodes)])
        
        # getting a list of neighgbors
        neighbor_coloring = []
        neighbors = self.get_neighbors_for_node(node=node)
        # finding the colors of the neighbors (if exist)
        for neighbor in neighbors:
            neighbor_color = self.node_coloring.get(neighbor)
            neighbor_coloring.append(neighbor_color)
        
        # choosing the smallest color that is not taken by one of the neighbors
        neighbor_coloring = set(neighbor_coloring)
        node_color = min(max_color_set - neighbor_coloring)
        
        # updating the node coloring dictionary
        self.node_coloring[node] = node_color
        
        return node_color
    
    def greedy_color(self):
        return [self.greedy_color_node(node) for node in self.nodes]
        
            
        
    
    
                
            
            
      