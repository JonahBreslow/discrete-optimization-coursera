import networkx as nx
from random import sample
import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm

from src.utils import length, edges

class TSP:
    def __init__(self, nodes) -> None:
        self.nodes = nodes
        self.graph = nx.DiGraph()

    def draw(self):
        pos = {node.name:node.coords for node in self.nodes.values()}
        nx.draw(self.graph, pos=pos)
        nx.draw_networkx_labels(self.graph, pos, font_size=6, font_family="sans-serif")
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        edge_labels = {k: round(v,1) for k,v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)    
        plt.show()

    def objective_value(self):
        return self.graph.size(weight='weight')


class GreedyTSP(TSP):
    def __init__(self, nodes) -> None:
        super().__init__(nodes)

    def greedy_path(self, n_sample=2_000):
        """
        locally optimal choice (the immediately closest path)

        Returns:
            list: a list of triples
            [(node1, node2, distance)...(node_n, node1, distance)]
        """
        node_set = set(self.nodes.keys())
        iter_node = node_set.pop()
        path = [(iter_node,)]
        pbar = tqdm(total=len(node_set))
        while len(node_set) > 1:
            min_distance = np.inf
            if len(node_set) > n_sample:
                node_sample = sample(list(node_set), n_sample)
            else:
                node_sample = node_set
            for node in node_sample:
                distance = length(self.nodes[iter_node], self.nodes[node])
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
            iter_node = closest_node
            path.append((iter_node, min_distance))
            node_set.remove(iter_node)
            pbar.update(1)

        # Final Node
        final_node = node_set.pop()
        final_distance = length(self.nodes[iter_node], self.nodes[final_node])
        path.append((final_node, final_distance))

        # Back to the starting node
        starting_node = path[0][0]
        looping_distance = length(self.nodes[final_node], self.nodes[starting_node])
        path.append((starting_node, looping_distance))

        edges_out = edges(path)
        self.graph.add_weighted_edges_from(edges_out)

        return edges(path)
    
class TabuTSP(GreedyTSP):
    def __init__(self, nodes) -> None:
        super().__init__(nodes)
        self.path = self.greedy_path()  
        pass

    def switch_edges(self, a, b, c, d):
        g = self.graph
        g.add_edges_from([(a, c),(b, d)])  
        g.remove_edges_from([(a, b), (c, d)])
    
    def tabu_path(self):
        # TODO: write a tabu TSP algorithm...
        return 

