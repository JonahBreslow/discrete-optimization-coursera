from copy import deepcopy
import networkx as nx
from random import sample, shuffle
import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm
from dask import delayed, compute, visualize

from src.utils import length, edges

class TSP:
    def __init__(self, graph) -> None:
        self.graph = graph

    def _reset_graph(self):
        self.graph = nx.DiGraph()
        return 

    def replace_graph(self, edges):
        self._reset_graph()
        self.graph.add_weighted_edges_from(edges)
        return

    def draw(self):
        pos = {node: (node.coords[0], node.coords[1]) for node in self.graph.nodes}
        nx.draw(self.graph, pos=pos)
        nx.draw_networkx_labels(self.graph, pos, font_size=6, font_family="sans-serif")
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        edge_labels = {k: round(v,1) for k,v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)    
        plt.show()

    @staticmethod
    def objective_value(graph):
        return graph.size(weight='weight')


class GreedyTSP(TSP):
    def __init__(self, graph) -> None:
        super().__init__(graph)

    def solve(self, n_sample=2_000):
        """
        locally optimal choice (the immediately closest path)

        Returns:
            list: a list of triples
            [(node1, node2, distance)...(node_n, node1, distance)]
        """
        node_set = set(self.graph.nodes)
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
                distance = length(
                    self.graph.nodes[iter_node], 
                    self.graph.nodes[node]
                    )
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
            iter_node = closest_node
            path.append((iter_node, min_distance))
            node_set.remove(iter_node)
            pbar.update(1)

        # Final Node
        final_node = node_set.pop()
        final_distance = length(
            self.graph.nodes[iter_node],
             self.graph.nodes[final_node])
        path.append((final_node, final_distance))

        # Back to the starting node
        starting_node = path[0][0]
        looping_distance = length(self.graph.nodes[final_node], self.graph.nodes[starting_node])
        path.append((starting_node, looping_distance))

        edges_out = edges(path)
        self.replace_graph(edges=edges_out)

        return edges_out


class DistanceDictionary:
    def __init__(self, graph) -> None:
        self.nodes = graph.nodes
        self.distance_dict = {}

    def add_distance(self, node1, node2, distance):
        assert node1 != node2
        if node1 < node2:
            key = f'{node1}-{node2}'
            self.distance_dict[key] = distance
        else:
            key = f'{node2}-{node1}'
            self.distance_dict[key] = distance
    
    def get_distance(self, node1, node2):
        assert node1 != node2
        if node1 < node2:
            key = f'{node1}-{node2}'
        else:
            key = f'{node2}-{node1}'

        distance = self.distance_dict.get('key')
        if distance is None:
            distance = length(self.nodes[node1], self.nodes[node2])
            self.add_distance(node1=node1, node2=node2, distance=distance)
        
        return distance

    
class TwoOptTSP(TSP):
    def __init__(self, graph) -> None:
        super().__init__(graph)
        self.nodes = self.graph.nodes

    def initialize_path(self, randomize: bool, distance_dict, greedy_samples: int = 2_000):
        node_list = list(self.nodes.keys())
        n_nodes = len(node_list)
        if randomize:
            shuffle(node_list)
            edges = []
            for i in range(n_nodes):
                init_node = node_list[i-1]
                end_node = node_list[i]
                distance = distance_dict.get_distance(init_node, end_node)
                edges.append((init_node, end_node, distance))
        else:
            print(f"randomize=False causes the path initialization to be greedy algorithm")
            greedy = GreedyTSP(nodes=self.nodes)
            edges = greedy.solve(n_sample=greedy_samples)

        return edges, n_nodes
    
    def _solve_thread(
        self, verbose: int = 0, 
        randomize: bool = True,
        greedy_samples: int = 2_000,
        max_iters: int = 10,
        sample_candidates: int = None
        ):

        graph = deepcopy(self.graph)
        
        distance_dict = DistanceDictionary(graph=graph)
        edges, n_nodes = self.initialize_path(
            randomize=randomize,
            distance_dict=distance_dict,
            greedy_samples=greedy_samples
            )
        graph.add_weighted_edges_from(edges)
        path = [edge[0] for edge in edges]

        improving = True
        swaps = 0
        iteration = 0
        while improving and iteration < max_iters:
            improving = False

            # if verbose in (1,2):
            #     print(f"""{'*'*50}\nITERATION {iteration}\nOBJECTIVE VALUE: {round(self.objective_value(),2)}\nSWAPS: {swaps}\n{'*'*50}""")
            iteration += 1 
            
            for i in range(n_nodes):
                # sample_candidates prevents us from comparing against every other node
                # increases speed.
                if sample_candidates is None:
                    candidates = range(i+2, n_nodes)
                else:
                    rng = [i for i in range(i+2, n_nodes)]
                    if len(rng) <= sample_candidates:
                        candidates = rng
                    else:
                        candidates = sample(rng, sample_candidates)
                for j in candidates:
                    # compute distance of these two edges
                    cur_1_len = edges[i][2]
                    cur_2_len = edges[j][2]

                    cur_len = cur_1_len + cur_2_len

                    # compute distance of new candidate endges
                    new1 = (edges[i][0], edges[j][0])
                    new2 = (edges[i][1], edges[j][1])

                    new_1_len = distance_dict.get_distance(new1[0], new1[1])
                    new_2_len = distance_dict.get_distance(new2[0], new2[1])

                    new_len = new_1_len + new_2_len

                    # If there is an improvement, do the following
                    if new_len < cur_len:
                        improving = True
                        swaps += 1
        
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        for iter in range(i, j+1):
                            new_start_node = path[iter]
                            new_end_node = path[(iter+1) % n_nodes]
                            distance = distance_dict.get_distance(new_start_node, new_end_node)
                            edges[iter] = (new_start_node, new_end_node, distance)
                        # remove all edges, then update with new best route
                        graph.remove_edges_from(list(graph.edges))
                        graph.add_weighted_edges_from(edges)

        #                 if swaps % 100 == 0 and verbose == 2:
        #                     print(f"""{'*'*50}\nITERATION {iteration}\nOBJECTIVE VALUE {round(self.objective_value(),2)}\nSWAPS: {swaps}\n{'*'*50}""")

        # print(f"""{'*'*50}\nITERATION {iteration}\nOBJECTIVE VALUE: {round(self.objective_value(graph),2)}\nSWAPS: {swaps}\n{'*'*50}""")
                
        return graph

    def solve(
        self, threads: int = 8,
        randomize: bool = True,
        greedy_samples: int = 2_000,
        max_iters: int = 10,
        sample_candidates: int = None
    
    ):
        results = []
        for i in range(threads):
            graph = delayed(self._solve_thread)(
                randomize=randomize, 
                greedy_samples=greedy_samples, 
                max_iters=max_iters, 
                sample_candidates=sample_candidates
                )
            results.append((graph, self.objective_value(graph)))

        min_tuple = delayed(min)(results, key=lambda tup: tup[1])
        visualize(min_tuple, engine="cytoscape", filename="compute_graph")


        return compute(min_tuple)[0]





