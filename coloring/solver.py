from dataclasses import dataclass
from typing import OrderedDict
from src.chromatic import Network, NodeColoringAlgorithms
from networkx.algorithms import coloring

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    network = Network(edges_import=edges)
    
    # # my custom algorithms
    # # network.draw_graph()
    # color_algo = NodeColoringAlgorithms(network=network)
    # solution = color_algo.rlf_sampling(n_searches=10000).values()
    # n_colors = max(solution) + 1

    # built in greedy algorithms
    solution = coloring.greedy_color(network.graph, strategy='independent_set', interchange=False)
    solution = OrderedDict(sorted(solution.items())).values()
    n_colors = max(solution) 



    # prepare the solution in the specified output format
    output_data = str(n_colors) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        file_location = 'data/gc_500_1'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
        # print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

