import numpy as np
from matplotlib import pyplot as plt 
import networkx as nx

from src.utils import length
from src.parser import PointParser
from src.tsp_solvers import GreedyTSP, TwoOptTSP

    

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    parser = PointParser(input_data=input_data, n_samples=None)
    graph = parser.points()

    # Instantiate tsp problem
    two_opt = TwoOptTSP(graph=graph)
    graph, obj = two_opt.solve(
        threads=32,
        randomize=True, 
        max_iters=50,
        sample_candidates = None
        )

    # calculate the length of the tour
    solution = [node[0] for node in nx.find_cycle(graph)]

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
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
        file_location = 'data/tsp_574_1'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
