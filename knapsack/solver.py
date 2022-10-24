from collections import namedtuple
import os
from pprint import pprint
from types import DynamicClassAttribute
from sqlalchemy import false
from src.knapsack_algorithms import KnapsackAlgorithms


Item = namedtuple("Item", ['index', 'value', 'weight', 'norm_value'])

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(
            Item(
                i-1, 
                int(parts[0]), 
                int(parts[1]),
                int(parts[0]) / int(parts[1])
                )
            )

    best_knapsack = KnapsackAlgorithms(items,capacity=capacity)
    branch_bound_knapsack = best_knapsack.branch_and_bound()
    return branch_bound_knapsack
    # # Instantiating the greedy knapsack algorithm class
    # if len(items) > 200:
    #     '''
    #     Greedy Algorithm
    #     1. maximum value
    #     2. minumum weight
    #     3. maximum value density (value / weight)
    #     '''
    #     grid = [(1, True), (2, False), (3, True)]
        
    #     greedy_knapsack = best_knapsack.greedy_choose(grid)
                
    #     value = greedy_knapsack.value
    #     taken = greedy_knapsack.taken
    
    # else:
    #     '''
    #     Dynamic Algorithm
    #     '''
    #     dynamic_knapsack = best_knapsack.dynamic_choose()
    #     value = dynamic_knapsack.value
    #     taken = dynamic_knapsack.taken
    

    
    # # # prepare the solution in the specified output format
    # output_data = str(value) + ' ' + str(0) + '\n'
    # output_data += ' '.join(map(str, taken))
    # return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        pprint(solve_it(input_data))
    else:
        file_location = 'knapsack/data/ks_4_0'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
