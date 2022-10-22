from collections import namedtuple
from sqlalchemy import false
from src.greedy import GreedyKnapsackAlgorithm

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

    # Instantiating the greedy knapsack algorithm class
    best_knapsack = GreedyKnapsackAlgorithm(items,capacity=capacity)
    
    '''
    Search greedily over
    1. maximum value
    2. minumum weight
    3. maximum value density (value / weight)
    '''
    grid = [(1, True), (2, False), (3, True)]
    
    optimal_knapsack = best_knapsack.greedy_choose(grid)
            
    value = optimal_knapsack.value
    taken = optimal_knapsack.taken
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

