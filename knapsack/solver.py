#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from pprint import pprint
import re

from sqlalchemy import false

Item = namedtuple("Item", ['index', 'value', 'weight', 'norm_value'])

def sort_tuple(tup: tuple, idx: int, reversed: bool):
    """sorts a tuple

    Args:
        tup (tuple): tuple to sort
        idx (int): index to sort tuple on
        reversed (bool): a boolean whether to reverse the sort or not
    """
    return(
        sorted(
            tup, 
            key = lambda x: x[idx],
            reverse = reversed
            )
        )

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

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

    """
    a trivial algorithm for filling the knapsack
    greedily looks for best option between:
    1. most valuable
    2. least heavy
    3. best value / weight
    
    `grid` is a bunch of tuples. Example:
    (1, True) -> 1 means we order by value, and True means it's descending
    (2, False) -> 2 means we order by weight, and False means it's ascending
    (3, True) -> 3 means we order by value / weight, True is descending
    """
    grid = [(1, True), (2, False), (3, True)]
    best_greedy_value = 0
    
    for _, enum in enumerate(grid):
        next_greedy_value, weight, next_taken = 0, 0, [0]*len(items)
        items_sorted = sort_tuple(items, enum[0], enum[1])

        for item in items_sorted:
            if weight == capacity:
                break
            elif weight + item.weight <= capacity:
                weight += item.weight
                next_taken[item.index] = 1
                next_greedy_value += item.value
                
        if next_greedy_value > best_greedy_value:
            best_greedy_value = next_greedy_value
            best_taken = next_taken
            
    value = best_greedy_value
    taken = best_taken
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

