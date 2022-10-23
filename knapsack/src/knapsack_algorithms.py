from dataclasses import dataclass
from typing import List
import time 
import math

import numpy as np



@dataclass
class ChosenKnapsack:
    value: int
    taken: List


class KnapsackAlgorithms:
    def __init__(self, tuples: List, capacity: int) -> None:
        """Basic knapsack algorithm

        Args:
            tuples (list): all the possible items to put in the knapsack
            capacity (int): the maximum capacity of the knapsack
        """
        self.tuples = tuples
        self._capacity = capacity
        self._length = len(tuples)

    def __len__(self):
        return self._length
        
    def sort_tuple(self, idx: int, reversed: bool) -> List:
        assert idx < self._length
        return(sorted(self.tuples, key = lambda x: x[idx],reverse = reversed))
            
    def greedy_choose(self, sort_grid: List) -> ChosenKnapsack:
        """chooses the optimal knapsack configuration over a grid of possible greedy choices.
        greedily looks for best option between:
        1. most valuable
        2. least heavy
        3. best value / weight

        Args:
            sort_grid (SortedGrid): A list of tuples that indicate which axes to
            sort on and how (either ascending or descending). 
            sort_grid is a bunch of tuples. 
        Returns:
            ChosenKnapsack: the optimal knapsack based on the algorithm chosed
        """
        
        best_greedy_value = 0
        for _, enum in enumerate(sort_grid):
            next_greedy_value, weight, next_taken = 0, 0, [0]*self._length
            items_sorted = self.sort_tuple(enum[0], enum[1])

            for item in items_sorted:
                if weight == self._capacity:
                    break
                elif weight + item.weight <= self._capacity:
                    weight += item.weight
                    next_taken[item.index] = 1
                    next_greedy_value += item.value
                    
            if next_greedy_value > best_greedy_value:
                best_greedy_value = next_greedy_value
                best_taken = next_taken
        return ChosenKnapsack(best_greedy_value,best_taken)
    
    def dynamic_choose(self):
        items = np.asarray(self.tuples, dtype=int)[:,0:3]
        st = time.monotonic()
        weight_array = np.array([cap for cap in items[:,2]])
        matrix_dimensions = (self._capacity+1, self._length+1)
        dynamic_matrix = np.zeros(matrix_dimensions, dtype=int)
        take_array = np.zeros(self._length, dtype=int)
        for capacity in range(1, matrix_dimensions[0]):
            for n_item in range(1, matrix_dimensions[1]):
                item_idx = n_item-1
                item_value = items[item_idx, 1]
                item_weight = items[item_idx, 2]
                # populate dynamic matrix
                if item_weight <= capacity:
                    excess_capacity = capacity - item_weight
                    value = np.max([
                        dynamic_matrix[capacity, n_item-1],
                        item_value + dynamic_matrix[excess_capacity, n_item-1]
                    ])
                    dynamic_matrix[capacity, n_item] = value
                elif item_weight > capacity:
                    dynamic_matrix[capacity, n_item] = dynamic_matrix[capacity, n_item-1]
                else:
                    pass
                
                
            # populate take_array
            if capacity == matrix_dimensions[0]-1:
                knapsack_cap = 0
                i = -1
                capacity_idx = -1
                # ensuring that the knapsack isn't full and that we don't search beyond 
                # the items we have in the possible search set
                while knapsack_cap < self._capacity and -1*i <= self._length:
                    # if bottom right element > element immediately to the left, then it was taken
                    take_array[i] = (dynamic_matrix[capacity_idx,i] > dynamic_matrix[capacity_idx, i-1])*1
                    # add weight of the item to the knapsack capacity only if it was taken
                    knapsack_cap += weight_array[i] * take_array[i]
                    # if not taken the capacity stays the same
                    if take_array[i] == 0:
                        capacity_idx = capacity_idx
                    # if taken, the remaining capacity reduces by the weight of the taken item
                    else:
                        capacity_idx = capacity_idx - weight_array[i]
                        
                    i -= 1
                
        et = time.monotonic()
        elapsed = et - st
    
        optimal_value = np.max(dynamic_matrix)
        return ChosenKnapsack(optimal_value, take_array)
    