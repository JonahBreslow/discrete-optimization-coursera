from dataclasses import dataclass
from typing import List



@dataclass
class ChosenKnapsack:
    value: int
    taken: List


class GreedyKnapsackAlgorithm:
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
        """Sorts a tuple on an index chosen

        Args:
            idx (int): index of tuple to sort on
            reversed (bool): If `False`, sorts in ascending order.
            If `True`, sorts in descending order
            
        Returns:
            List: the possible items sorted by the index provided
        """
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
            Example:
            (1, True) -> 1 means we order by value, and True means it's descending
            (2, False) -> 2 means we order by weight, and False means it's ascending
            (3, True) -> 3 means we order by value / weight, True is descending

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