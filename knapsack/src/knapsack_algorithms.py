import time
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import numpy as np


@dataclass
class ChosenKnapsack:
    value: int
    taken: List
    
@dataclass
class Node:
    item: int
    taken_items: np.array
    weight: int
    value: int
    idealized_value: int
    terminal: bool
    

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
        """
        #TODO write the doc string

        Returns:
            _type_: _description_
        """
        items = np.asarray(self.tuples, dtype=int)[:,0:3]
        weight_array = np.array([cap for cap in items[:,2]])
        matrix_dimensions = (self._capacity+1, self._length+1)
        dynamic_matrix = np.zeros(matrix_dimensions, dtype=int)
        take_array = np.zeros(self._length, dtype=int)
        for capacity in tqdm(range(1, matrix_dimensions[0])):
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
                
        optimal_value = np.max(dynamic_matrix)
        return ChosenKnapsack(optimal_value, take_array)
    
    def linear_relaxation_bound (
        self, 
        items_sorted: List,
        weight: int, value: int, 
        item_iterator: int
        ) -> int:
        while weight < self._capacity and item_iterator < len(items_sorted):
            # get item information
            item = items_sorted[item_iterator]
            item_weight, item_value = item.weight, item.value
            # how much knapsack capacity is left
            excess_capacity = self._capacity - weight
            # figuring out how much of the item we can insert into the knapsack
            if excess_capacity >= item_weight:
                pct_item = 1
            else:
                pct_item = excess_capacity / item_weight
            value += item_value * pct_item
            weight += item_weight * pct_item
            
            item_iterator += 1
        return value
    
    def branch_and_bound(self) -> ChosenKnapsack:
        """_summary_
        """
        # sorting all the items by value density (value / weight)
        items_sorted = self.sort_tuple(3, True)
        
        # iteratively take the best item and then use linear relaxation to 
        # fill the knapsack on the last item (if the whole item doesn't fit)
        max_knapsack_value = self.linear_relaxation_bound(
            items_sorted=items_sorted,
            weight=0,
            value=0,
            item_iterator=0
            )
        
        take_array = np.zeros(self._length, dtype=int)
        root_node = Node(
            item=0,
            taken_items=take_array,
            weight=0,
            value=0,
            idealized_value=max_knapsack_value,
            terminal=False
        )
        
        # setting things up for this awesome loop
        stack = [root_node]
        max_realized_value = 0
        best_taken_items = []
        while len(stack)>0:
            # get current node off the stack
            node = stack.pop()
            next_item_idx = node.item + 1  
    
            # if node is a terminal node, check to see if it is the best so far
            # if so, update the max_realized_value and best_taken_items objects
            if node.terminal == True:
                if node.value > max_realized_value:
                    max_realized_value = node.value
                    best_taken_items = node.taken_items
            # if the node has a current_idealized_value less than or equal to the current best,
            # ignore this node
            elif node.idealized_value <= max_realized_value:
                pass
                
            # if the node is not terminal and has the potential to beat the current best, and
            # the next item actually exists let's do it!

            elif next_item_idx <= self._length :
                # Finding the next item in the list of possibilities
                next_item_object = items_sorted[next_item_idx - 1]
                item_weight, item_value = next_item_object.weight, next_item_object.value
                
                # The node constructed by choosing the next item (assuming not terminal unless end of items)
                taken_items_array = node.taken_items.copy()
                taken_items_array[next_item_object[0]] = 1
                choose_node = Node(
                    item = next_item_idx,
                    taken_items = taken_items_array,
                    weight = node.weight + item_weight,
                    value = node.value + item_value,
                    idealized_value = node.idealized_value,
                    terminal = next_item_idx == self._length
                )
                
                # The node constructed by ignoring the next item (assuming not terminal unless end of items)
                # First, find new idealized_value with new options
                new_bound = self.linear_relaxation_bound(
                    items_sorted=items_sorted,
                    weight=node.weight,
                    value=node.value,
                    item_iterator=next_item_idx
                    )
                
                ignore_node = Node(
                    item = next_item_idx,
                    taken_items = node.taken_items,
                    weight = node.weight,
                    value = node.value,
                    idealized_value = min(
                        node.idealized_value,
                        new_bound
                        ),
                    terminal = next_item_idx == self._length
                )
                
                # if smallest of remaning items doesn't fit,
                # then any combo of them also will not fit. DO NOT PUT ON STACK
                if next_item_idx == self._length:
                    smallest_remaining_item_size = 0
                else: 
                    smallest_remaining_item_size = np.min([item[2] for item in items_sorted[next_item_idx:]])
                    
                smallest_remaining_capacity =self._capacity - node.weight
                if smallest_remaining_item_size > smallest_remaining_capacity:
                    if node.value > max_realized_value:
                        max_realized_value = node.value
                        best_taken_items = node.taken_items
                # if adding next item keeps weight under capacity, and the optimistic idealized weight 
                # is > the current max_realized_value, add both nodes to the stack
                elif node.weight + item_weight < self._capacity :
                    stack.append(ignore_node)
                    stack.append(choose_node)
                # if adding next item keeps weight exactly as capacity, and the optimistic idealized weight 
                # is > the current max_realized_value, add both nodes to the stack and mark as terminal
                elif node.weight + item_weight == self._capacity :
                    choose_node.terminal = True
                    stack.append(ignore_node)
                    stack.append(choose_node)
                
                # if adding next item esceeds weight capacity, and the optimistic idealized weight 
                # is > the current max_realized_value, add both nodes to the stack
                else:
                    stack.append(ignore_node)
            # if next item doesn't exist it is therefore terminal in a sense
            elif next_item_idx-1 == self._length:
                if node.value > max_realized_value:
                    max_realized_value = node.value
                    best_taken_items = node.taken_items
        
        return ChosenKnapsack(max_realized_value, best_taken_items)
            