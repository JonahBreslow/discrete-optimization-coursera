import math
import networkx as nx
import matplotlib.pyplot as plt

def length(point1, point2) -> float:
    """Computes the euclidean distance between 2 points

    Args:
        point1 (tuple): cartesian coordinates of point 1
        point2 (tuple): cartesian coordinates of point 2

    Returns:
        float: L2 norm (euclidean norm) between 2 points
    """
    return math.sqrt(
        (point1['coords'][0] - point2['coords'][0])**2 + 
        (point1['coords'][1] - point2['coords'][1])**2)

def edges(path):
    """converts a path list of tuples into edges for networkx graph

    Args:
        path (list): a list of tuples (point, distance)

    Returns:
        list: _description_
    """
    edges = []
    for idx in range(len(path) - 1):
        edges.append((path[idx][0], path[idx+1][0], path[idx+1][1]))
    
    return edges
