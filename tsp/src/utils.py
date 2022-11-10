import math

def length(point1, point2) -> float:
    """Computes the euclidean distance between 2 points

    Args:
        point1 (tuple): cartesian coordinates of point 1
        point2 (tuple): cartesian coordinates of point 2

    Returns:
        float: L2 norm (euclidean norm) between 2 points
    """
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

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