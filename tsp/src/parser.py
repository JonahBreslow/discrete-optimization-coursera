from random import sample

from src.coordinate_node import CoordinateNode

class PointParser:
    def __init__(self, input_data, n_samples: int = None) -> None:
        self.n_samples = n_samples
        self.input_data = input_data
        self.lines = self._lines()
        self.node_count = self._node_count()

    def _lines(self):
        return self.input_data.split('\n')

    def _node_count(self) -> int:
        lines = self.lines
        return int(lines[0])
    
    def points(self):
        points = {}

        # generating samples
        if self.n_samples is not None:
            items = [i for i in range(1, self.node_count + 1)]
            items = sample(items, self.n_samples)
            pass

        for i in items:
            line = self.lines[i]
            parts = line.split()
            points[i-1] = CoordinateNode(i-1, (float(parts[0]), float(parts[1])))

        return points
