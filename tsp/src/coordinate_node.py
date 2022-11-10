
class CoordinateNode():
    def __init__(self, name, coords):
        self.name = name
        self.coords = coords
        self.x = self.coords[0]
        self.y = self.coords[1]
    def __repr__(self):
        return self.name