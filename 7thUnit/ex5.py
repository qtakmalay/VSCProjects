from ex3 import Shape
import math
class Circle(Shape):
    def __init__(self, x: int, y: int, radius: int):
        super().__init__(x, y)
        self.radius = radius
    
    def to_string(self) -> str:
        return f"{type(self).__name__}: x={self.x}, y={self.y}, radius={self.radius}"
    
    def area(self) -> float:
        return float(pow(self.radius, 2) * math.pi)