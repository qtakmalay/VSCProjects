from ex4 import Rectangle
from ex3 import Shape
from ex5 import Circle

class Square(Rectangle):
    def __init__(self, x: int, y: int, length: int):
        super(Square, self).__init__(x, y, length, length)
        self.length = length

    def to_string(self) -> str:
        return f"{type(self).__name__}: x={self.x}, y={self.y}, width={self.width}, height={self.height}"


    def area(self) -> float:
        return float(self.width * self.height)

# s = Shape(4, 9)
# print(s.to_string())
# r = Rectangle(1, 2, 3, 4)
# print(r.to_string())
# print("Rectangle area:", r.area())
# c = Circle(5, 2, 2)
# print(c.to_string())
# print("Circle area:", c.area())
# s = Square(0, 0, 10)
# print(s.to_string())
# print("Square area:", s.area())
