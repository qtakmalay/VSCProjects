class Shape:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def to_string(self) -> str:
        return f"{type(self).__name__}: x={self.x}, y={self.y}"

    def area(self) -> float:
        raise NotImplementedError

