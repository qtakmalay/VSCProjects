import math
class  Complex:
    # real, imaginary = self.real, None
    def __init__(self, real: float, imaginary: float):
        self.real = real
        self.imaginary = imaginary

    def print(self):
        sign=""
        if self.imaginary > 0: sign = "+"
        print(f"{self.real}{sign}{self.imaginary}i")

    def abs(self) -> float:
        return math.sqrt(pow(self.real, 2) + pow(self.imaginary, 2))

# c1 = Complex(1.2, -5.4)
# c1.print()
# c2 = Complex(3.0, 4.0)
# c2.print()
# print(c2.abs())