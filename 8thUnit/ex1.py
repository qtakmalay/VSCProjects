import math
class  Complex:
    # real, imaginary = self.real, None
    def __init__(self, real: float, imaginary: float):
        self.real = real
        self.imaginary = imaginary  

    def abs(self) -> float:
        return math.sqrt(pow(self.real, 2) + pow(self.imaginary, 2))

    def __eq__(self, other):
        if isinstance(other, Complex):
            return self.real == other.real and self.imaginary == other.imaginary
        raise NotImplemented

    def __repr__(self):
        raise NotImplementedError
    def __str__(self):
        sign=""
        if self.imaginary > 0: sign = "+"
        print(f"{self.real}{sign}{self.imaginary}i")
    def __add__(self, other):
        raise NotImplementedError
    def __iadd__(self, other):
        raise NotImplementedError
    def add_all(comp: "Complex", *comps: "Complex") -> "Complex":
        raise NotImplementedError

c1 = Complex(-1, -2)
c2 = Complex(2, 4)
c3 = Complex(1, 2)
print(c1 == c3, c1 + c2 == c3)
print(repr(c1))
print(c1)
print(abs(c1))
print(c1 + c2)
c1 += c3
print(c1)
print(Complex.add_all(c2, c2, c3))
try:
    c1 + 1
except TypeError as e:
    print(f"{type(e).__name__}: {e}")