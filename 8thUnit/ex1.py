import math
class  Complex:
    # real, imaginary = self.real, None
    def __init__(self, real: float, imaginary: float):
        self.real = real
        self.imaginary = imaginary  

    def __abs__(self) -> float:
        return math.sqrt(self.real**2 + self.imaginary**2)

    def __eq__(self, other):
        if isinstance(other, Complex):
            return self.real == other.real and self.imaginary == other.imaginary
        return NotImplemented

    def __repr__(self):
        return f"Complex(real={self.real}, imaginary={self.imaginary})"

    def __str__(self):
        sign=""
        if self.imaginary >= 0: sign = "+"
        return f"{self.real}{sign}{self.imaginary}i"
        
    def __add__(self, other):
        if isinstance(other, Complex):
            return Complex(self.real + other.real, self.imaginary + other.imaginary)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Complex):
            return self + other
        return NotImplemented

    @staticmethod
    def add_all(comp: "Complex", *comps: "Complex") -> "Complex":
        try:
            if not (any(isinstance(x, Complex) for x in comps)): raise TypeError
            for comp_i in comps:
                comp += comp_i
            return comp
        except TypeError:
            print('TypeError : can only add %s, not %s' % ("Complex", type(comps[0])) )   

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