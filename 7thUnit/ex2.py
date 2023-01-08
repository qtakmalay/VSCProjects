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

    def add(self, other: "Complex"):
        try:
            self.real += other.real
            self.imaginary += other.imaginary
        except TypeError as ex:
            print(ex)
    @staticmethod
    def add_all(comp: "Complex", *comps: "Complex") -> "Complex":
        try:
            if not (any(isinstance(x, Complex) for x in comps)): raise TypeError
            comp = Complex(0.0,0.0)
            for comp_i in comps:
                comp.add(comp_i)
            return comp
        except TypeError:
            print('TypeError : can only add %s, not %s' % ("Complex", type(comps[0])) )    

c1 = Complex(1.0, -2.0)
c1.print()
c2 = Complex(9.0, 100.0)
c1.add(c2)
c1.print()
c_sum = Complex.add_all(c1, c1, c2, Complex(33.75, -14.25))
c_sum.print()
c1.print()
will_fail = Complex.add_all(100)



