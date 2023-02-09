# -*- coding: utf-8 -*-
"""
Author -- Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 04.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

Example solutions for tasks in the provided tasks file.
"""


# Given the "Fraction" implementation below which models a mathematical
# fraction, your tasks will be to extend the functionality of this class.


class Fraction:
    
    def __init__(self, numerator: int, denominator: int):
        self.n = numerator
        self.d = denominator
    
    #
    # Task 1
    #
    
    # Extend the "Fraction" class by providing the special method "__eq__" to
    # enable equality checks on "Fraction" objects. A "Fraction" should be
    # considered equal to another "Fraction" if and only if they have the same
    # numerators and denominators, i.e., fractions such as 1/2 and 2/4 are not
    # considered equal. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__eq__
    def __eq__(self, other):
        pass
    
    #
    # Task 2
    #
    
    # Extend the "Fraction" class by providing the special method "__str__" to
    # get a human-friendly string representation of a fraction. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__str__
    def __str__(self):
        pass
    
    #
    # Task 3
    #
    
    # Extend the "Fraction" class by providing the special method "__add__" to
    # add another "Fraction" object or an integer object. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__add__
    def __add__(self, other):
        pass
    
    #
    # Task 4
    #
    
    # Extend the "Fraction" class by providing the special method "__mul__" to
    # multiply another "Fraction" object or an integer object. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__mul__
    def __mul__(self, other):
        pass
    
    #
    # Task 5
    #
    
    # Extend the "Fraction" class by providing the special method "__float__" to
    # convert a "Fraction" object to a float object. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__float__
    def __float__(self):
        pass


# Some testing code including output and assertions

# Printing
print(Fraction(12, 2500))

# Equality
f1 = Fraction(1, 2)
f2 = Fraction(1, 2)
f3 = Fraction(1, 3)
f4 = Fraction(2, 2)
f5 = Fraction(2, 4)
assert f1 == f2
assert f1 != f3
assert f1 != f4
assert f1 != f5  # Although the evaluated values are the same, the numerators and denominators are not

# Addition
assert f1 + f2 == Fraction(4, 4)
assert f1 + f3 == Fraction(5, 6)
assert Fraction(-1, 1) + Fraction(1, 1) == Fraction(0, 1)
assert Fraction(1, -1) + Fraction(1, 1) == Fraction(0, -1)
assert f1 + 2 == Fraction(5, 2)
try:
    f1 + 2.0
except TypeError:
    pass  # Expected, so fail silently
else:
    assert False

# Multiplication
assert f1 * f1 == Fraction(1, 4)
assert f4 * f5 == Fraction(4, 8)
assert Fraction(-1, 1) * Fraction(1, -1) == Fraction(-1, -1)
assert f1 * 2 == f4
try:
    f1 * 2.0
except TypeError:
    pass  # Expected, so fail silently
else:
    assert False

# Float conversion
assert float(f1) == 0.5
assert float(f4) == 1.0


# Given the "MinMaxCollector" implementation below which collects the running
# minimum and maximum of passed arguments, your tasks will be to extend the
# functionality of this class.


class MinMaxCollector:
    
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    #
    # Task 6
    #
    
    # Extend the "MinMaxCollector" class by providing the special method
    # "__call__" to make a "MinMaxCollector" object callable. The implementation
    # only needs to work on positional arguments. These positional arguments can
    # be assumed to be comparable (with comparison operators), and you can also
    # assume that the user will only input the same data type. Your task will be
    # to continuously update the instance attributes "min_" and "max_"
    # accordingly, given the passed arguments. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__call__
    def __call__(self, *args):
        pass


# Some testing code including assertions
mmc = MinMaxCollector()
assert mmc.min_ is None and mmc.max_ is None
mmc()
assert mmc.min_ is None and mmc.max_ is None
mmc(5)
assert mmc.min_ == 5 and mmc.max_ == 5
mmc(1, 2, 3)
assert mmc.min_ == 1 and mmc.max_ == 5
mmc(7)
assert mmc.min_ == 1 and mmc.max_ == 7


# You are given the "IndexDict" implementation below which models a dict that
# can also be indexed like a regular sequence. The indexing happens through the
# custom attribute "index", which contains an instance of the class "Indexer".
# Your tasks will be to extend the functionality of this class.


class IndexDict(dict):
    # Inner helper class that wraps an "IndexDict" object and returns items
    # (key-value pairs) from this wrapped dictionary using "__getitem__".
    class Indexer:
        
        def __init__(self, index_dict: "IndexDict"):
            self._index_dict = index_dict
        
        #
        # Task 7
        #
        
        # Extend the "Indexer" class by providing the special method
        # "__getitem__" to make an "Indexer" object support the index operator
        # "[key"]. Only integer objects need to be supported as "key", i.e.,
        # only plain integer indices. Such an index then used to return the
        # "key"-th item (dictionary key-value pair) in the wrapped "IndexDict"
        # dictionary, which is stored in the attribute "_index_dict".
        # Specification:
        # https://docs.python.org/3/reference/datamodel.html#object.__getitem__
        def __getitem__(self, key):
            pass
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = IndexDict.Indexer(self)


# Some testing code including assertions
d = IndexDict(a=1, b=2)
assert d.index[0] == ("a", 1)
assert d.index[-1] == ("b", 2)
try:
    d.index["a"]
except TypeError:
    pass  # Expected, so fail silently
else:
    assert False
try:
    d.index[100]
except IndexError:
    pass  # Expected, so fail silently
else:
    assert False
