# -*- coding: utf-8 -*-
"""
Author -- Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 01.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will learn some more about classes in Python.
"""

################################################################################
# Classes - advanced topics
################################################################################

# With classes, we can define our own attributes and methods, but we can go even
# further and implement existing methods that will increase the power and
# functionality of our own custom classes. Most notably, we will be making use
# of methods already available in the base class "object", and we will implement
# special methods to provide support for typical Python-like programming such as
# indexing, iteration, etc.

# We will only cover parts of Python's data model, the full information is here:
# https://docs.python.org/3/reference/datamodel.html

# More information on Python's built-in types:
# https://docs.python.org/3/library/stdtypes.html

# We will need this imported class later, but here is the documentation anyway:
# https://docs.python.org/3/library/collections.abc.html
from collections.abc import Sequence


#
# Overriding methods defined in the base class "object"
#

# We already know that every class in Python derives from the root class called
# "object". While it does not have any attributes, it does have some interesting
# methods that we can override to adapt their behavior to our custom class.
# Let's assume we want to create a class that represents coordinates or points
# in a 2D-plane, i.e., something like (x, y).
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


# We can now create "Point" objects. Let's create two such objects with the same
# x- and y-coordinates. We know that two independent objects will be created, so
# checking whether they are the same (identical) object naturally returns False.
p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1 is p2)

# However, it would be nice if we could compare these two points based on their
# x and y values (rather than their object identity), where we would expect that
# "p1" and "p2" are equal.
print(p1 == p2)


# We get False above, because the default behavior of the equals operator "=="
# of the base class "object" simply falls back to comparing object identity.
# Luckily, we can change this behavior by overriding the special method
# "__eq__", which is invoked when comparing two objects with "==" (this is
# called operator overloading).
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Here, we provide a custom implementation of the method "__eq__", which
    # will then be used when we compare "Point" objects with "==". When
    # overriding such special methods, we must ensure that we adhere to the
    # requirements that are listed in the specification to avoid unexpected or
    # incorrect behavior:
    # https://docs.python.org/3/reference/datamodel.html#object.__eq__
    #
    # Note: When returning "NotImplemented" instead of False, we allow Python to
    # fall back to the "__eq__" method of the "other" object. This can be
    # important in case of subclasses that also define the "__eq__" method if we
    # want to retain equality comparison symmetry. Example of such a symmetry:
    # https://stackoverflow.com/a/9844427/8176827
    # More details on the special value "NotImplemented":
    # https://docs.python.org/3/library/constants.html#NotImplemented
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented


# Now, we can actually compare two "Point" objects based on their content rather
# than their object identity.
p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1 == p2)  # Will invoke "Point.__eq__(self=p1, other=p2)"

# Let's look at the representation of our points when we display them in the
# console (execute the following line in your Python console):
# p1
# It will look something like this: <__main__.Point at 0x1c40a893af0>


# Naturally, we want to change this default representation provided by the base
# class "object", which can easily be done by overriding the method "__repr__".
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented
    
    # According to the specification, this method should return a string
    # representation that is ideally a Python expression which can be used to
    # recreate this object. For our simple "Point" class, this is
    # straightforward (check in the console again). Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__repr__
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"


# Next, we also want to print our points.
p1 = Point(1, 2)
print(p1)


# This will simply use the implementation provided in the "__repr__" method, but
# there is yet another method which we can customize: the "__str__" method.
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    # This should return a human-friendly string representation. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__str__
    def __str__(self):
        return f"({self.x}, {self.y})"


# Finally, we get a nice and clean print output.
p1 = Point(1, 2)
print(p1)  # Will invoke "Point.__str__(self=p1)"


#
# Special methods not part of "object"
#

# We can already do some nice stuff with our "Point" objects, but we can go even
# further by implementing other special methods that will be automatically used
# by Python for certain functionality such as indexing, iteration, etc. Again,
# we only have to implement the corresponding methods (according to their
# specification and requirements) and the rest is done for you. Note that these
# special methods are NOT part of the base class "object". Python just assumes
# that if your class has a matching implementation, your class will then support
# functionality like indexing and so on.

# Let's start by adding indexing support to our "Point" class. For simplicity's
# sake, we will just include the "get at index"-functionality, i.e., users can
# retrieve data from our Point objects not only via the attributes "x" and "y",
# but also via indexing, e.g., p1[0] to retrieve the first element. All we have
# to do is provide an appropriate implementation for the method "__getitem__".
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    # If a class implements the "__getitem__" method, Python assumes that it now
    # provides indexing functionality. Again, the implementation is up to us,
    # and we have to make sure to adhere to the requirements that are listed in
    # the specification to avoid unexpected or incorrect behavior:
    # https://docs.python.org/3/reference/datamodel.html#object.__getitem__
    #
    # Note: This is just for demonstrating purposes. It does not really make
    # sense for our Point class to have indexing support. Moreover, we do not
    # check for "key" being of type "slice" either (would make even less sense),
    # which, however, would be OK since the general recommendation is to create
    # an implementation "to the degree that it makes sense for the object being
    # modelled".
    # https://docs.python.org/3/reference/datamodel.html#special-method-names
    def __getitem__(self, key):
        if isinstance(key, int):
            if key == 0 or key == -2:
                return self.x
            if key == 1 or key == -1:
                return self.y
            raise IndexError("Point index out of range")
        raise TypeError(f"Point indices must be integers, not {type(key).__name__}")


# Indexing example:
p1 = Point(1, 2)
print(p1[0])  # Will invoke "Point.__getitem__(self=p1, key=0)"


# Next, we want to enable the add operation with "+", so we can easily add two
# "Point" objects together with "p3 = p1 + p2". For this, we want to provide an
# implementation of the "__add__" method, which will automatically be invoked
# when we use the "+" operator on our "Point" objects.
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __getitem__(self, key):
        if isinstance(key, int):
            if key == 0 or key == -2:
                return self.x
            if key == 1 or key == -1:
                return self.y
            raise IndexError("Point index out of range")
        raise TypeError(f"Point indices must be integers, not {type(key).__name__}")
    
    # Again, we are free to choose our own implementation as long as it aligns
    # with the specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__add__
    #
    # Note: Here, we actually provide multiple ways how we can make an addition:
    # We allow another "Point" object to be added to this "Point" object with
    # (p1 + p2), we allow some number to be added (p1 + 123.4), and we allow
    # some sequence to be added, as long as it has exactly two elements and
    # these two elements are again numbers.
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        if isinstance(other, (int, float)):
            return Point(self.x + other, self.y + other)
        # Convenient way of checking whether "other" is a sequence type, where
        # sequence is defined as "Sequence" in the following documentation:
        # https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence
        # For other "collection" types, see this link:
        # https://docs.python.org/3/library/collections.abc.html
        #
        # Side note in case of interest: Checking whether "other" is some sort
        # of sequence beforehand and then accessing the corresponding
        # attributes/methods/functionality is known as LBYL (look before you
        # leap). The "collections.abc" module was introduced as a convenient way
        # of such a lookup instead of having to manually (and possibly multiple
        # times) check using the built-in function "hasattr". The LBYL style is
        # in direct contrast to EAFP (easier to ask for forgiveness than
        # permission), where the corresponding attributes, etc. are simply
        # assumed to be there and directly accessed, leading to exceptions (that
        # should be handled) afterwards in case the attribute, etc. was not
        # available. Some more details in the Python glossary:
        # https://docs.python.org/3/glossary.html#term-LBYL
        # https://docs.python.org/3/glossary.html#term-EAFP
        # https://docs.python.org/3/glossary.html#term-abstract-base-class
        # https://docs.python.org/3/glossary.html#term-duck-typing
        # And some interesting discussions on this topic:
        # https://stackoverflow.com/q/1952464/8176827
        # https://stackoverflow.com/q/610883/8176827
        if isinstance(other, Sequence):
            if len(other) == 2 and all(isinstance(i, (int, float)) for i in other):
                return Point(self.x + other[0], self.y + other[1])
            else:
                raise ValueError("sequence must have exactly two numbers")
        return NotImplemented


# We can now add something to a "Point" object, which will return a new "Point"
# object (the one we returned in "__add__").
p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1 + p2)  # Will invoke "Point.__add__(self=p1, other=p2)"
print(p1 + 123.4)  # Will invoke "Point.__add__(self=p1, other=123.4)"
print(p1 + [100, 50])  # Will invoke "Point.__add__(self=p1, other=[100, 50])"

# Note that with swapped operands and different types, it does not work anymore,
# because now, the "__add__"" method of the other type is called rather than our
# method in the "Point" class.
print(123.4 + p1)  # Will invoke "float.__add__(self=123.4, other=p1)"


# However, we can still easily fix this by providing yet another method called
# "__radd__", which will be invoked automatically in precisely such cases.
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __getitem__(self, key):
        if isinstance(key, int):
            if key == 0 or key == -2:
                return self.x
            if key == 1 or key == -1:
                return self.y
            raise IndexError("Point index out of range")
        raise TypeError(f"Point indices must be integers, not {type(key).__name__}")
    
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        if isinstance(other, (int, float)):
            return Point(self.x + other, self.y + other)
        if isinstance(other, Sequence):
            if len(other) == 2 and all(isinstance(i, (int, float)) for i in other):
                return Point(self.x + other[0], self.y + other[1])
            else:
                raise ValueError("sequence must have exactly two numbers")
        return NotImplemented
    
    # We just reuse the normal operand order here, i.e., the normal "__add__"
    # method will be invoked. Specification:
    # https://docs.python.org/3/reference/datamodel.html#object.__radd__
    def __radd__(self, other):
        return self + other


# Now, we can also add something with swapped operands.
p1 = Point(1, 2)
print(123.4 + p1)
