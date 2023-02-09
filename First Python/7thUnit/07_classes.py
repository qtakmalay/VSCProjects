# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
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

In this file, we will learn about classes in Python.
https://docs.python.org/3/tutorial/classes.html
"""

################################################################################
# Classes
################################################################################

# As we have seen with functions, the ability to effectively reuse code can
# benefit the program design. Object-oriented programming tries to achieve a
# higher level of reusability and modularity based on the idea of "objects".
# Such objects can hold data (referred to as "fields", "attributes" or
# "properties") and procedures/functions (referred to as "methods").

# In Python, this concept is applied using "classes" to create a class of an
# object, e.g., the class "Dog", and "instances", which are object instances
# from the class, e.g., an instance of "Dog" with name "bello" and another
# instance of "Dog" with name "barky". We actually already worked with classes
# and instances/objects the entire time, e.g., by using something like "x = 5",
# we let "x" refer to an instance/object of the built-in class "int" (note,
# however, that the built-in classes and objects get special treatment, which
# makes it not directly evident that we are indeed dealing with objects).


################################################################################
# Class definition and instance/object creation
################################################################################

# Classes are defined via the "class" statement. Class names should use CapWords
# format. Instances of classes should use lower_case format, i.e., analogous to
# function and variable names.

# Let's create a class "Dog":
class Dog:
    
    def __init__(self, name):
        # The "__init__" method will be executed at instantiation of the class.
        # As all object-functions (=methods), it has to take the "self" object
        # as first argument. "self" will give us access to the attributes and
        # methods within the instance. The name "self" is a convention.
        
        # Objects defined within the functions of the class are only available
        # in the function scope (like a normal variable in a function; see the
        # section on namespaces and scopes in the unit on functions)
        some_variable = 5  # This variable exists only within this function
        
        # However, they can be made available by turning them into attributes.
        # The syntax for creating an attribute is self.varname = var
        self.name = name  # Create an attribute "name"
        
        print(f"Created {self.name}")


# We can now create instances of our "Dog" class by "calling" the class via the
# so-called object constructor ("__init__" is not a constructor, since object
# construction in Python is actually done in two parts: First, a new instance of
# the class is created in the special method "__new__". Second, the "__init__"
# method is called to initialize this new instance appropriately):
dog1 = Dog(name="bello")  # This creates an initialized instance of our class

# We can now access the instance and its attributes:
print(f"dog1: {dog1}")
print(f"dog1.name: {dog1.name}")

# We can also extract the type/class itself via the built-in "type":
# https://docs.python.org/3/library/functions.html#type
print(type(dog1))
print(type(dog1).__name__)

# We can create another instance of our "Dog" class:
dog2 = Dog(name="barky")
print(f"dog2: {dog2}")
print(f"dog2.name: {dog2.name}")

# Note that "dog1" and "dog2" are not the same object. They are two different
# instances/objects, each pointing to different memory locations:
print(f"dog1 is dog2? answer: {dog1 is dog2}")


# We can add methods (=functions of the class and its objects) to our class:
class Dog:
    
    def __init__(self, name):
        self.name = name
    
    def communicate(self):
        # Here, we defined another method available through the class instances.
        # Note that the "self" object has to be the first argument again.
        
        # Let's say we want the dog to bark:
        communication = "bark"
        # We also want to include the name of the dog in the communication. For
        # this, we can access the "name" attribute that we created as follows:
        communication = f"{self.name}: {communication}"
        return communication


# Again, we create instances of our new "Dog" class:
dog1 = Dog(name="bello")  # This will create an instance of our class
dog2 = Dog("barky")  # This will create another instance of our class

# We can now access the methods the same way we accessed the attributes:
print(dog1.communicate())  # Effectively calls: Dog.communicate(dog1)
print(dog2.communicate())

# We already used many methods, e.g., "some_string.upper()" with "some_string"
# being an instance of the built-in class "str" and "upper" one of its methods.
# Another example would be "some_list.append(...)" with "some_list" being an
# instance of the built-in class "list" and "append" one of its methods.


################################################################################
# Inheritance
################################################################################

# Let's say we want to add a class "Cat" and "Frog" that also have an attribute
# "name" and a method "communicate". However, we want to change what the
# "communicate" method does for "Cat" and "Frog". To avoid code duplication, we
# make use of "inheritance", where we can reuse code in so-called subclasses
# that inherit code from base classes/superclasses. This creates a type/class
# hierarchy of base classes and subclasses.

# First, we define a base class that we call "Animal":
class Animal:
    
    def __init__(self, name):
        self.name = name
    
    def communicate(self):
        # As a generic animal, we do not really know how to communicate, this
        # should be implemented by concrete subclasses. Our "communicate" method
        # is now called an abstract method, it does not have an (actual)
        # implementation. This can be indicated by raising a NotImplementedError
        raise NotImplementedError


# Then, we can define a class "Dog" that inherits from the class "Animal". By
# doing this, we will have a class "Dog" that contains the same attributes and
# methods as the base class "Animal" does. If we want to add or modify the
# methods/attributes, we can simply override them:
class Dog(Animal):
    # We do not need a new "__init__" method since the "__init__" method from
    # the "Animal" base class already does everything what we want it to do.
    
    # Here, we override the method "communicate" by a new version:
    def communicate(self):
        return f"{self.name}: bark"


# We can do the same for the classes "Cat" and "Frog":
class Cat(Animal):
    
    def communicate(self):
        return f"{self.name}: meow"


class Frog(Animal):
    
    def communicate(self):
        return f"{self.name}: croak"


# We can now again create and use instances of our classes "Dog", "Cat" and
# "Frog" that were all derived from the class "Animal":
animal1 = Dog(name="bello")
animal2 = Cat(name="scratchy")
animal3 = Frog(name="jumper")

print(animal1.communicate())
print(animal2.communicate())
print(animal3.communicate())


# We could again derive a new class from our classes. Let's say we want to add a
# class "SouthernLeopardFrog" that is a more specialized frog type:
class SouthernLeopardFrog(Frog):
    
    def communicate(self):
        return f"{self.name}: screech"


animal4 = SouthernLeopardFrog(name="froggy")
print(animal4.communicate())


# We can also add new methods when deriving from a class:
class Bird(Animal):
    
    def communicate(self):
        return f"{self.name}: peep"
    
    # Here, we add a new method "fly":
    def fly(self, distance):
        return f"{self.name} flew {distance}km!"


animal5 = Bird(name="peeps")
print(animal5.communicate())
print(animal5.fly(500))


# We can also choose not to override anything if we are already satisfied with
# the implementation in the superclass. If we do not override, simply the first
# found implementation will be used when going upwards the class hierarchy:
class Canary(Bird):
    
    def communicate(self):
        return f"{self.name}: *sings song*"
    
    # "fly" method of superclass is already satisfactory


animal6 = Canary(name="Joe")
# Uses the first found implementation of "fly" when going upwards the class
# hierarchy, which is the implementation in class "Bird":
print(animal6.fly(10))


# Note: Every class automatically inherits from the base class "object", which
# is the root of the class hierarchy. This can be implemented explicitly via
# "class MyClass(object)" but can be omitted, as it is done regardless.


################################################################################
# Instance checks
################################################################################

# In some cases, it might be useful to check whether some identifier refers to
# an object of a certain class. For this, the built-in function "isinstance"
# can be used, which checks if an object is an instance of a class or of a
# superclass thereof. For more details, see
# https://docs.python.org/3/library/functions.html#isinstance
# If you want to check whether a class (rather than an object) is some class or
# subclass, you can use the function "issubclass". For more details, see
# https://docs.python.org/3/library/functions.html#issubclass

# We use the same type hierarchy from above:
#
#                    Animal
#                      ^
#                      |
#           ----------------------
#           |      |      |      |
#         Frog    Dog    Cat    Bird
#           ^                    ^
#           |                    |
#   SouthernLeopardFrog        Canary
#
# Here are some examples of checking one type (you can provide multiple types to
# check if any of them matches by providing a tuple of types):
my_animal = Animal("animal")
my_cat = Cat("cat")
my_frog = Frog("frog")
my_sl_frog = SouthernLeopardFrog("sl_frog")
print(isinstance(my_animal, Animal))
print(isinstance(my_animal, Cat))
print(isinstance(my_cat, Animal))
print(isinstance(my_cat, Cat))
print(isinstance(my_cat, Frog))
print(isinstance(my_cat, SouthernLeopardFrog))
print(isinstance(my_frog, Animal))
print(isinstance(my_frog, Frog))
print(isinstance(my_frog, SouthernLeopardFrog))
print(isinstance(my_sl_frog, Animal))
print(isinstance(my_sl_frog, Frog))
print(isinstance(my_sl_frog, SouthernLeopardFrog))
print(isinstance(my_sl_frog, str))  # Checking against built-in type/class "str"
print(isinstance(my_sl_frog, (int, float)))  # Check if either "int" or "float"
print(isinstance(my_sl_frog, object))


################################################################################
# Some more details
################################################################################

class MyClass:
    
    # Variables defined here will be shared by all instances/objects and are
    # accessible like attributes (also by the class itself). They are called
    # class attributes.
    var = 55
    
    # Mutable objects are dangerous here, as we will see below
    weird_list = []
    
    def __init__(self, a):
        self.a = a
        
        # A method ("__init__") may call other methods ("some_method")
        self.some_method()
        
        # Class-private members (attributes and methods) can be indicated with
        # underscores, (see
        # https://docs.python.org/3/tutorial/classes.html#private-variables
        # and https://dbader.org/blog/meaning-of-underscores-in-python for
        # general underscore details). Example where, by convention, users
        # should not access this variable from the outside:
        self._do_not_access = 4
        
    def some_method(self):
        # The class attribute "var" will be the same for every object (it is
        # shared across all objects)
        print(f"MyClass says: {self.a * MyClass.var}")
    
    # Just like class attributes, there can also be class methods, i.e., methods
    # that do not belong to an instance/object but rather the class. Since there
    # is no longer a dedicated instance/object, the "self" parameter is replaced
    # with a "cls" parameter (again, the name is just a convention), which then
    # contains the class instead of the object, when the method was invoked
    # (this invocation can either be done through an object or through the
    # class). To mark a method as class method, a so-called decorator has to be
    # used on top of the method definition: @classmethod
    # https://docs.python.org/3/library/functions.html#classmethod
    @classmethod
    def some_class_method(cls):
        print(f"this is a class method: {cls}")

    # There can also be static methods that take no object or class parameter at
    # all. In this case, the object or the class at the invocation will simply
    # be ignored entirely. To mark a method as static method, a so-called
    # decorator has to be used on top of the method definition: @staticmethod
    # https://docs.python.org/3/library/functions.html#staticmethod
    @staticmethod
    def some_static_method():
        print("this is a static method")


# Create instances and note that "__init__" calls the method "some_method":
instance1 = MyClass(a=1)
instance2 = MyClass(a=2)

# You can access the class attribute via instances/objects as well, however,
# this should be avoided to not confuse object attributes with class attributes
print(f"MyClass.var: {MyClass.var}")  # Recommended class.attribute notation
print(f"instance1.var: {instance1.var}")  # Misleading object.attribute notation
print(f"instance2.var: {instance2.var}")

# "Overriding" the class attribute "var" via an instance/object actually creates
# a new object attribute for this instance/object with the same name:
instance2.var = 4
print(f"MyClass.var: {MyClass.var}")  # Still the same value
print(f"instance1.var: {instance1.var}")  # Still refers to the class attribute
print(f"instance2.var: {instance2.var}")  # New attribute "var" for "instance2"

# However, be careful when modifying elements of mutable objects (e.g., lists,
# dictionaries) that are shared between instances, since this will affect all
# instances (compare mutable default arguments in functions):
print(f"MyClass.weird_list: {MyClass.weird_list}")
print(f"instance1.weird_list: {instance1.weird_list}")
print(f"instance2.weird_list: {instance2.weird_list}")
# "instance2.weird_list.append(...)" and "MyClass.weird_list.append(...)" are
# the same, as they all refer to the same class attribute, which is this
# modifiable list
instance1.weird_list.append("element")
print(f"MyClass.weird_list: {MyClass.weird_list}")
print(f"instance1.weird_list: {instance1.weird_list}")
print(f"instance2.weird_list: {instance2.weird_list}")
instance3 = MyClass(a=3)
print(f"instance3.weird_list: {instance3.weird_list}")

# Example of invoking a class method (analogous for static methods):
MyClass.some_class_method()  # Recommended class.method() notation
instance1.some_class_method()  # Misleading object.method() notation


################################################################################
# Modifying "__init__" and overriding an existing method
################################################################################

class DerivedClass(MyClass):
    
    def __init__(self):
        # If we modify "__init__", we have to call "__init__" of our parent
        # class at some point. This is important, since the "__init__" method
        # essentially sets up/creates an instance that should be "finished"
        # afterwards. When initializing a derived class, we simply need to call
        # the "__init__" of the superclass, which handles all initialization
        # of all base-class-related data. The only thing we need to do is deal
        # with our own data, i.e., the data of our newly derived class. If we
        # always follow this principle (=ensuring that the base class is
        # properly initialized by calling its "__init__" method), we guarantee
        # that the instance of our newly derived class is properly initialized
        # as well. The built-in "super" helps us here, since it will return the
        # class that we have derived the current class from:
        super(DerivedClass, self).__init__(a=4)  # calls MyClass.__init__(a=4)
        # A shorter notation would be to simply write "super().__init__(a=4)";
        # the two arguments will be inserted automatically.
    
    def some_method(self):
        # Here, we override the method from the base class with new behavior:
        print(f"DerivedClass says: {self.a * MyClass.var}")
        # If we want to include the behavior of the base class method, we can do
        # this again with "super" (uncomment to see the effect):
        # super().some_method()


# Create instances and note that "__init__" calls the method "some_method":
instance1 = MyClass(a=1)
instance2 = DerivedClass()

print(f"instance1.some_method():")
instance1.some_method()
print(f"instance2.some_method():")
instance2.some_method()

#
# Caution
#

# As we just saw above, creating an instance with "DerivedClass()" invokes the
# "MyClass.__init__" method, which, in turn, calls the method "some_method".
# Since we are dealing with a "DerivedClass" instance and overrode this
# particular method, "DerivedClass.some_method" is executed instead of
# "MyClass.some_method". Such intra-class method calls (especially in case of
# the "__init__" method calling other methods) can potentially break your code.


class ProblematicDerivedClass(MyClass):
    
    def __init__(self, a, b):
        # Initialize all base classes, so everything in the inheritance
        # hierarchy above our newly derived class is properly set up
        super().__init__(a)
        # Afterwards, initialize our own object data, making the instance
        # creation of a "ProblematicDerivedClass" object complete
        self.b = b
    
    def some_method(self):
        # While we expect this to work (we correctly initialized everything in
        # our "__init__" method, so everything should be properly set up), we
        # will actually run into an error because "super().__init__(a)" invokes
        # the "some_method" during object initialization. Since we overrode this
        # method here, this means that we actually execute this implementation.
        # However, the object initialization of "ProblematicDerivedClass" is not
        # yet finished (we are still in the super-initialization), and the
        # attribute "self.b" does not exist yet.
        print(f"DerivedClass says: {self.a + self.b}")


# Creating an instance of "ProblematicDerivedClass" will fail:
ProblematicDerivedClass(a=1, b=2)

# How can we solve such problems? There are multiple ways, depending on what you
# ultimately need (not all are always possible/make sense):
# 1) Move the "super().__init__" call to the end of "__init__".
# 2) Utilize Python's name mangling mechanism to make the corresponding method
#    "private", so it cannot be overridden in subclasses (for more details, see
#    https://docs.python.org/3/tutorial/classes.html#private-variables).
# 3) Avoid instance methods and use class or static methods instead, including
#    the proper/recommended way of calling them via "MyClass.some_class_method"
#    or "MyClass.some_static_method" (not via "self.some_class/static_method").
# 4) Avoid such method calls in "__init__" entirely.
