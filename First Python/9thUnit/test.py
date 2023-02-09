import matplotlib.pyplot as plt
import numpy as np
import os
# class Animal:
#     def __init__(self, name):
#         self.name = name
# a1 = Animal("Gabe")
# a2 = Animal("Judy")
# a3 = a2
# a3.name = "Anna"
# print(a1.name)
# print(a2.name)
# print(a3.name)
class Animal:
    def eat(self):
        print("Animal eats")
class Fish(Animal):
    # def eat(self):
    #     print("Fish eats")
    pass
class Shark(Fish, Animal):
    def eat(self):
        super().eat()
        print("Shark eats")
for a in [Animal(), Fish(), Shark()]:
    a.eat() 

class Animal:
    def bark(self):
        print("Bark!")
    
animal_1 = Animal()

animal_1.bark()