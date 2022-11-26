# to sum it up the basics are : 
# datatypes, 
# generally construct a pogram, 
# how to index list/dictonary/…, 
# basic idea of built in functions, 
# (sum, min, max, return), 
# call a funktion, 
# get an elemnt of a list, 
# in/output,
# loops
import ex1, ex2, ex3, ex4, ex5
# print("EX1 ----------------------------------")
# for i in range(-2, 20):
#     print(f"fib({i}) = {ex1.fib(i)}")
# print("EX2 ----------------------------------")
# print(f"clip() = {ex2.clip()}")
# print(f"clip(1, 2, 0.1, 0) = {ex2.clip(1, 2, 0.1, 0)}")
# print(f"clip(-1, 0.5) = {ex2.clip(-1, 0.5)}")
# print(f"clip(-1, 0.5, min_=-2) = {ex2.clip(-1, 0.5, min_=-2)}")
# print(f"clip(-1, 0.5, max_=0.3) = {ex2.clip(-1, 0.5, max_=0.3)}")
# print(f"clip(-1, 0.5, min_=2, max_=3) = {ex2.clip(-1, 0.5, min_=2, max_=3)}")
# print("EX3 ----------------------------------")
# print(ex3.create_train_test_splits([], 0.5))
# print(ex3.create_train_test_splits(list(range(10)), 0.5))
# print(ex3.create_train_test_splits(list(range(10)), 0.67))
# print("EX4 ----------------------------------")

print(ex4.round_(777.5759823, 4)) # 778.0

# print("EX5 ----------------------------------")
some_list = [1, 3, 0, 4, 5] 
print(ex5.sort(some_list))
print(ex5.sort(some_list, ascending=False))