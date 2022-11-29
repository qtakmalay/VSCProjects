my_sorted_list = [1, 2, 5, 7, 8, 10, 20, 30, 41, 100]
print(int(len(my_sorted_list)/2))
del my_sorted_list[int(len(my_sorted_list)/2) : int(len(my_sorted_list))]
print(my_sorted_list)