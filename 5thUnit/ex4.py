def flatten(nested: list) -> list:
    fl_list = list()
    curr_i = 0
    if not isinstance(nested, list):
        return nested
    for i, val in enumerate(nested):


print(flatten([1, 2, [4, [8, 9, [11, 12], 10], 5], 3, [6, 7]])) #= [1, 2, 4, 8, 9, 11, 12, 10, 5, 3, 6, 7]
print(flatten([[]])) #= []
print(flatten([[], [], [1], [], [1, [], [4, 5, [[[6]]]], 2, 3]])) #= [1, 1, 4, 5, 6, 2, 3]