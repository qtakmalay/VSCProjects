def flatten(nested : list):
    if not nested:
        return nested
    if isinstance(nested[0], list):
        return flatten(*nested[:1]) + flatten(nested[1:])
    return nested[:1] + flatten(nested[1:])
             

    
print(flatten([1, 2, [4, [8, 9, [11, 12], 10], 5], 3, [6, 7]])) #= [1, 2, 4, 8, 9, 11, 12, 10, 5, 3, 6, 7]
print(flatten([[]])) #= []
print(flatten([[], [], [1], [], [1, [], [4, 5, [[[6]]]], 2, 3]])) #= [1, 1, 4, 5, 6, 2, 3]


