def binary_search(elements: list, x) -> bool:
    half_list = elements.copy()
    mid_val = half_list[int(len(half_list)/2)]

    if not isinstance(x, str):
        if x == mid_val:
            return True
        if len(half_list) == 1 and not x == mid_val:
            return False  
        if x < mid_val:
            del half_list[int(len(half_list)/2) : int(len(half_list))]
            mid_val = half_list[int(len(half_list)/2)]
            return binary_search(half_list, x)
        else:
            del half_list[ : int(len(half_list)/2)]
            mid_val = half_list[int(len(half_list)/2)]
            return binary_search(half_list, x)
    else:
        return False
        


my_sorted_list = [1, 2, 5, 7, 8, 10, 20, 30, 41, 100]
print(binary_search(my_sorted_list, 1)) #-> True
print(binary_search(my_sorted_list, 20)) #-> True
print(binary_search(my_sorted_list, 21)) #-> False
print(binary_search(my_sorted_list, "hello")) #-> False