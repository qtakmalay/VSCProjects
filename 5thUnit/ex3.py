def binary_search(elements: list, x) -> bool:
    half_list = elements.copy()
    mid_val = half_list[int(len(half_list)/2)]
    try:
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
    except TypeError:
        return False