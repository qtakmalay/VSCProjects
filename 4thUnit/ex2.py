def clip(*values, min_=0, max_=1) -> list:
    cl_list = list()
    if len(values) == 0:
        return cl_list
    for val in values:
        if val < min_:
            cl_list.append(min_)
        elif val > max_:
            cl_list.append(max_)
        else:
            cl_list.append(val)
    return cl_list


