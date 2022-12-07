import os, re
def read_numbers(path: str) -> list:
    fl_reader = open(path, "r")
    fl_mess= list(str(fl_reader.read()).replace("  ", " ").split())
    nums_list = list()

    for val in fl_mess:
        try:
            nums_list.append(int(val))
        except:
            try:
                nums_list.append(float(val))
            except:
                continue
    return nums_list
