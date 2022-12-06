import os, re
def read_numbers(path: str) -> list:
    ab_path = os.path.dirname(__file__)
    fl_reader = open(os.path.join(ab_path, path), "r")
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
    print(nums_list)

read_numbers("ex1_data.txt")