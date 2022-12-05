import os, re
def read_numbers(path: str) -> list:
    ab_path = os.path.dirname(__file__)
    fl_reader = open(os.path.join(ab_path, path), "r")
    fl_numsl = list(str(fl_reader.read()).split())
    int_check = re.compile(r'^\-?[1-9][0-9]*$')
    str_check = re.compile(r'^[a-zA-Z]+$')
    for val in fl_numsl:
        if(re.match(str_check, val)):
            fl_numsl.remove(val)
        
        # else:
        #     if(re.match(int_check,val)):
        #         fl_numsl.append(int(val))
            
    print(fl_numsl)   
    return fl_numsl

read_numbers("ex1_data.txt")