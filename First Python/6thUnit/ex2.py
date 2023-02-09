import os

def write_dict(d: dict, path: str, encoding: str = "utf-8"):
    fl_reader = open(path, "w")
    for key, value in d.items():  # Instead of: key_value tuple
        fl_reader.write(str(key)+" "+str(value)+"\n")
    


def read_dict(path: str, encoding: str = "utf-8") -> dict:
    fl_reader = open(path, "r")
    dict_new = dict()
    for line in fl_reader:
        temp_list = str(line).split()
        try:
            dict_new.update({temp_list[0]:int(temp_list[1])})
        except:
            try:
                dict_new.update({temp_list[0]:float(temp_list[1])})
            except:
                dict_new.update({temp_list[0]:str(temp_list[1])})
            
    return dict_new

