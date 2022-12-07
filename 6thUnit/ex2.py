import os

def write_dict(d: dict, path: str, encoding: str = "utf-8"):
    ab_path = os.path.dirname(__file__)
    fl_reader = open(os.path.join(ab_path, path), "w")
    for key, value in d.items():  # Instead of: key_value tuple
        fl_reader.write(str(key)+" "+str(value)+"\n")
    


def read_dict(path: str, encoding: str = "utf-8") -> dict:
    ab_path = os.path.dirname(__file__)
    fl_reader = open(os.path.join(ab_path, path), "r")
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

file_ex2 = "ex2_data.txt"
dict_ex2 = {"stringkey": 55, "foo" : 43}
write_dict(dict_ex2, "ex2_data.txt")
print(read_dict(file_ex2))
new_dict = read_dict(file_ex2)
if(dict_ex2 == new_dict):
    print("True") # This must evaluate to True
