# import re
# stir = r"""12 w 21  d23g780nb deed e2 21.87
# 43 91 - . 222 mftg 21 bx .1 3 g d e 6 de ddd32 3412
# 0.3 0 0. 0 0 1

# 70

# n 12 1    9    m1 1m 445
# x 100"""
# print(repr(stir))
import os, re
def read_numbers(path: str) -> list:
    ab_path = os.path.dirname(__file__)
    fl_reader = open(os.path.join(ab_path, path), "r")
    fl_numsl = list(str(fl_reader.read()).replace("  ", " ").split())
    str_l = re.compile(r"[a-zA-Z]+")
    real_index, real_length = 0,
    while :
        print(fl_numsl[i])
        if  not (re.match(str_l,fl_numsl[real_index]) == None):
            del fl_numsl[real_index]
            real_index -= 1
        real_index += 1
            
        # if val.isupper() or val.islower():
        #     fl_numsl.remove())
        # if not (val.find(".") == -1 ):    
        #     fl_numsl[fl_numsl.index(val)] = float(val)
        # else:
        #     fl_numsl[fl_numsl.index(val)] = int(val)
    print(fl_numsl)   
    return fl_numsl

read_numbers("ex1_data.txt")

# regex = r"[a-zA-Z]+"

# test_str = ("12 w 21  d23g780nb deed e2 21.87\n"
# 	"# 43 91 - . 222 mftg 21 bx .1 3 g d e 6 de ddd32 3412\n"
# 	"# 0.3 0 0. 0 0 1\n\n"
# 	"# 70\n\n"
# 	"# n 12 1    9    m1 1m 445\n"
# 	"# x 100")

# matches = re.finditer(regex, test_str, re.MULTILINE)

# for matchNum, match in enumerate(matches, start=1):
    
#     print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
    
#     for groupNum in range(0, len(match.groups())):
#         groupNum = groupNum + 1
        
#         print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))