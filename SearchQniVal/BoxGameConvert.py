in_game = """G F D
J D B
J D b
i f C
g b a
H e C
g f b
i e D
H g d
g f E
h e c
g e a
E C A
g D B
G C B
g b A
I e D
f e A
I h A
i f d
I G b
e D c
I H G
j H e
g E a
G d C
j h A
c B A
J f b
E d A
I H E
j I e
j i B
h f D
I h d
G f c
I C A
J D c
I E c
G B a"""

convert_str = "("
for x in in_game:
    if(x.isupper()):
        convert_str += "!" + x.lower()
    if(x.islower()):
        convert_str += x
    if(x.isspace()):
        convert_str += "|"
    if(x == "\n"):
        convert_str += ") & ("  
print(convert_str.replace("\n","").replace("\r","").replace(" ","").replace("|)&(",")&("))
# Put your values from limbole 
# g = 0
# f = 1
# d = 0
# j = 1
# b = 1
# i = 0
# c = 1
# a = 1
# h = 0
# e = 0
arr = {"g0", "f1",
"d0",
"j1",
"b1",
"i0",
"c1",
"a1",
"h0",
"e0"}

results = ""
count = 0
for val in (in_game):
    if(count % 3 == 0 and count % 12 != 0):
        results += "    " 
    if(count % 12 == 0):
        results += "\n" 
        
    if(val.isupper()):
        count += 1
        for val_arr in (arr):
            if(val.lower() == str(val_arr[0])):
                if(int(val_arr[1]) == 0):
                    results += "T "
                else:
                    results += "F "
    if(val.islower()):
        count += 1
        for val_arr in (arr):
            if(val.lower() == str(val_arr[0])):
                if(int(val_arr[1]) == 0):
                    results += "F "
                else:
                    results += "T "
        
print(results)