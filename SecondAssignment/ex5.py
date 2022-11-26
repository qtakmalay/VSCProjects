import re

def find_min(list_digs):
    a = list_digs[0]
    for i in list_digs:
        if(a > i):
            a = i
    return a

def find_max(list_digs):
    a = list_digs[0]
    for i in list_digs:
        if(a < i):
            a = i
    return a


print("Welcome to Data Statistics!")
exit_act = False
pos_acts = ('a', 'v', 'x')
num_list = list()
num_check = re.compile(r'-?\d+\.?\d*')
while(not exit_act):
    action = input("""Available actions:
    a - Add numbers
    v - View statistics
    x - Exit the program
    Enter action: """)
    if(action in pos_acts):
        while(action == pos_acts[0]):

            numbers = input("Enter number or 'x' when you are done: ")
            if(numbers in pos_acts[2]): break                
            else:    
                if(re.match(num_check,numbers)):
                    num_list.append(int(numbers))
                if(numbers not in pos_acts[2] and not re.match(num_check,numbers)):
                    print("Invalid input '%s'. Try again!" %(numbers))
        if(action == pos_acts[1]):
            if(len(num_list) == 0):
                print("No numbers have been added yet!")
            else:
                print("""Count: %d
    Sum: %d
    Avg: %f
    Min: %d
    Max: %d""" %(int(len(num_list)), sum(num_list), sum(num_list)/len(num_list), find_min(num_list), find_max(num_list)))
                
        
        if(action == pos_acts[2]):
            print("Bye!")
            exit_act = True
            

    else: print("Invalid action '%s'. Try again!" %(action))