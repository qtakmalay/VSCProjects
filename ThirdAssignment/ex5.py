import re


def parse_list(in_string): # input string with values and spaces
    var = ''
    new_list = list()
    for i,val in enumerate(in_string):
        if not val.isspace():
            var += val
        if val.isspace() or len(in_string)-1 == i:
            new_list.append(int(var))
            var = ''
    return new_list


print("Welcome to Powerlifting Data Collector!")
exit_act = False
pos_acts = ('a', 'r', 'u', 'v', 'x')
lift_types = ('squat', 'bench press', 'deadlift')
dict_names = dict()
num_check = re.compile(r'-?\d+\.?\d*')
lift_dflt = {"squat": [],
"bench_press": [],
"deadlift": []
}
while(not exit_act):
    action = input("""Available actions:
a - Add lifter
r - Remove lifter
u - Update lifter
v - View lifters
x - Exit the program
Enter action: """)
    if(action in pos_acts):
        curr_name = '' 
        # choosed a - Add lifter
        while(action == pos_acts[0]):

            name = input("Enter new lifter name: ")
            if(name in pos_acts[4]): break                
            else:
                if( name in dict_names):
                    print("Lifter '%s' already exists!" %name)    
                if not(name in dict_names):
                    dict_names[name] = lift_dflt
                break
        #choosed r - Remove lifter
        while(action == pos_acts[1]):
            name_r = input("Enter lifter name to remove:")
            if(name_r in pos_acts[4]): break                
            else: 
                if(not name_r in dict_names):
                    print("Lifter '%s' does not exist!" %name_r)   
                if(name_r in dict_names):
                   del dict_names[name_r]
                   if(name_r in curr_name):
                    curr_name = ''
                   break
                
        #choosed u - Update lifter
        while(action == pos_acts[2]):

            name_u = input("Enter lifter name to update:")
            if(name_u in pos_acts[4]): break                
            else:
                if(name_u in dict_names):
                   curr_name = name_u 
                   lift_type = input("Enter lift (one of 'squat', 'bench press', 'deadlift'):")
                   if(lift_type in lift_types):
                        weights = input('Enter weight(s):')
                        if(re.match(num_check, weights)):
                            dict_names[curr_name][lift_type.replace(" ", "_")].append(weights.split())
                            break
                if(not name_u in dict_names):
                    print("Lifter '%s' does not exist!" %name_u)
        if(action == pos_acts[3]):
            if(len(dict_names) > 0):
                for val in dict_names: 
                    print('------------------------------')
                    print("Name: %s\nsquat: %s\nbench press: %s\ndeadlift: %s" %(val, str(dict_names[val]['squat']).replace("'","").replace("[","",1).replace("]","",1), str(dict_names[val]['bench_press']).replace("'","").replace("[","",1).replace("]","",1), str(dict_names[val]['deadlift']).replace("'","").replace("[","",1).replace("]","",1)))
        if(action == pos_acts[4]):
            print("Bye!")
            exit_act = True
            

    else: print("Invalid action '%s'. Try again!" %(action))