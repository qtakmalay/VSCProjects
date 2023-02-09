import re
import argparse

args = argparse.ArgumentParser()
args.add_argument("-i", "--input_file", type=str, required=True)
args.add_argument("-p", "--patterns", nargs="+", type=list, required=True)
args.add_argument("-s", "--separator", type=str, required=False)
args.add_argument("-e", "--encoding", type=str, required=False)

parsed_args = args.parse_args()
try:
    with open(parsed_args.input_file, "r") as file:
        
        in_str = file.read()
        for index, i in enumerate(parsed_args.patterns):
            matches = list()
            pattern = re.compile(i)
            matches.append(re.findall(i, in_str))
            with open(parsed_args.input_file+"_"+index+".txt", "w+") as out_file:
                out_file.write("regex:"+i+"\n"+matches)


            
except ValueError:
    print(f"Filepath {parsed_args.input_file} is not specified")
