import argparse

parser = argparse.ArgumentParser()
parser.add_argument("my_arg", type=int)
args = parser.parse_args()
print(args.my_arg)
