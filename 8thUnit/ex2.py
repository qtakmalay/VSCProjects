import math
class Reader:
    def __init__(self, path: str):
        raise NotImplementedError
    def close(self):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, key):
        raise NotImplementedError

r = Reader("ex2_data.txt")
print(r[0])
print(r[1])
print(r[-1])
try:
    r["hi"]
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
try:
    r[100]
except IndexError as e:
    print(f"{type(e).__name__}: {e}")