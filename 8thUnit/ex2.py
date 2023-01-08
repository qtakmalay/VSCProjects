import math, os
class Reader:
    def __init__(self, path: str):
        self.path = path
        if os.path.isfile(path):
            self.file = open(path, "rb")
        else: raise ValueError
    
    def close(self):
        self.file.close()
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, key):
        raise NotImplementedError

r = Reader("C:\\Users\\azatv\\VSCProjects\\8thUnit\\ex2_data.txt")
r.file.close()
# print(r[0])
# print(r[1])
# print(r[-1])
# try:
#     r["hi"]
# except TypeError as e:
#     print(f"{type(e).__name__}: {e}")
# try:
#     r[100]
# except IndexError as e:
#     print(f"{type(e).__name__}: {e}")