import math, os
class Reader:
    def __init__(self, path: str):
        self.path = path
        if os.path.isfile(path):
            self.fh = open(path, "rb")
        else: raise ValueError
    
    def close(self):
        self.fh.close()
    def __len__(self):
        return os.path.getsize(self.path)
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += self.__len__()
            if key<=self.__len__() and key>=0:
                self.fh.seek(key)
                return self.fh.read(1) 
            else:
                raise IndexError("Point index out of range")
        raise TypeError(f"Point indices must be integers, not {type(key).__name__}")

# r = Reader("ex2_data.txt")
# print(os.path.getsize(r.path))
# # print(r.__len__)
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