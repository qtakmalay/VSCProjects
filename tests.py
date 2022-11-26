import os

directory = 'files'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)

for filename in os.scandir(directory):
    if filename.is_file():
        print(filename.path)