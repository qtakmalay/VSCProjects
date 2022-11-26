import re
# Output:
# ['file0.txt', 'file1.txt', 'file2.txt', 'file3.txt', 'file5.txt', 'file7.txt']
fnames = ["file7.txt", "file1.png", "file3.txt", "file2.txt",
"file7.txt", "file1.txt", "file3.txt", "file4.png",
"file4.png", "file5.txt", "file0.txt", "file7.dat"]
new_fnames = list()
for fname in fnames:
    if(".txt" in fname and not fname in new_fnames):
        new_fnames.append(fname)
new_fnames.sort(key=lambda f: int(re.sub('\D', '', f)))
print(new_fnames)