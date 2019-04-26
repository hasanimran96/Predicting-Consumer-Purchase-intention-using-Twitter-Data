import os

path = "uploadeddata\\"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    files.append(f)

for f in files:
    for x in f:
        print(x)