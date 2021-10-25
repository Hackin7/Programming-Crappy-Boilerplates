# https://www.kite.com/python/answers/how-to-traverse-a-directory-in-python
import os
path = os.walk(".")

for root, directories, files in path:
    print("###", root)
    for directory in directories:
        print(directory)
    for file in files:
        print(file)
