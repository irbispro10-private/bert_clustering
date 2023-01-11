import csv
import pandas as pd
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    print("The file is opening...")
    file = open(filename,mode)

    yield file
    print("The file is closing...")
    file.close()

with file_manager('input_data.csv', 'r+') as f:
    f.write("\n")
    for line in f.readlines():
        f.write(line.replace('\n', '"\n"'))