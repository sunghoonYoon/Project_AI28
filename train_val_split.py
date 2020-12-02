import glob
import os
import random
import shutil
import pdb

files= glob.glob("F:\\AI_dataset3\\*.png")

# print(files)
# print(files)

val_dir = "F:\\AI_dataset3\\val"
train_dir = "F:\\AI_dataset3\\train"

random.shuffle(files)

# print(files)

total_file = len(files)

print(total_file)

train_files = files[:int(len(files)*0.7)]
val_files = files[int(len(files)*0.7):]

# pdb.set_trace()

# print(train_files)


for train_file in train_files:
    shutil.move(train_file,train_dir)

for val_files in val_files:
    shutil.move(val_files,val_dir)



