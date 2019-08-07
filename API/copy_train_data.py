import shutil
import os
import glob

base_dst = os.path.join('..', 'input', 'train')
# base_src = os.path.join('..', 'original', 'train')
base_src = os.path.join('..', 'input', 'original', 'train')
print(base_dst)
print(base_src)
# base_dst = "C:\\Users\\Manthan1412\\Documents\\MATLAB\\input\\train"
# base_src = "C:\/Users\/Manthan1412\/Downloads\/Compressed\/imgs\/train"

try:
    os.makedirs(base_dst)
except OSError:
    pass

Count = int(input("Enter count: "))
categories = 10

for i in range(0, categories):
    src = base_src + "/c" + str(i) + "/*.jpg"
    dst = base_dst + "/c" + str(i)
    try:
        os.makedirs(dst)
    except OSError:
        pass
    count = Count
    for file in glob.iglob(src):
        if count > 0:
            shutil.copy2(file, dst)
            count -= 1
