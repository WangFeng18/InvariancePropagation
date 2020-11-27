import os

with open('new.txt') as f:
    lines = f.readlines()

lines = map(lambda x:x.replace("\n",""), lines)


str = '& '.join(lines)
with open('new2.txt', 'w') as f:
    f.writelines([str])

