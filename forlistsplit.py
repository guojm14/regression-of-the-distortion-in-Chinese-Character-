datalist=open('../textgenerator-/list.txt').readlines()
import random
fop1=open('train.txt','w')
fop2=open('test.txt','w')
for item in datalist:
    if random.randint(0,9)==9:
        fop2.write(item)
    else:
        fop1.write(item)
