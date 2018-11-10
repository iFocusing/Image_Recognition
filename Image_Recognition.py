import numpy as np
import re
def loadData():
    with open('usps.train') as fl:
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        lables = []
        images = []
        lines = fl.readlines()
        single_image = []
        for line in lines:
            li = re.findall(r"\d+\.?\d*",line)
            if len(li) == 1:
                lables.append(int(li[0]))
            elif len(single_image) != dimension:
                single_image += li
            else:
                images.append(single_image)
                single_image = li
        images.append(single_image)
    #print(len(lables))
    #print(lables[7289], images[7289])
    #print(len(images))
    fl.close()
    return nclass, dimension, lables, images


nclass, dimension, lables, images= loadData()
#print(nclass, dimension, lables[0], images[0])
lables = np.array(lables)
images = np.array(images).astype(int)
#print(type(images[1]))
u = {}
for i in range(1,nclass+1):  #using key:value pairs to store the classname:[N_k,sum[]] 这里写死了classname必须是int 1，2，3，4, ... ,10, not good
    u[i] = [0,np.zeros(dimension)]
for index,item in enumerate(lables):
    #print(u.get(item)[0],u.get(item)[1])
    u[item] = [u.get(item)[0] + 1, u.get(item)[1] + images[index]]
    #print(index,item)
#print(u)
for item in u:
    u[item][1] = u.get(item)[1]/u.get(item)[0]
#print(u)

p={}
for i in range(1,nclass+1):  #using key:value pairs to store the classname:[N_k,sum[]] 这里写死了classname必须是1，2，3，4, ... ,10, not good
    p[i] = u.get(i)[0] / len(lables)

delta_2 = np.zeros(dimension)
for index,item in enumerate(images):
    delta_2 += np.power(item - u.get(lables[index])[1],2)
delta_2 = delta_2/len(images)
#print(len(delta_2))

# create parameter file
def usps_d_param(nclass, dimension, u, delta_2, p):
    f = open("usps_d.param", "w")
    f.write('d\n')
    f.write(str(nclass)+'\n')
    f.write(str(dimension)+'\n')
    for key in u:
        f.write(str(key)+'\n')
        f.write(str(p.get(key))+'\n')
        f.write(str(u[key][1])+'\n')
        f.write(str(delta_2)+'\n')
    f.close()

usps_d_param(nclass, dimension, u, delta_2, p)

