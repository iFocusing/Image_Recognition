import re
import numpy as np

def loadData(file_path):
    with open(file_path) as fl:
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        lables = []
        images = []
        lines = fl.readlines()
        single_image = []
        for line in lines:
            li = re.findall(r"\d+\.?\d*", line)
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

def getParameters(file_path):
    u = {}
    p = {}
    with open(file_path) as fl:
        d = fl.readline()
        print(d)
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        print(d,nclass,dimension)
        for i in range(nclass):
            c = np.int(fl.readline())
            p[c] = np.float(fl.readline())
            u[c] = [0, np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)]
            delta_2 = np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)
    return d, u, delta_2, p