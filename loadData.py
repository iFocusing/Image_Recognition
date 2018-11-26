import re
import numpy as np

def loadData(file_path):
    '''
    :param file_path:
    :return:
    nclass: The number of classes(int), For this question should be 10.
    dimension: 16*16 = 256(int) per image.
    labels: It is a np.narray, which contains the real class of corresponding image.
    images: It is a 2D np.narray. Each row of it is also a np.narray(256 dimension,float), which is an image.
            labels[i] stores the class of the images[i].
    '''
    with open(file_path) as fl:
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        labels = []
        images = []
        lines = fl.readlines()
        single_image = []
        for line in lines:
            li = re.findall(r"\d+\.?\d*", line)
            if len(li) == 1:
                labels.append(int(li[0]))
            elif len(single_image) != dimension:
                single_image += li
            else:
                images.append(single_image)
                single_image = li
        images.append(single_image)
    fl.close()
    labels = np.array(labels)
    images = np.array(images).astype(int)
    return nclass, dimension, labels, images


def getParameters_usps_pd(file_path):
    u = {}
    p = {}
    with open(file_path) as fl:
        t = fl.readline().strip('\n')
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        for i in range(1,nclass+1):
            c = int(fl.readline())
            p[c] = np.float(fl.readline())
            u[c] = [0, np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)]
            delta_2 = np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)
        #print('usps_pd', delta_2)
    return t, u, delta_2, p


def getParameters_usps_pf(file_path):
    u = {}
    p = {}
    with open(file_path) as fl:
        t = fl.readline().strip('\n')
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        delta_2 = np.zeros(dimension * dimension).reshape(dimension, dimension)
        for i in range(1, nclass + 1):
            c = int(fl.readline())
            p[c] = float(fl.readline())
            u[c] = [0, np.array(list(map(float, fl.readline().strip().split(' '))))]
            for j in range(0, dimension):
                delta_2[j] = np.array(list(map(float, fl.readline().strip().split(' '))))
            # print('usps_pf', delta_2)
    return t, u, delta_2, p


def getParameters_usps_cd(file_path):
    u = {}
    p = {}
    with open(file_path) as fl:
        t = fl.readline().strip('\n')
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        delta_2 = {}
        for i in range(1, nclass + 1):
            c = int(fl.readline())
            p[c] = np.float(fl.readline())
            u[c] = [0, np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)]
            delta_2[c] = np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)
        # print('usps_cd', delta_2)
    return t, u, delta_2, p


def getParameters_usps_cf(file_path):
    u = {}
    p = {}
    with open(file_path) as fl:
        t = fl.readline().strip('\n')
        nclass = int(fl.readline())
        dimension = int(fl.readline())
        delta_2 = {}
        delta = np.zeros(dimension * dimension).reshape(dimension, dimension)
        for i in range(1, nclass + 1):
            c = np.int(fl.readline().strip('\n'))
            p[c] = np.float(fl.readline().strip('\n'))

            u[c] = [0, np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)]
            for j in range(0, dimension):
                delta[j] = np.array(re.findall(r"\d+\.?\d*", fl.readline())).astype(float)

            delta_2[c] = delta
        # print(delta_2)
        # print('usps_cf', delta_2)
        # TODO:ValueError: could not broadcast input array from shape (260) into shape (256)
        return t, u, delta_2, p