import sys
import numpy as np
import loadData as ld

def estimators(nclass, dimension, labels, images):
    '''
    :param nclass: The number of classes(int), For this question should be 10.
    :param dimension: 16*16 = 256(int) per image.
    :param labels: It is a np.narray, which contains the real class of corresponding image.
    :param images: It is a 2D np.narray. Each row of it is also a np.narray(256 dimension,float), which is an image.
                   labels[i] stores the class of the images[i].
    :return:
    u: A dictionary whose key is the class label k(int), and corresponding value is a list, which
       contains two element [n,m], n indicates the number of images which are labeled with
       class k, and the m is mean of these images(np.narray(256 dimension,float)).
    delta_2: A np.narray(256 dimension,float)
    p: A dictionary whose key is the class label k(int), and corresponding value is the prior probability of class k(float).
    '''
    u = {}
    for i in range(1, nclass + 1):
        u[i] = [0, np.zeros(dimension)]
    for index, item in enumerate(labels):
        u[item] = [u.get(item)[0] + 1, u.get(item)[1] + images[index]]
    for item in u:
        u[item][1] = u.get(item)[1] / u.get(item)[0]
    p = {}
    for i in range(1, nclass + 1):
        p[i] = u.get(i)[0] / len(labels)
    delta_2 = np.zeros(dimension)
    for index, item in enumerate(images):
        delta_2 += np.power(item - u.get(labels[index])[1], 2)
    delta_2 = delta_2 / len(images)
    return u, delta_2, p


# create parameter file
def usps_d_param(nclass, dimension, u, delta_2, p):
    '''
    :param delta_2: It should be a 1D np.narray with 256 dimension or with 256*256 demension
    '''
    t = ''
    f = open("usps_d.param", "w")
    if len(delta_2) == dimension:
        t = 'd'
        f.write(t + '\n')
    elif len(delta_2) == dimension * dimension:
        t = 'f'
        f.write(t + '\n')
    else:
        print("The parameters are in wrong form.\n")
        quit()
    f.write(str(nclass)+'\n')
    f.write(str(dimension)+'\n')
    for key in u:
        f.write(str(key)+'\n')
        f.write(str(p.get(key))+'\n')
        for i in u[key][1]:
            f.write(str(i) + ' ')
        f.write('\n')
        if t == 'd':
            for i in delta_2:
                f.write(str(i) + ' ')
            f.write('\n')
        elif t == 'f':
            for i in delta_2:
                for j in range(dimension):
                    f.write(str(i) + ' ')
                f.write('\n')
            f.write('\n')
    f.close()


def main():
    '''
    When this file is called, one training data file path should be given as argument.
    '''
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    if len(sys.argv) == 1:
        print("Please give the training data file.")
    else:
        file_path = sys.argv[1]
        nclass, dimension, labels, images = ld.loadData(file_path)
        u, delta_2, p = estimators(nclass, dimension, labels, images)
        usps_d_param(nclass, dimension, u, delta_2, p)

main()