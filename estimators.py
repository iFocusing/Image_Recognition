import sys
import numpy as np
import loadData as ld


def estimators(nclass, dimension, labels, images):
    """
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
    """
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
    # delta_2 = np.zeros(dimension)
    # for index, item in enumerate(images):
    #     delta_2 += np.power(item - u.get(labels[index])[1], 2)
    # delta_2 = delta_2 / len(images)
    cf_delta, cd_delta, pf_delta, pd_delta = estimate_covariance(nclass, dimension, labels, images, u)
    return u, cf_delta, cd_delta, pf_delta, pd_delta, p


def estimate_covariance(nclass, dimension, labels, images, u):
    cf_delta = {}
    cd_delta = {}
    pf_delta = np.zeros(dimension * dimension).reshape(dimension, dimension)
    pd_delta = np.zeros(dimension)

    for index, item in enumerate(labels):
        # calculate cf_delta
        if cf_delta.get(item) is None:
            cf_delta[item] = (images[index] - u.get(item)[1]).reshape(dimension, 1) * (images[index] - u.get(item)[1])
        else:
            cf_delta[item] = cf_delta.get(item) + (images[index] - u.get(item)[1]).reshape(dimension, 1) \
                             * (images[index] - u.get(item)[1])
        # calculate cd_delta
        if cd_delta.get(item) is None:
            cd_delta[item] = np.power(images[index] - u.get(item)[1],2)
        else:
            cd_delta[item] = cd_delta.get(item) + np.power(images[index] - u.get(item)[1], 2)

    # calculate cf_delta
    for item in cf_delta:
        pf_delta += cf_delta.get(item)
        cf_delta[item] = cf_delta.get(item)/u.get(item)[0]
    pf_delta = pf_delta/len(labels)
    print('cf_delta:', cf_delta)
    print('pf_delta:', pf_delta)

    # calculate cd_delta
    for item in cd_delta:
        pd_delta += cd_delta.get(item)
        cd_delta[item] = cd_delta.get(item)/u.get(item)[0]
    pd_delta = pd_delta/len(labels)
    print('cd_delta:', cd_delta)
    print('pd_delta:', pd_delta)

    return cf_delta, cd_delta, pf_delta, pd_delta


# create parameter file
def usps_d_param(nclass, dimension, u, delta_2, t, p):
    # TODO: According t(type) write files;
    """
    :param delta_2: It should be a 1D np.narray with 256 dimension or with 256*256 demension
    """
    if t == 'usps_cf':
        f = open(t + ".param", "w")
        f.write(t + '\n')
        f.write(str(nclass) + '\n')
        f.write(str(dimension) + '\n')
        for key in u:
            f.write(str(key) + '\n')
            f.write(str(p.get(key)) + '\n')
            for i in u[key][1]:
                f.write(str(i) + ' ')
            f.write('\n')
            for row in delta_2.get(key):
                for i in row:
                    f.write(str(i) + ' ')
                f.write('\n')
            f.write('\n')
        f.close()

    elif t == 'usps_cd':
        f = open(t + ".param", "w")
        f.write(t + '\n')
        f.write(str(nclass) + '\n')
        f.write(str(dimension) + '\n')
        for key in u:
            f.write(str(key) + '\n')
            f.write(str(p.get(key)) + '\n')
            for i in u[key][1]:
                f.write(str(i) + ' ')
            f.write('\n')
            for i in delta_2.get(key):
                f.write(str(i) + ' ')
            f.write('\n')
        f.close()

    elif t == 'usps_pf':
        f = open(t + ".param", "w")
        f.write(t + '\n')
        f.write(str(nclass) + '\n')
        f.write(str(dimension) + '\n')
        for key in u:
            f.write(str(key)+'\n')
            f.write(str(p.get(key))+'\n')
            for i in u[key][1]:
                f.write(str(i) + ' ')
            f.write('\n')
            for row in delta_2:
                for i in row:
                    f.write(str(i) + ' ')
                f.write('\n')
            f.write('\n')
        f.close()

    elif t == 'usps_pd':
        f = open(t + ".param", "w")
        f.write(t + '\n')
        f.write(str(nclass)+'\n')
        f.write(str(dimension)+'\n')
        for key in u:
            f.write(str(key)+'\n')
            f.write(str(p.get(key))+'\n')
            for i in u[key][1]:
                f.write(str(i) + ' ')
            f.write('\n')
            for i in delta_2:
                f.write(str(i) + ' ')
            f.write('\n')
        f.close()

    else:
        print("The parameters are in wrong form.\n")
        quit()

def main():
    """
    When this file is called, one training data file path should be given as argument.
    """
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    if len(sys.argv) == 1:
        print("Please give the training data file.")
    else:
        file_path = sys.argv[1]
        nclass, dimension, labels, images = ld.loadData(file_path)
        u, cf_delta, cd_delta, pf_delta, pd_delta, p = estimators(nclass, dimension, labels, images)
        usps_d_param(nclass, dimension, u, cf_delta, 'usps_cf', p)
        usps_d_param(nclass, dimension, u, cd_delta, 'usps_cd', p)
        usps_d_param(nclass, dimension, u, pf_delta, 'usps_pf', p)
        usps_d_param(nclass, dimension, u, pd_delta, 'usps_pd', p)

main()