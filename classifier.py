import sys
import loadData as ld
import numpy as np
from numpy.linalg import inv
import math

def classifer(nclass, dimension, u, delta_2, p, real_labels, testing_images):
    error = 0
    con_matrix = {(i,j):0 for i in range(1, nclass+1) for j in range(1, nclass+1)}
    testing_labels = np.zeros(len(real_labels)).astype(int) #labels that our classifier gives
    delta_2 = np.diag(delta_2)
    delta_2_inv = inv(delta_2)

    for index, x in enumerate(testing_images):
        r_x = {}  # using dictinary {(1:.),(2:.)...(k:.)}store the values of discriminate function for each class k
        for key, value in p.items():
            # d.iteritems: an iterator over the (key, value) items
            r_x[key] = value * math.exp(-1/2 * np.matmul(np.matmul(x - u.get(key)[1], delta_2_inv), (x - u.get(key)[1])))
        sorted_by_value = sorted(r_x.items(), key=lambda kv: kv[1], reverse=True)
        testing_labels[index] = sorted_by_value[0][0]
        con_matrix[real_labels[index], testing_labels[index]] = con_matrix.get((real_labels[index],testing_labels[index])) + 1
        if real_labels[index] != testing_labels[index]:
            error += 1
    error_rate = error/len(real_labels)
    return testing_labels, error_rate, con_matrix


def usps_d_error(error_rate):
    f = open("usps_d.error", "w")
    f.write(str(error_rate) + '\n')
    f.close()

def usps_d_cm(con_matrix,nclass):
    f = open("usps_d_cm", "w")
    for i in range(1, nclass+1):
        for j in range(1, nclass+1):
            f.write(str(con_matrix.get((i,j))) + '\t')
        f.write('\n')
    f.close()

def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    if len(sys.argv) == 1:
        print("Please give the parameters and testing files.")
    elif len(sys.argv) == 2:
        print("Please give one more file.")
    else:
        d, u, delta_2, p = ld.getParameters(sys.argv[1])
        nclass, dimension, real_labels, testing_images = ld.loadData(sys.argv[2])
        testing_labels, error_rate, con_matrix= classifer(nclass, dimension, u, delta_2, p, real_labels, testing_images)
        usps_d_error(error_rate)
        usps_d_cm(con_matrix,nclass)

main()