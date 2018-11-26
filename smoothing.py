import sys
import estimators as es
import loadData as ld
import numpy as np
import math
import math
import matplotlib.pyplot as plt

def smoothing(cf_delta, cd_delta, pf_delta, pd_delta):
    # smooth cf_delta and pf_delta
    lamda = [1, 0.5, 0.1, 0.01, math.pow(10, -3), math.pow(10, -4), math.pow(10, -5), math.pow(10, -6)]  # math.pow()
    print(len(cf_delta),len(cd_delta))
    smoothed_delta = {}
    for item in lamda:
        print(item)
        for key in cf_delta:
            cf_delta[key] = item * pf_delta + (1 - item) * cf_delta[key]
        smoothed_delta[item] = cf_delta
    return smoothed_delta

def classifer_smoothed_delta(nclass, dimension, u, smoothed_delta, p, real_labels, testing_images):
    error = 0
    error_rates = {}
    testing_labels = np.zeros(len(real_labels)).astype(int)  # labels that our classifier gives
    # print(smoothed_delta.get(1))
    for lamda in smoothed_delta:
        print('lamda:',lamda) # this item should be the lamda
        # print(smoothed_delta.get(lamda)) # this should be a dict b = {} / b.get(class) is the full covariance matrix for that class
        for index, x in enumerate(testing_images):
            # print('testing_images',index)
            r_x = {}  # using dictinary {(1:.),(2:.)...(k:.)}store the values of discriminate function for each class k
            for key, value in p.items():
                # print('p:',key)
                # d.iteritems: an iterator over the (key, value) items
                r_x[key] = value * math.exp(
                    -1 / 2 * np.matmul(np.matmul(x - u.get(key)[1], np.linalg.inv(smoothed_delta.get(lamda).get(key))), (x - u.get(key)[1])))
            # print(r_x)
            sorted_by_value = sorted(r_x.items(), key=lambda kv: kv[1], reverse=True)
            testing_labels[index] = sorted_by_value[0][0]
            if real_labels[index] != testing_labels[index]:
                error += 1
        error_rates[lamda] = error / len(real_labels)
    print(error_rates)
    return testing_labels, error_rates


def errors(error_rate):
    f = open("errors_of_smoothed_delta", "w")
    for key in error_rate:
        f.write(str(key) + ' ' + str(error_rate.get(key)))
        f.write('\n')
    f.close()


def plot_errors(error_rates):
    lamda = []
    error = []
    for key in errors:
        lamda.append(math.log(key, 10))
        error.append(errors.get(key))
    print(lamda, "\n", error)
    plt.plot(lamda, error, 'ro')
    plt.axis([-7, 1, 0, 1])
    plt.show()


def main():
    if len(sys.argv) == 1:
        print("Please give the training data file.")
    elif len(sys.argv) == 2:
        print("Please give the testing data file.")
    else:
        training = sys.argv[1]
        testing = sys.argv[2]
        nclass, dimension, labels, images = ld.loadData(training)
        u, cf_delta, cd_delta, pf_delta, pd_delta, p = es.estimators(nclass, dimension, labels, images)
        print('whats the problem')
        smoothed_delta = smoothing(cf_delta, cd_delta, pf_delta, pd_delta)
        nclass, dimension, real_labels, testing_images = ld.loadData(testing)
        testing_labels, error_rates = classifer_smoothed_delta(nclass, dimension, u, smoothed_delta, p, real_labels,
                                                              testing_images)
    errors(error_rates)
    plot_errors(error_rates)




main()