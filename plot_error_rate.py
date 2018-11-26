import math
import matplotlib.pyplot as plt

def plot():
    error_rates = {1: 0.11559541604384654, 0.5: 0.2311908320876931, 0.1: 0.34678624813153963, 0.01: 0.4623816641753862,
              0.0010000000000000002: 0.5779770802192327, 0.00010000000000000002: 0.6935724962630793,
              1.0000000000000003e-05: 0.8091679123069258, 1.0000000000000004e-06: 0.9247633283507724}
    lamda = []
    error = []
    for key in error_rates:
        lamda.append(math.log(key,10))
        error.append(error_rates.get(key))
    print(lamda,"\n",error)
    plt.plot(lamda, error, 'ro')
    plt.axis([-7, 1, 0, 1])
    plt.show()


plot()
