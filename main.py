import pandas as pd
import numpy as np
import random
import math

training = pd.read_csv('mnist_train.csv')
#print(training.head())
#This confirms that the data is there


def make_traindata():
    nums = training.iloc[:,0]
    nums = nums.as_matrix()
    data = training.iloc[:,1:]
    data = data.as_matrix()
    return nums,data

def make_rand_layer(input, output):
    """layer = []
    for i in range(input):
        row = []
        for j in range(output):
            row.append(random.random())
        layer.append(row)"""
    layer = np.random.rand(input, output) * .01

    return layer

def make_layers(input_size, output_size, num):
    hidden_size = 20
    layers = []
    if(num <= 1):
        layers.append(make_rand_layer(input_size, output_size))
    else:
        layers.append(make_rand_layer(input_size, hidden_size))
        for i in range(num - 2):
            layers.append(make_rand_layer(hidden_size, hidden_size))
        layers.append(make_rand_layer(hidden_size, output_size))
    return layers

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_all(x):
    sig = np.array([[sigmoid(i) for i in j] for j in x])
    return sig


def forward(layers, data):
    a = []
    z = []
    a.append(data.T)
    for layer in layers:
        print("layer")
        z.append(np.dot(a[-1], layer))
        a.append(sigmoid_all(z[-1]))
    return a[1:], z

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def w_sum(num):
    sum = 0
    for i in range(len(num[1])):
        print("layer " + str(i))
        if(num[0] == i - 1):
            sum += (1 - num[1][i]) ** 2
        else:
            sum += num[1][i] ** 2
    return sum

def w_sum_all(data):
    sum = 0
    for d in data:
        sum += w_sum(d)
    print(sum)
    sum /= len(data)
    print(len(data))
    return sum


if __name__ == '__main__':
    nums, data = make_traindata()
    print(data.shape[0])
    layers = make_layers(data.shape[0], 10, 3)
    print(len(layers))
    a,z = forward(layers, data)
    print(a[-1][0])
