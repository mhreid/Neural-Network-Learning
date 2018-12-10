import pandas as pd
import numpy as np
import random
import math

training = pd.read_csv('mnist_train.csv')
#print(training.head())
#This confirms that the data is there


def make_traindata():
    #tested and works
    nums = training.iloc[:,0]
    nums = nums.as_matrix()
    data = training.iloc[:,1:]
    data = data.as_matrix()
    return nums,data

def make_rand_layer(input, output):
    #tested and works
    layer = (np.random.rand(input, output) - .5)
    return layer

def make_layers(input_size, output_size, num):
    hidden_size = 200
    layers = []
    biases = []
    if(num <= 1):
        layers.append(make_rand_layer(input_size, output_size))
    else:
        layers.append(make_rand_layer(input_size, hidden_size))
        for i in range(num - 2):
            layers.append(make_rand_layer(hidden_size, hidden_size))
        layers.append(make_rand_layer(hidden_size, output_size))
    for i in range(num - 1):
        biases.append(np.random.rand(1, hidden_size) / 2 + .2)
    biases.append(np.random.rand(1, output_size) / 2 + .2)
    return layers, biases

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_all(x, func):
    #tested and works
    sig = np.array([[func(i) for i in j] for j in x])
    return sig


def forward(layers, biases, data):
    a = [data]
    z = []
    for i in range(len(layers)):
        z.append(np.dot(a[-1], layers[i]) + biases[i])
        a.append(sigmoid_all(z[-1], sigmoid))
    return a, z

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def d_loss_sum(num, y_hat):
    #tested and works
    y = [0 for i in range(len(y_hat))]
    y[num] = 1
    return 2 * (y - y_hat)

def d_loss_sum_all(nums, y_hats):
    #tested and works
    sum = []
    for i in range(nums.shape[0]):
        sum.append(d_loss_sum(nums[i], y_hats[:][i]))
    return np.matrix(sum)

def gradients(nums, a, z, layers, biases):

    d_layers = []
    d_biases = []
    d_cost = [np.multiply(-d_loss_sum_all(nums, a[-1]), sigmoid_all(z[-1], sigmoid_derivative))]
    print("Current Cost: " + str(np.sum(d_cost)))
    d_layers.append(np.dot(a[-2].T,d_cost[-1]))
    d_biases.append(np.sum(d_cost[-1]))
    for i in range(len(layers) - 2, -1, -1):
        d_cost.append(np.multiply(np.dot(d_cost[-1], layers[i + 1].T), sigmoid_all(z[i], sigmoid_derivative)))
        d_biases.append(np.sum(d_cost[-1]))
        d_layers.append(np.dot(a[i].T, d_cost[-1]))
    d_layers.reverse()
    d_biases.reverse()
    return d_layers, d_biases

def back_prop(layers, biases, data, nums, step, batch, repeats):
    for j in range(repeats):
        for q in range(int(len(nums) / batch)):
            temp_data = data[batch * q: batch * (q+1)]
            temp_nums = nums[batch * q: batch * (q+1)]
            a,z = forward(layers, biases, temp_data)
            output = a[-1]
            print("repeat {} of {}, batch {} of {}".format(j + 1, repeats, q + 1, int(len(nums) / batch)))
            d_layers, d_biases = gradients(temp_nums, a, z, layers, biases)
            for i in range(len(layers)):
                layers[i] -= np.asarray(d_layers[i]) * step / batch
                biases[i] -= d_biases[i] * step / batch

    a,z = forward(layers, biases, data)
    return layers, a[-1]


if __name__ == '__main__':
    #really good example

    nums, data = make_traindata()

    nums = nums[:10000]
    data = data[:10000]
    layers, biases = make_layers(data.shape[1], 10, 3)

    a, z = forward(layers, biases, data)
    print(a[-1][1])
    print(nums[1])


    layers, output = back_prop(layers, biases, data, nums, .1, 100,4)
    print(output[10])
    print(nums[10])
    print(output[8])
    print(nums[8])
    print(output[7])
    print(nums[7])
"""
    #Example of how it poorly handles complexity:

    nums, data = make_traindata()
    nums = nums[:1000]
    data = data[:1000]
    layers, biases = make_layers(data.shape[1], 10, 3)
    layers, output = back_prop(layers, biases, data, nums, .1, 20)
    print(output[1])
    print(nums[1])
    print(output[8])
    print(nums[8])
    print(output[7])
    print(nums[7])"""
