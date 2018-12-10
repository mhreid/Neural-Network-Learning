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
    layer = np.random.rand(input, output) - .5
    return layer

def make_layers(input_size, output_size, num):
    #tested and works
    #should have a minimum of two layers for num, which is equivalent to 1 hidden layer
    hidden_size = 100
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

def sigmoid_all(x, func):
    #tested and works
    sig = np.array([[func(i) for i in j] for j in x])
    return sig


def forward(layers, data):
    #tested and is giving same answer regardless of input unless weights are super super small
    a = [data]
    z = []
    for layer in layers:
        z.append(np.dot(a[-1], layer))
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

def gradients(nums, a, z, layers):
    #a 0,1,2 z 1,2, l 1,2
    #their activation numbers match mine
    #their weights also match mine
    #z should always be minus 1
    #layer dimensions in order: 784 x 20, 20 x 10
    #but we calculate the derivatives backwards
    #in progress, need to work on transposes being correct

    d_layers = []
    d_cost = [np.multiply(-d_loss_sum_all(nums, a[-1]), sigmoid_all(z[-1], sigmoid_derivative))]
    d_layers.append(np.dot(a[-2].T,d_cost[-1]))
    for i in range(len(layers) - 2, -1, -1):
        d_cost.append(np.multiply(np.dot(d_cost[-1], layers[i + 1].T), sigmoid_all(z[i], sigmoid_derivative)))


        d_layers.append(np.dot(a[i].T, d_cost[-1]))
    d_layers.reverse()
    return d_layers

def back_prop(layers, data, nums, step, runs):
    for j in range(runs):
        a,z = forward(layers, data)
        print("backpropagation " + str(j + 1) + " out of " + str(runs))
        d_layers = gradients(nums, a, z, layers)
        for i in range(len(layers)):
            print(layers[i])
            layers[i] -= np.asarray(d_layers[i]) * step
            print(layers[i])

    a,z = forward(layers, data)
    return layers, a[-1]


if __name__ == '__main__':
    nums, data = make_traindata()
    nums = nums[:2]
    data = data[:2]
    layers = make_layers(data.shape[1], 10, 2)
    layers, output = back_prop(layers, data, nums, .1, 100)
    print(output[1])
    print(nums[1])
    print(output[0])
    print(nums[0])
