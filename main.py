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
    layer = np.random.rand(input, output) *.01
    return layer

def make_layers(input_size, output_size, num):
    #tested and works
    #should have a minimum of two layers for num, which is equivalent to 1 hidden layer
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

def sigmoid_all(x, func):
    #tested and works
    sig = np.array([[func(i) for i in j] for j in x])
    return sig


def forward(layers, data):
    #tested and works
    a = [data]
    z = []
    for layer in layers:
        print("layer")
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
    print("original layer")
    print(layers[-1].shape)
    d_layers = []
    d_cost = [np.multiply(-d_loss_sum_all(nums, a[-1]), sigmoid_all(z[-1], sigmoid_derivative))]
    print(d_cost[0][0].shape)
    d_layers.append(np.dot(a[-2].T,d_cost[-1]))

    for i in range(len(layers) - 2, -1, -1):
        print(i)
        #i = 0

        d_cost.append(np.multiply(np.dot(d_cost[-1], layers[i + 1].T), sigmoid_all(z[i], sigmoid_derivative)))


        d_layers.append(np.dot(a[i].T, d_cost[-1]))
    d_layers.reverse()
    return d_layers

def back_prop(layers, data, nums, step, runs):
    for i in range(runs):
        a,z = forward(layers, data)
        #yhat shape
        print("run " + str(i))
        gradient = gradients(nums, a, z, layers)
        layers -= np.multiply(gradient, step)
    return layers


if __name__ == '__main__':
    nums, data = make_traindata()
    #print(data.shape[0])
    print(nums[1])
    #print(data[1][200:250])
    layers = make_layers(data.shape[1], 10, 2)
    print(layers[0].shape)
    #keep num as two for 1 hidden layer
    #print(len(layers))
    a,z = forward(layers, data)

    d_layers = gradients(nums, a, z, layers)
    print("layers")
    for layer in layers:
        print(layer[0].shape)
    print("d layers")
    for layer in d_layers:
        print(layer[0].shape)

    layers = d_layers, 0.001

    a,z = forward(layers, data)
    print(a[-1].shape)




    #print(d_layers[-1])
    #print(layers[1])
    #print(len(layers))
    #layers = back_prop(layers, data, nums, .0001, 4)
    #print(layers[-1])

    #a,z = forward(layers, data)
    #print(a[-1][1])
    #print(nums[1])
