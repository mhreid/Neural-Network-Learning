import pandas as pd
import numpy as np
import random
import math

training = pd.read_csv('mnist_train.csv')
testing = pd.read_csv('mnist_test.csv')



def make_traindata():
    nums = training.iloc[:,0]
    nums = nums.as_matrix()
    data = training.iloc[:,1:]
    data = data.as_matrix()
    return nums,data

def make_testdata():
    nums = testing.iloc[:,0]
    nums = nums.as_matrix()
    data = testing.iloc[:,1:]
    data = data.as_matrix()
    return nums,data

def make_rand_layer(input, output):
    layer = (np.random.rand(input, output) - .5)
    return layer

def make_layers(input_size, output_size, num):
    hidden_size = 50
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
    y = [0 for i in range(len(y_hat))]
    y[num] = 1
    return 2 * (y - y_hat)

def d_loss_sum_all(nums, y_hats):
    sum = []
    for i in range(nums.shape[0]):
        sum.append(d_loss_sum(nums[i], y_hats[:][i]))
    return np.matrix(sum)
def total_error(nums, output):
    m = len(nums)
    error = 0
    for i in range(m):
        y = [0 for i in range(len(output[0]))]
        diff = (y - output[i]) ** 2
        error += np.sum(diff)
    return error / m

def gradients(nums, a, z, layers, biases):

    d_layers = []
    d_biases = []
    d_cost = [np.multiply(-d_loss_sum_all(nums, a[-1]), sigmoid_all(z[-1], sigmoid_derivative))]
    #print("Current Cost: " + str(np.sum(d_cost)))
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

def save(layers, biases, modifier):
    #modifier is a string so you can save different iterations
    for i in range(len(layers)):
        f1 = "./layer" + modifier + str(i) + ".csv"
        f2 = "./bias" + modifier + str(i) + ".csv"
        np.savetxt(f1, layers[i], delimiter=",")
        np.savetxt(f2, biases[i], delimiter=",")
def open(num, modifier):
    layers = []
    biases = []
    for i in range(num):
        layers.append(np.genfromtxt("layer" + modifier +  str(i) + ".csv",delimiter=','))

        biases.append(np.genfromtxt("bias" + modifier + str(i) + ".csv",delimiter=','))
    return layers, biases

if __name__ == '__main__':

    nums, data = make_traindata()
    tnums, tdata = make_testdata()
    nums = nums[:3000]
    data = data[:3000]
    tnums = nums[:100]
    tdata = data[:100]
    #layers, biases = make_layers(data.shape[1], 10, 3)
    #save(layers, biases)
    #print(biases[-1])

    layers, biases = open(3, "3l")
    #a, z = forward(layers, biases, data)
    #print(a[-1][1])
    #print(nums[1])


    layers, output = back_prop(layers, biases, data, nums, .1, 100,3)
    a, z = forward(layers, biases, tdata)

    output = a[-1]
    print(output[14])
    print(tnums[14])
    print(output[10])
    print(tnums[10])
    print(output[12])
    print(tnums[12])
    print(total_error(tnums, output))
    save(layers, biases, "3l")
