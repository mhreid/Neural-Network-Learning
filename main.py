import pandas as pd
import numpy as np
import random
import math

training = pd.read_csv('mnist_train.csv')
#print(training.head())
#This confirms that the data is there


def make_vector(index):
    num = training.loc[index, :][0]
    vector = training.loc[index,:][1:]
    return [num, vector]

def make_traindata():
    data = []
    for i in range(training.shape[0]):
        data.append(make_vector(i))
        #each vector is actually of format [number, [vector]]
    return data

def make_rand_layer(input, output):
    layer = []
    for i in range(input):
        row = []
        for j in range(output):
            row.append(random.random())
        layer.append(row)
    return layer

def make_layers(input_size, output_size, num):
    #hidden layer neurons are difference in input and output, (input must be greater size)
    hidden_size = input_size - output_size
    hidden_size = 20
    layers = [make_rand_layer(input_size, hidden_size)]
    for i in range(num - 1):
        layers.append(make_rand_layer(hidden_size, hidden_size))
    layers.append(make_rand_layer(hidden_size, output_size))
    return layers

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def vector_sigmoid(vector):
    out = []
    for x in vector:
        out.append(sigmoid(x))
    return out

def forward(layers, data):
    out = []
    for d in data:
        x = d[1]
        for layer in layers:
            x = np.dot(x, layer)
            x = vector_sigmoid(x)
        out.append([d[0], x])
    return out

if __name__ == '__main__':
    data = make_traindata()
    data = data
    print("data made")
    size = len(data[0][1])
    layers = make_layers(size,10,1)
    print("layers made")
    out = forward(layers, data)
    print("done!!")
    print(out[0][1])
