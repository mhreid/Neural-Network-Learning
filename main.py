import pandas as pd
import numpy as np

TRAINING = pd.read_csv('mnist_train.csv')
#print(training.head())
#This confirms that the data is there


def make_vector(index):
    num = TRAINING.loc[index, :][0]
    vector = TRAINING.loc[index,:][1:]
    return [num, vector]

if __name__ == '__main__':
    print(make_vector(1))
