#!/usr/bin/python
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy import matrix
from random import seed
from random import randrange
import csv
from math import sqrt
import math
import random

def column(dataset, i):
    return [row[i] for row in dataset]

def knn():
    return


# HERE TO END COPIED FROM IA2
def get_data(filename):
    data = []
    myText = np.loadtxt(open(filename, "rb"), delimiter=",")
    x = list(myText)
    data = np.array(x).astype("float")

    y = matrix(column(data, 256)).T
    data = matrix(np.delete(data, 256, axis=1))

    return data, y

filename = "knn_train.csv"
filenameT = "knn_test.csv"

data, y = get_data(filename)
w = matrix(np.zeros(256))

iterations, avAccuracy, w2, lossList = batch_gradient_descent(data, y, w,0)	
graph_error_over_iterations(iterations, avAccuracy, 0)
graph_convergence(iterations, lossList, 0)


#Test data
data, y = get_data(filenameT)
#print w2
iterations, avAccuracy, wunsued, lossList = batch_gradient_descent(data, y, w2, 0)	
graph_error_over_iterations(iterations, avAccuracy, 1)
