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
from operator import itemgetter

def column(dataset, i):
    return [row[i] for row in dataset]

def euclidian_distance(instance1, instance2, s):
    distance = 0

    for x in range(s):
        distance += pow((instance1[x] - instance2[x]), 2)

    return math.sqrt(distance)

def calc_distances(data, a, b):
    distances = []
    distance = []

    temp = [0, 0, 0]
    #make 2D array of distances (284x284)
    for i in range(0, a):
        distance.append(temp)

    for j in range(0, a):
        distances.append(distance)

    #calculate the euclidian distance between all data points except itself
    for i in range(0, a):
        for j in range(0, a):
            if(j != i):
                distances[i][j] = (euclidian_distance(data[i], data[j], b), i, j)

    return distances

def get_neighbors(data, distances, k):
    neighbors = []

    # for each of the 284 data points find the k closest points
    for i in range(0, len(distances)):
        templist = []
        neighbor = distances[i]

        #order the values in the neighbor list by distance, neighbor[x][0]
        sorted(distances, key=itemgetter(0))

        #put the closest k values in a list
        for x in range(0, k):
            templist.append(neighbor[x])

        neighbors.append(templist)

    return neighbors

def respond(responses, neighbors, k):
    sum_resps = 0
    response = 0

    for i in range(0, k):
        tupl = neighbors[i]
        a = tupl[1]
        b = tupl[2]
        sum_resps += responses[b]

    #average the responses to return as the response
    response = sum_resps / k

    if(response == 0):
        randnum = random.randint(0, 1)

        if(randnum == 0):
            response = -1
        else:
            response = 1
    
    return response

def get_responses(neighbors, a, k):
    responses = []
    
    #init responses to be all 0
    for i in range(0, len(data)):
        responses.append(0)

    for i in range(0, a):
        responses[i] = respond(responses, neighbors[i], k)

    print responses
    return responses

def calc_error(data, responses):
    error = 0

    feature_data = np.array(column(data, 0))

    for i in range(0, len(feature_data)):
        if(feature_data[i] != responses[i]):
            error += 1

    return error

def knn(data, test, splits, k, n):
    total_error = 0
    test_error = 0
    error = 0
    distances = []
    neighbors = []
    responses = []

    # TRAINING DATA 
    #store the dimmentions of the data array for future use 
    a = len(data)
    b = len(data[0])

    distances = calc_distances(data, a, b)

    neighbors = get_neighbors(data, distances, k)

    responses = get_responses(neighbors, a, k)

    total_error = calc_error(data, responses)

    # LEAVE ONE OUT CROSS VALIDATION
    sum_error = 0
    loo_error = 0
    a =  len(data) / n
    
    for i in range(0, n):
        b = len(splits[i][0])

        print splits[i]
        distances = calc_distances(splits[i], a, b)
        
        neighbors = get_neighbors(splits[i], distances, k)

        responses = get_responses(neighbors, a, k)

        sum_error += calc_error(splits[i], responses)

    loo_error = sum_error / n

    # TESTING DATA
    #store the dimmentions of the data array for future use 
    a = len(dataT)
    b = len(dataT[0])
    
    distances = calc_distances(dataT, a, b)

    neighbors = get_neighbors(dataT, distances, k)

    responses = get_responses(neighbors, a, k)

    test_error = calc_error(data, responses)

    return total_error, test_error, loo_error

def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)

    return dataset_split

def count(classifier, label):
	count = 0
	for i in range(0, len(classifier)):
		if classifier[i] == label:
			count = count + 1
	return count

def scale_data(data):
    for i in range(1, len(data[0])):
        max_col = 0
        feature_data = np.array(column(data, i))

        # find the max values for each column
        for j in range(1, (len(feature_data)-1)):
            if(feature_data[j] > max_col):
                max_col = feature_data[j]

            if(feature_data[j]*-1 > max_col):
                max_col = feature_data[j] * -1

        # scale the data if the max isn't 1
        for j in range(1, len(data[i])):
            if(max_col > 1):
                data[i][j] = feature_data[j] / int(max_col)
            
    return data

def get_data(filename):
    data = []
    myText = np.loadtxt(open(filename, "rb"), delimiter=",")
    x = list(myText)
    data = np.array(x).astype("float")

    y = (column(data, 0))

    return data, y

#define k here
k = 1
num_folds = 5

#define filename here
filename = "knn_train.csv"
filenameT = "knn_test.csv"

#read in the data
data, y = get_data(filename)
dataT, yT = get_data(filenameT)

#scale the data
data = scale_data(data)
dataT = scale_data(dataT)

#split the data for leave one out cross validation
folds = cross_validation_split(data, num_folds)

#run knn on training and test data
train_error, test_error, loo_error = knn(data, dataT, folds, k, num_folds)

print "Total training error"
print float(train_error) / float(len(data))
print "Leave-one-out cross-validation error"
print float(loo_error)
print "Number of testing errors"
print float(test_error)

