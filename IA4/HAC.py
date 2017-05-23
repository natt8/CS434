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
from numpy import dot
import operator
from itertools import combinations
from functools import reduce
 
def column(dataset, i):
    return [row[i] for row in dataset]

def get_data(filename):
	data = []
   # myText = np.loadtxt(open(filename, "rb"), delimiter=",")
   # x = list(myText)
   # data = np.array(x).astype("float")

	with open(filename) as file:
		data = [[float(num) for num in line.split(",")] for line in file]

    #y = (column(data, 0))

	return data#, y

def HAC(vectors, threshold):
    def similarity(pair):
        #calculate the similarity between the pair
        a = np.array([vectors[i] for i in pair[0]])
        b = np.array([vectors[i] for i in pair[1]])

        b = np.matrix.transpose(b)

        sol = np.matrix.dot(a, b)

        return sol[0][0]

    size = len(vectors)
    labels = [0 for i in range(size)]

    # Initially: num_clusters == num_items
    clusters = {(i,) for i in range(size)}

    # Init label
    j = len(clusters)

    while True:
        pairs = combinations(clusters, 2)

        # calculate the similarity for each pair in p
        scores = [(pair, similarity(pair)) for pair in pairs]

        # Get the highest similarity to determine which pair shall be merged
        max_sim = max(scores, key=operator.itemgetter(1))

        # break if the highest similarity is below the threshold
        if(max_sim[1] < threshold):
            break

        # Remove the pair to be merged
        pair = max_sim[0]
        clusters -= set(pair)

        # Flatten the pair
        flat_pair = reduce(lambda x,y: x + y, pair)

        # update labels for the flat_pair
        for i in flat_pair:
            labels[i] = j

        # add the flattened and relabeled pair to the clusters
        clusters.add(flat_pair)

        # break if there is only one cluster left
        if len(clusters) == 1:
            break

        # increment label
        j += 1

    return labels

filename = "data-2.txt"

data = get_data(filename)

labels = HAC(data, 2)

print labels

#ks = []
#mins = []
#for k in range(3,10):
#	abs_min = 999999999
#	for i in range(0,10):
#		minSSE = kmeans(k, data)
#		if minSSE < abs_min:
#			abs_min = minSSE
#	mins.append(abs_min)
#	ks.append(k)	
#plot_kmeans(mins, ks, 'diffks.png')


