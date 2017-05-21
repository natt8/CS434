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
#from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy import spatial as sp


def column(dataset, i):
    return [row[i] for row in dataset]

def kmeans(k, data):
	#d = #distance function between examples
	#select k random samples from data d
	centers = []
	iter_list = []
	SSE_list = []
	centers = select_random_centers(k, data)
	print "items in 1 center"
	print len(centers[0])
	print "number of centers"	
	print len(centers)
	min_SSE = 999999999999999999
	iterations = 0 
	while(iterations < 5):	
		cluster = [[] for _ in xrange(k)]

		for x in data:
		#assign xi to cj such that d(uj, xi) 
			#is minimized

			min_center = distance(centers, x)
			#print "closest center"
			#print min_center
			cluster[min_center].append(x)
		count = 0  
		for j in cluster:
		#for each cluster (points and center)
			sum_ex = 0
			for example in j:
			#for each point example cluster j
				#ex_norm = np.linalg.norm(example)
				sum_ex = sum_ex + matrix(example)
			new_center = sum_ex / len(j)
			centers[count] = new_center
			#print centers[count]
			count = count + 1
		SSE = objective(cluster, centers) 
		SSE_list.append(SSE)
		iter_list.append(iterations)		
		if SSE < min_SSE:
			min_SSE = SSE
			print "best min so far ", iterations
			print min_SSE
		iterations = iterations + 1
#			uj = 1/size of cluster j  * sum x for x in clusterj
#	until convergence/objective cannot be improved
	if k == 2:
		plot_kmeans(SSE_list, iter_list, 'k2means2.png')
	return min_SSE

def plot_kmeans(SSEs, iterations, name):
	plt.plot(np.array(iterations), np.array(SSEs), 'r')
	plt.xlabel('Iterations')
	plt.ylabel('SSE')
	plt.savefig(name)


def select_random_centers(k, data):
	return random.sample(data, k);
	

def distance(centers, point):
	min_distance_center = 0
	min_distance = 999999999999999
	for i in range(0, len(centers)):
	#	print len(centers) 
		distance = sp.distance.cosine(point, centers[i])#cos_sim(point, centers[i])
		#print distance, i
		if distance < min_distance:
			min_distance = distance
			min_distance_center = i
			
	#min_distance_center = random.randint(0,1)	

	return min_distance_center


#objective: minimize SSE, compute center of each cluster 
def objective(cluster, centers):
	minimum = 0
	norm_sums = 0
	count = 0
	for k in cluster:
		for x in k:
			norm = np.linalg.norm(matrix(x) - matrix(centers[count])) 
			norm_sums = norm_sums + norm
		count = count + 1
	return norm_sums #SSE 

def print_images(images):
	for i in range(0, len(images)): 
		plt.imshow(np.reshape(images[i],28,28))
	

def get_data(filename):
	data = []
   # myText = np.loadtxt(open(filename, "rb"), delimiter=",")
   # x = list(myText)
   # data = np.array(x).astype("float")

	with open(filename) as file:
		data = [[float(num) for num in line.split(",")] for line in file]

    #y = (column(data, 0))

	return data#, y

filename = "data-1.txt"
#filenameT = "knn_test.csv"

data = get_data(filename)

ks = []
mins = []
for k in range(3,10):
	abs_min = 999999999
	for i in range(0,10):
		minSSE = kmeans(k, data)
		if minSSE < abs_min:
			abs_min = minSSE
	mins.append(abs_min)
	ks.append(k)	
plot_kmeans(mins, ks, 'diffks.png')


#dataT = get_data(filenameT)
