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

def batch_gradient_descent(data, y, w):
	convergence = False
	learningRate = .0000001
	errors = 0
	yHat = 0
	x = 0
	max_exponent = 0;
	iterations = []
	avAccuracyList = []
	dOld = np.zeros((len(data[0]),1))
	while (convergence != True):
		correct = 0
		dNew = np.zeros((len(data[0]),1))
		dataT = data.T
		print "----------------------------------------------"
		for i in range(1, len(data)):
			exponent = (-1*(w)) * (data[i-1].T)
			yHat = 1/(1+ math.exp(exponent))
			yHatRound = round(yHat)
	
			errors = y[i] - yHatRound
			if(errors == 0):
				correct = correct + 1
			dNew = dNew + (errors.item(0) * data[i])
		w = w + (learningRate * dNew)

		avAccuracy = float(correct) / float(len(y))
		print correct, " Correct, out of ", len(y) 
		print "% Accurracy" , avAccuracy
		avAccuracyList.append(avAccuracy)
		iterations.append(x)	

		#if(np.linalg.norm((dNew - dOld), 'fro') < .1 or x > 10000):
		if(x >= 100):
			convergence = True;	
		else:
			print np.linalg.norm((dNew - dOld), 'fro')
			print x
			dOld = dNew;
			x = x + 1

	return iterations, avAccuracyList, w

def graph_error_over_iterations(iterations, avAccuracyList, source):
	it = np.array(iterations)
	err = np.array(avAccuracyList)
	err.shape = it.shape

	print it.shape
	print err.shape	
	plt.figure(1)
	if(source == 0):
		plt.plot(it, err, 'r', label='train')
	else:
		plt.plot(it, err, 'b', label='test')
	plt.xlabel('Iterations')
	plt.ylabel('% Accuracy')
	plt.savefig('gradientDescentImprovement.png')


def get_data(filename):
	data = []
	myText = np.loadtxt(open(filename, "rb"), delimiter=",")
	x = list(myText)
	data = np.array(x).astype("float")

	y = matrix(column(data, 256)).T
	data = matrix(np.delete(data, 256, axis=1))

	return data, y

filename = "usps-4-9-train.csv"
filenameT = "usps-4-9-test.csv"

data, y = get_data(filename)
w = matrix(np.zeros(256))

iterations, avAccuracy, w2 = batch_gradient_descent(data, y, w)	
graph_error_over_iterations(iterations, avAccuracy, 0)

#Test data
data, y = get_data(filenameT)
#print w2
iterations, avAccuracy, wunsued = batch_gradient_descent(data, y, w2)	
graph_error_over_iterations(iterations, avAccuracy, 1)
