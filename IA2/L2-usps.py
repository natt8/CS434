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
		#print w
		#print data[x]
		dNew = np.zeros((len(data[0]),1))
		dataT = data.T
		print "----------------------------------------------"
		for i in range(1, len(data)):
		#	print "i ", i
			exponent = (-1*(w)) * (data[i-1].T)
			yHat = 1/(1+ math.exp(exponent))
			yHatRound = round(yHat)
	
			errors = y[i] - yHatRound
			if(errors == 0):
				correct = correct + 1
			dNew = dNew + (errors.item(0) * data[i])
		w = w + (learningRate * dNew)

		#sumError = sum(errorList)
		avAccuracy = float(correct) / float(len(y))
		print correct, " Correct, out of ", len(y) 
		print "% Accurracy" , avAccuracy
		avAccuracyList.append(avAccuracy)
		iterations.append(x)	

		#if(np.linalg.norm((dNew - dOld), 'fro') < .1 or x > 10000):
		if(x >= 200):
			convergence = True;	
		else:
			print np.linalg.norm((dNew - dOld), 'fro')
			print x
			dOld = dNew;
			x = x + 1

	return iterations, avAccuracyList, w

def graph_error_over_iterations(iterations, avErrorList, source):
	#print iterations
	#print avError
		

	it = np.array(iterations)
	err = np.array(avErrorList)
	err.shape = it.shape



	print it.shape
	print err.shape	
	plt.figure(1)
	if(source == 0):
	#	plt.subplot(211)
		plt.plot(it, err, 'r', label='train')
	else:
	#	plt.subplot(212)
		plt.plot(it, err, 'b', label='test')
	plt.xlabel('Iterations')
	plt.ylabel('% Accuracy')
	plt.savefig('gradientDescentImprovement.png')


def get_train_data(filename):
#	reader = open("usps-4-9-train.csv", "rb")
	data = []
	myText = np.loadtxt(open(filename, "rb"), delimiter=",")
	x = list(myText)
	data = np.array(x).astype("float")
	#rowTemp  = []
	#for line in reader:
	#	rowTemp = line.split(",")
#		numbers = [float(n) for n in rowTemp]
	#	data.append(rowTemp)#numbers)
	rowNum = len(data)
	y = matrix(column(data, 256)).T
	data = matrix(np.delete(data, 256, axis=1))

	#Add columns of ones 
	dataD = data
#	b = np.ones((rowNum,1))
#	dataD = np.append(dataD, b, axis=1)

	return data, dataD, y

filename = "usps-4-9-train.csv"
filenameT = "usps-4-9-test.csv"

dataNo, data, y = get_train_data(filename)
w = matrix(np.zeros(256))

iterations, avError, w2 = batch_gradient_descent(data, y, w)	
graph_error_over_iterations(iterations, avError, 0)

dataNo, data, y = get_train_data(filenameT)
#w = matrix(np.zeros(256))
#print w2
iterations, avError, wunsued = batch_gradient_descent(data, y, w2)	
graph_error_over_iterations(iterations, avError, 1)
