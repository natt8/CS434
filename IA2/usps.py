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
	errorList = []	
	dOld = np.zeros((len(data[0]),1))
	while (convergence != True):
		print w
		print data[x]
		dNew = np.zeros((len(data[0]),1))
		dataT = data.T
		print "----------------------------------------------"
		for i in range(1, len(data)):
			print "i ", i
			yHat = 1 / (1 + math.exp(((-1 *(w)) * data[i-1].T)))
			errors = y[i] - yHat

			errorList.append(errors)		
			dNew = dNew + (errors.item(0) * data[i])
		w = w + (learningRate * dNew)

		if(np.linalg.norm((dNew - dOld), 'fro') < .01 or x > 100000):
			convergence = True;	
			exponent = (-1*(w)) * (data[i-1].T)
			yHat = 1/(1+ math.exp(exponent))

			errors = y[i] - yHat
			#print "errors"

			errorList.append(errors)		
			d = d + (errors.item(0) * data[i])

		w = w + (learningRate * d) 

		if(x < 10):#w < error):
			#convergence = True;	
			x = x + 1
			
		else:
			dOld = dNew;
		#print d
	return d

def get_train_data():
#	reader = open("usps-4-9-train.csv", "rb")
	data = []
	myText = np.loadtxt(open("usps-4-9-train.csv", "rb"), delimiter=",")
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

dataNo, data, y = get_train_data()
w = matrix(np.zeros(256))

d = batch_gradient_descent(data, y, w)	
print d
print w
print d
