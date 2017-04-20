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
	learningRate = .000001
	errors = 0
	yHat = 0
	x = 0
	errorList = []	
	while (convergence != True):
		d = np.zeros((len(data[0]),1))
		dataT = data.T
		print "----------------------------------------------"
		for i in range(1, 1400):
			print "i ", i
			print "w shape"
			print (w).shape
			print "y"
			print y.shape
			print data[i-1].shape
			yHat = 1 / (1 + math.exp(((-1 *(w)) * data[i-1].T)))
			print yHat	
			errors = y[i] - yHat
			print "errors"

			#print errors[0]
			#print errors[0].shape
			errorList.append(errors)		
			d = d + (errors.item(0) * data[i])
			print "d shape"
			print d.shape
		w = w + (learningRate * d) 
		if(x < 10):#w < error):
			#convergence = True;	
			x = x + 1
		else:
			convergence = True		
		#print "d"
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

#print "w"
#print w
#print "len y"
#print len(y)

#print y

#print "len data"
#print len(data)
#print "len data 0"
#print len(data[0])

#print "len w"
#print len(w)

#print "data 0"
#print data[0]

#print y.shape
#print w.shape
#print data.shape
for i in range(0, len(data), 10): 
	dataBatch = data[[i, i+9],]
	yBatch = y[[i, i+9],]
	d = batch_gradient_descent(data, y, w)	
	print d
print w
print d
