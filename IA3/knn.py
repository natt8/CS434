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



def decision_stump(data, classifier):
	splits = []
	left_class = []
	right_class = []
	feature_data = []
	max_info_gain = -1
	best_feature = []
	best_split = 0
	#print data.shape
	#print classifier.shape
	tmp = column(data, 0) 
	print len(tmp)
	#print len(tmp[0])
	#print tmp[0][0]

	print classifier	

	for i in range(0, len(data[0])):
		print i
		info_gain = 0
		feature_data = np.array(column(data, i))
		np.sort(feature_data)
		splits =  find_splits(feature_data, classifier)
		for j in splits:
			left = []
			right = []
			for k in range(0, len(feature_data)):
				if feature_data[k] < j:
					left.append(feature_data[k])
					left_class.append(classifier[k])
				else:
					right.append(feature_data[k])
					right_class.append(classifier[k])

			p1 = float(len(left)) / float(len(feature_data))
			p2 = float(len(right)) / float(len(feature_data))
				
			base_entropy = entropy(feature_data, classifier)
			left_entropy =   entropy(left, left_class)
			right_entropy = entropy(right, right_class)

			#print "LEFT RIGHT ENTROPY"
			#print left_entropy
			#print right_entropy 
			node_entropy = -(p1 * left_entropy + (p2 * right_entropy))
			info_gain =  base_entropy + node_entropy
			if (info_gain > max_info_gain):
				max_info_gain = info_gain
				best_feature  = i #feature_data
				best_split = j
	print best_feature
	print best_split
	print max_info_gain		  	


#then build stump based on feature and split
#repeat this process for left and right branches of the stump
#do this until the node is pure 
#in which case it is a leaf and all points are of a single class 

def entropy(data, classifier):
	total = len(data)
	wrong = 0
	correct = 0
	entropy = 0
	p1 = 0
	p0 = 0
	correct =  count(classifier, 1.0)
	wrong = count(classifier, -1.0)
	#print "RIGHT THEN WRONG"
	#print correct
	#print wrong	

	p1 = float(correct) / float(total)
	p0 = float (wrong) / float(total)
	#print "p0"
	#print p0
	#print "p1"
	#print p1

	if( p1 > 0 and p0 > 0):
		#print math.log(p1)
		#print math.log(p0)
		entropy =  -(p1* math.log(p1)) - (p0* math.log(p0))

	return entropy


def find_splits(data , classifier):
	split = []
#for the data for feature order data from least to greatest
	for i in range(0, (len(data)-1)):
		#find 2 indecies that do not share a class, 
		if classifier[i] != classifier[i+1] : 
			#add values divide by 2, this is your split
			#keep list of splits
			split.append((data[i] + data[i+1]) / 2)
	return split



def count(classifier, label):
	count = 0
	for i in range(0, len(classifier)):
#		print classifier[i] , "== ", label , "?"
		if classifier[i] == label:
#			print True
			count = count + 1
	return count


def build_tree():
	return 0 
#given feature and split order data so that values less than split go to left and values greater than or equal to, go right, send that data through decision stump    

# HERE TO END COPIED FROM IA2
def get_data(filename):
    data = []
    myText = np.loadtxt(open(filename, "rb"), delimiter=",")
    x = list(myText)
    data = np.array(x).astype("float")

    y = (column(data, 0))
    data = (np.delete(data, 0, axis=1))

    return data, y

filename = "knn_train.csv"
filenameT = "knn_test.csv"

data, y = get_data(filename)
#w = matrix(np.zeros(256))
decision_stump(data, y)
