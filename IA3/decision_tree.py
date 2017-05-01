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


def decision_stump(data):
	splits = []
	feature_data = []
	max_info_gain = -1
	best_feature = []
	best_split = 0
	best_base = 0
	best_left = 0
	best_right = 0
	best_left_class = []
	best_right_class = []
	base_error = 0
	left_error = 0
	right_error = 0

	for i in range(1, len(data[0])):
		#print i
		info_gain = 0
		feature_data = np.array(column(data, i))
		np.sort(feature_data)
		sorted_data = sorted(data, key=lambda x: x[i])
		classifier = column(sorted_data, 0)
		splits, split_index = find_splits(feature_data, classifier)
	
		for j in range(0, len(splits)):
			left_class = []
			right_class = []
		
			left = (feature_data[:split_index[j] ])
			left_class = (classifier[:split_index[j]])
			left_len = len(left_class)

			right = (feature_data[split_index[j]:])
			right_class = (classifier[split_index[j]:])
			right_len = len(right)

			if(left_len > 0 and right_len > 0):
				p1 = float(len(left)) / float(len(feature_data))
				p2 = float(len(right)) / float(len(feature_data))
				base_entropy, base_label, base_error = entropy(classifier)
				left_entropy, left_label, left_error = entropy(left_class)
				right_entropy, right_label, right_error = entropy(right_class)
				node_entropy = -(p1 * left_entropy + (p2 * right_entropy))
				info_gain =  base_entropy + node_entropy
				if (info_gain > max_info_gain):
					data_right = data[:split_index[j]]
					data_left = data[split_index[j]:]
					max_info_gain = info_gain
					best_feature  = i #feature_data
					best_split = splits[j]
					best_split_index = split_index[j]
					best_base = base_label
					best_left = left_label
					best_right = right_label
					best_left_data = left
					best_right_data = right
					best_left_class = left_class
					best_right_class = right_class
			
	print "Best Feature: ", best_feature
	print "Best Split: ", best_split
	print "Info Gain: ", max_info_gain
	print "base branch class", best_base
	print "base branch error", base_error
	print "left branch class", best_left
	print "right branch class", best_right		  	

	return best_split, best_feature, best_left_class, best_right_class, best_base, base_error

def build_tree(data, test, depth, tree, di):
	total_error = 0
	test_error = 0
	if depth  < 6:
		print "-------------------------DEPTH ", depth, "---------"
		left = []
		right = []
		left_test = []
		right_test = []
		left_test_class = []
		right_test_class = []
		left_class = []
		right_class = []
		split, index, left_class, right_class, base_label, error = decision_stump(data)
		total_error = total_error + error 
		tree.append((index,split,depth,di,base_label))
	
		sorted_data = sorted(data, key=lambda x: x[index])
		sorted_test = sorted(test, key=lambda x: x[index])
		test_classifier = column(test, 0)
		if(len(test_classifier) > 1):
			en, lbl, t_error = entropy(test_classifier)
			test_error = test_error + t_error
			print "Test Error"
			print test_error

		print "total"
		print(len(data))
		for i in range(0, len(data)):
			if sorted_data[i][index] > split:
				left.append(data[i])
			else:
				right.append(data[i]) 
		for j in range(0, len(test)):
			if sorted_test[j][index] > split:
				left_test.append(test[j])
				left_test_class.append(test_classifier[j])
			else:
				right_test.append(test[j]) 
				right_test_class.append(test_classifier[j])


		print "right, left"
		print len(right)
		print len(left)
		if(len(left_class) > 1 ):	
			print "going left"
			train_error, t_error = build_tree(left, left_test, depth + 1, tree, "l")
			total_error = total_error + train_error
			test_error = test_error + t_error
	
		if(len(right_class) > 1):
			print "going right"
			train_error, t_error = build_tree(right, right_test, depth + 1, tree, "r")
			total_error = total_error + train_error
			test_error = test_error + t_error
	return total_error, test_error

#then build stump based on feature and split
#repeat this process for left and right branches of the stump
#do this until the node is pure 
#in which case it is a leaf and all points are of a single class 

def entropy(classifier):
	total = len(classifier)
	wrong = 0
	branch_label = 0
	correct = 0
	entropy = 0
	p1 = 0
	p0 = 0
	pos =  count(classifier, 1.0)
	neg = count(classifier, -1.0)
	if pos > neg:
		branch_label = 1.0
	else:
		branch_label = -1.0 

	p1 = float(pos) / float(total)
	p0 = float (neg) / float(total)
	error = min(p1,p0)

	if( p1 > 0 and p0 > 0):

		entropy =  ((-p1)* math.log(p1)) - (p0* math.log(p0))
	return entropy, branch_label, error 


def find_splits(data , classifier):
	split = []
	split_index = []
#for the data for feature order data from least to greatest
	for i in range(0, (len(data)-1)):
		#find 2 indecies that do not share a class, 
		if classifier[i] != classifier[i+1] : 
			#add values divide by 2, this is your split
			#keep list of splits
			split.append((data[i] + data[i+1]) / 2)
			split_index.append(i)
	return split, split_index


def count(classifier, label):
	count = 0
	for i in range(0, len(classifier)):
		if classifier[i] == label:
			count = count + 1
	return count

#given feature and split order data so that values less than split go to left and values greater than or equal to, go right, send that data through decision stump    

# HERE TO END COPIED FROM IA2
def get_data(filename):
    data = []
    myText = np.loadtxt(open(filename, "rb"), delimiter=",")
    x = list(myText)
    data = np.array(x).astype("float")

    y = (column(data, 0))

    return data, y

filename = "knn_train.csv"
filenameT = "knn_test.csv"

tree = []
data, y = get_data(filename)
dataT, yT = get_data(filenameT)
train_error, test_error = build_tree(data, dataT, 0, tree, "b")

print "Total training error"
print float(train_error) / float(30)
print "Total testing error"
print float(test_error) / float(30)


print "----------------TEST------------------------"
#build_test_tree(data, 0, tree)


for i in range(0, len(tree)):
	depth = tree[i][2]
	buildstring = ""
	for j in range(0, depth):
		buildstring = buildstring + "	"
	if tree[i][0] != []:
		print buildstring, tree[i][0], tree[i][1], tree[i][3], tree[i][4]
	



