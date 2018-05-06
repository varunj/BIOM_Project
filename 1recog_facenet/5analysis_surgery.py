import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.nan)
import os
import pickle
from pprint import pprint
from time import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score
import glob
from sklearn.preprocessing import label_binarize
import sklearn
import random


def readFile(path):
	f = open(path, 'r+')
	trueLabels, predLabels, scoresPred, scores = [], [], [], []
	for eachLine in f.readlines():
		splitt = eachLine[:-1].replace(']','').replace('[','').split(',')
		
		if (len(splitt) == 3):
			trueLabels.append(splitt[0].strip())
			predLabels.append(splitt[1].strip())	
			scoresPred.append(float(splitt[2].strip()))

		if (len(splitt) == 135 or len(splitt) == 75):
			scores.append([float(x) for x in splitt])

	trueLabels, predLabels, scores, scoresPred = np.asarray(trueLabels), np.asarray(predLabels), np.asarray(scores), np.asarray(scoresPred)

	return trueLabels, predLabels, scores, scoresPred



def calcMetrics(trueLabels, predLabels, scores, filename):

	print('prec, acc \t: ', precision_score(trueLabels, predLabels, average='weighted')*100, \
		accuracy_score(trueLabels, predLabels)*100)


if __name__ == "__main__":
	for eachFile in glob.iglob('models/model-20180422-125750_test.txt'):
		print(eachFile)
		trueLabels, predLabels, scores, scoresPred = readFile(eachFile)
		print(trueLabels.shape, predLabels.shape, scores.shape, scoresPred.shape)
		calcMetrics(trueLabels, predLabels, scores, eachFile)
		print()