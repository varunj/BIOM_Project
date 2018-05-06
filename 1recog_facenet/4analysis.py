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

'''
	#classes 136 - 1(w-006)	
'''

# ar
# CLASSES = ['m-001', 'm-002', 'm-003', 'm-004', 'm-005', 'm-006', 'm-007', 'm-008', 'm-009', 'm-010', 'm-011', 'm-012', 'm-013', 'm-014', 'm-015', 'm-016', 'm-017', 'm-018', 'm-019', 'm-020', 'm-021', 'm-022', 'm-023', 'm-024', 'm-025', 'm-026', 'm-027', 'm-028', 'm-029', 'm-030', 'm-031', 'm-032', 'm-033', 'm-034', 'm-035', 'm-036', 'm-037', 'm-038', 'm-039', 'm-040', 'm-041', 'm-042', 'm-043', 'm-044', 'm-045', 'm-046', 'm-047', 'm-048', 'm-049', 'm-050', 'm-051', 'm-052', 'm-053', 'm-054', 'm-055', 'm-056', 'm-057', 'm-058', 'm-059', 'm-060', 'm-061', 'm-062', 'm-063', 'm-064', 'm-065', 'm-066', 'm-067', 'm-068', 'm-069', 'm-070', 'm-071', 'm-072', 'm-073', 'm-074', 'm-075', 'm-076', 'w-001', 'w-002', 'w-003', 'w-004', 'w-005', 'w-007', 'w-008', 'w-009', 'w-010', 'w-011', 'w-012', 'w-013', 'w-014', 'w-015', 'w-016', 'w-017', 'w-018', 'w-019', 'w-020', 'w-021', 'w-022', 'w-023', 'w-024', 'w-025', 'w-026', 'w-027', 'w-028', 'w-029', 'w-030', 'w-031', 'w-032', 'w-033', 'w-034', 'w-035', 'w-036', 'w-037', 'w-038', 'w-039', 'w-040', 'w-041', 'w-042', 'w-043', 'w-044', 'w-045', 'w-046', 'w-047', 'w-048', 'w-049', 'w-050', 'w-051', 'w-052', 'w-053', 'w-054', 'w-055', 'w-056', 'w-057', 'w-058', 'w-059', 'w-060']

# disguise
CLASSES = ['P1', 'P2',  'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P50', 'P51', 'P52', 'P53', 'P54', 'P55', 'P56', 'P57', 'P58', 'P59', 'P60', 'P61', 'P62', 'P63', 'P64', 'P65', 'P66', 'P67', 'P68', 'P69', 'P70', 'P71', 'P72', 'P73', 'P74', 'P75']

# plastic

def readFile(path):
	f = open(path, 'r+')
	trueLabels, predLabels, scoresPred, scores = [], [], [], []
	for eachLine in f.readlines():
		splitt = eachLine[:-1].replace(']','').replace('[','').split(',')
		
		if (len(splitt) == 3):
			trueLabels.append(CLASSES.index(splitt[0].strip()))
			predLabels.append(CLASSES.index(splitt[1].strip()))	
			scoresPred.append(float(splitt[2].strip()))

		if (len(splitt) == 135 or len(splitt) == 75):
			scores.append([float(x) for x in splitt])

	trueLabels, predLabels, scores, scoresPred = np.asarray(trueLabels), np.asarray(predLabels), np.asarray(scores), np.asarray(scoresPred)

	# house cleaning
	# print('nos classes true, pred: ', len(set(trueLabels)), len(set(predLabels)))
	# print('classes only in trueLabels', [CLASSES[x] for x in np.setdiff1d(trueLabels,predLabels)])
	# print('classes only in predLabels', [CLASSES[x] for x in np.setdiff1d(predLabels,trueLabels)])
	# print('trueLabels, predLabels, scores, scoresPred shapes: ', trueLabels.shape, predLabels.shape, scores.shape, scoresPred.shape)
	
	# for eachDiff in np.setdiff1d(trueLabels,predLabels):
	# 	for eachIndex in np.where(trueLabels == eachDiff)[0]:
	# 		trueLabels = np.delete(trueLabels, eachIndex) 
	# 		predLabels = np.delete(predLabels, eachIndex) 
	# 		scores = np.delete(scores, eachIndex, axis=0) 
	# 		scoresPred = np.delete(scoresPred, eachIndex)

	return trueLabels, predLabels, scores, scoresPred



def calcMetrics(trueLabels, predLabels, scores, filename):
	# cmc curve
	genuineArr, imposterArr = [], []
	for x in range(0,len(trueLabels)):
		genuineArr.append(scores[x][trueLabels[x]])
		imposterArr.append(np.concatenate((scores[x][:trueLabels[x]], scores[x][trueLabels[x]+1:]), axis=0))
	genuineArr = np.asarray(genuineArr).flatten()
	imposterArr = np.asarray(imposterArr).flatten()
	p, x = np.histogram(genuineArr, bins=int(len(scores)/30))
	pMatch = p/max(p)
	x = x[:-1] + (x[1] - x[0])/2
	plt.plot(x, pMatch, label='genuine')
	p2, x2 = np.histogram(imposterArr, bins=int(len(scores)/30))
	pImposter = p2/max(p2)
	x2 = x2[:-1] + (x2[1] - x2[0])/2
	plt.plot(x2, pImposter, label='imposter')
	plt.tight_layout()
	plt.legend()
	plt.title('Match Score Distribution')
	plt.xlabel('Score')
	plt.ylabel('P(Match at Score)')
	plt.tight_layout()
	splitt = filename.split('_')
	plt.savefig(splitt[0] + '_matchscore.png')
	plt.close()


	# roc, tpr, fpr
	trueLabelsBin = label_binarize(trueLabels, classes=list(set(trueLabels)))
	tpr, fpr, roc_auc = {}, {}, {}
	for i in range(len(CLASSES)-1):
		fpr[i], tpr[i], _ = roc_curve(trueLabelsBin[:, i], scores[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	fpr["micro"], tpr["micro"], thresholds = roc_curve(trueLabelsBin.ravel(), scores.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])	
	# for i in range(len(CLASSES)):
	# 	try:
	# 		plt.plot(fpr[i], tpr[i])
	# 	except:
	# 		pass
	plt.plot(fpr["micro"], tpr["micro"])
	plt.plot([0, 1], [0, 1], 'k--', label='ROC Curve (AUC = %0.6f)' % (roc_auc["micro"]))
	plt.tight_layout()
	plt.legend()
	plt.title('ROC')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.savefig(splitt[0] + '_roc.png')
	plt.close()


	print('prec, acc, tpr, fpr \t: ', precision_score(trueLabels, predLabels, average='weighted')*100, \
		accuracy_score(trueLabels, predLabels)*100, np.average(tpr["micro"])*100, np.average(fpr["micro"])*100)

	# EER: point where TPR FPR meet 135deg line OR  the common value when the false acceptance rate (FAR) and false rejection rate (FRR) are equal
	fnr = 1 - tpr["micro"]
	eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr["micro"])))]
	eer = min(fpr["micro"][np.nanargmin(np.absolute((fnr - fpr["micro"])))], fnr[np.nanargmin(np.absolute((fnr - fpr["micro"])))])
	print('equal err rate \t\t: ', eer*100)

	# HTER: 1- 0.5(TP / (TP + FN) + TN / (TN + FP)) OR (FAR[index of EER] + FRR[index of EER])/2
	hter = sum([fpr["micro"][np.nanargmin(np.absolute((fnr - fpr["micro"])))], fnr[np.nanargmin(np.absolute((fnr - fpr["micro"])))]])/2
	print('half total err rate \t: ', hter*100)



if __name__ == "__main__":
	for eachFile in glob.iglob('models/model-20180422-113901_test.txt'):
		print(eachFile)
		trueLabels, predLabels, scores, scoresPred = readFile(eachFile)
		print(trueLabels.shape, predLabels.shape, scores.shape, scoresPred.shape)
		calcMetrics(trueLabels, predLabels, scores, eachFile)
		print()