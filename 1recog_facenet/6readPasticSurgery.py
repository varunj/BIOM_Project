import scipy.io
import numpy as np
import cv2
import os

def readImages(filePath):
	mat = scipy.io.loadmat(filePath)
	after = np.asarray(mat['after'])

	before, beforeLabels = [[] for x in range(900)], []
	for x, eachImg in enumerate(np.asarray(mat['before'])[0]):
		before[x] = eachImg[0][0][0]
		beforeLabels.append(eachImg[0][0][1][0])
	before = np.asarray(before)
	beforeLabels = np.asarray(beforeLabels)

	after, afterLabels = [[] for x in range(900)], []
	for x, eachImg in enumerate(np.asarray(mat['after'])[0]):
		after[x] = eachImg[0][0][0]
		afterLabels.append(eachImg[0][0][1][0])
	after = np.asarray(after)
	afterLabels = np.asarray(afterLabels)

	return before, beforeLabels, after, afterLabels


def dumpFolder(before, after, folderPath):
	for x, eachImg in enumerate(before):
		cv2.imwrite(os.path.join(folderPath, str(x) + '_before.png'), before[x][:,:,::-1])
		cv2.imwrite(os.path.join(folderPath, str(x) + '_after.png'), after[x][:,:,::-1])


if __name__ == "__main__":
	before, beforeLabels, after, afterLabels = readImages('data/IIITPlasticSurgeryDb.mat')
	print('before, after dataaa shapes: ', before.shape, after.shape)
	print('before, after labels shapes: ', beforeLabels.shape, afterLabels.shape)

	dumpFolder(before, after, 'data/plasticSurgery/')
