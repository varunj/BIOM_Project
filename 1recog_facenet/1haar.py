import numpy as np
import cv2
import os

DELTA_X = 20/2
DELTA_Y = 50/2

def performHAARandSaveImgAR(inpFolder, outFolder):
	nosDetections, sumx, sumy = 0, 0, 0
	face_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_nose.xml')

	for filename in os.listdir(inpFolder):
		img = cv2.imread(os.path.join(inpFolder,filename))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		noses = face_cascade.detectMultiScale(gray, 1.3, 5)

		nosLocalDetec = 0
		for (x,y,w,h) in noses:
			nosLocalDetec = nosLocalDetec + 1
			nosDetections = nosDetections + 1
			splitt = filename.split('.')
			filenameTemp = splitt[0] + '-' + str(nosLocalDetec) + '.png'

			crop = img[y-DELTA_Y:y+h+DELTA_Y, x-DELTA_X:x+w+DELTA_X]
			sumx = sumx + crop.shape[0]
			sumy = sumy + crop.shape[1]
			crop = cv2.resize(crop, (182, 182)) 
			cv2.imwrite(os.path.join(outFolder,filenameTemp), crop)
			
			print('done: ' + filename)

	print('total detected: ', nosDetections)
	print('av x, y: ', sumx/nosDetections, sumy/nosDetections)


def performHAARandSaveImgDisguise(inpFolder, outFolder):
	nosDetections, sumx, sumy = 0, 0, 0
	face_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_nose.xml')

	for filename in os.listdir(inpFolder):
		img = cv2.imread(os.path.join(inpFolder,filename))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		noses = face_cascade.detectMultiScale(gray, 1.3, 5)

		nosLocalDetec = 0
		for (x,y,w,h) in noses:
			try:
				nosLocalDetec = nosLocalDetec + 1
				nosDetections = nosDetections + 1
				splitt = filename.split('.')
				filenameTemp = splitt[0] + '-' + str(nosLocalDetec) + '.png'

				crop = img[y-DELTA_Y:y+h+DELTA_Y, x-DELTA_X:x+w+DELTA_X]
				sumx = sumx + crop.shape[0]
				sumy = sumy + crop.shape[1]
				crop = cv2.resize(crop, (182, 182)) 
				cv2.imwrite(os.path.join(outFolder,filenameTemp), crop)
				
				print('done: ' + filename)
			except:
				print('err')
				pass

	print('total detected: ', nosDetections)
	print('av x, y: ', sumx/nosDetections, sumy/nosDetections)



def performHAARandSaveImgARFacewithoutNose(inpFolder, outFolder):
	nosDetections, sumx, sumy = 0, 0, 0
	face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
	nose_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_nose.xml')

	for filename in os.listdir(inpFolder):
		img = cv2.imread(os.path.join(inpFolder,filename))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		nosLocalDetec = 0
		for (x,y,w,h) in faces:
			nosLocalDetec = nosLocalDetec + 1
			nosDetections = nosDetections + 1
			crop = img[y-DELTA_Y:y+h+DELTA_Y, x-DELTA_X:x+w+DELTA_X]

			noses = nose_cascade.detectMultiScale(crop, 1.3, 5)
			for (x1,y1,w1,h1) in noses:
				nosLocalDetec = nosLocalDetec + 1
				
				splitt = filename.split('.')
				filenameTemp = splitt[0] + '-' + str(nosLocalDetec) + '.png'

				cv2.rectangle(crop,(x1-DELTA_X,y1-DELTA_Y),(x1+w1+DELTA_X, y1+h1+DELTA_Y),(0,0,0),-1)
				crop = cv2.resize(crop, (182, 182)) 
				cv2.imwrite(os.path.join(outFolder,filenameTemp), crop)

			sumx = sumx + crop.shape[0]
			sumy = sumy + crop.shape[1]
			
		print('done: ' + filename)

	print('total detected: ', nosDetections)
	print('av x, y: ', sumx/nosDetections, sumy/nosDetections)


if __name__ == "__main__":
	# performHAARandSaveImgAR('data/AR_FaceDB/data/', 'data/AR_FaceDB_haarNose/data/')
	# performHAARandSaveImgDisguise('data/disguise/', 'data/disguise_haarNose/')
	# performHAARandSaveImgDisguise('data/plasticSurgery/', 'data/plasticSurgery_haarNose/')
	performHAARandSaveImgARFacewithoutNose('data/AR_FaceDB/data/', 'data/AR_FaceDB_haarFace/data/')