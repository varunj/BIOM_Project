import os
from os import walk, getcwd
from PIL import Image
import cv2
import glob
from shutil import copyfile
import random


def trainTestSplitAR():
    for eachfile in glob.iglob('data/AR_FaceDB_haarNose_folder/*/*.png'):
        splitt = eachfile.split('/')

        className = splitt[3][:5]
        filename = splitt[3][6:].zfill(12)

        if (random.random() < 0.7):
            newPathFolder = os.path.join('data/AR_FaceDB_haarNose_train', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        else:
            newPathFolder = os.path.join('data/AR_FaceDB_haarNose_test', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        if not os.path.exists(newPathFolder):
            os.mkdir(newPathFolder)

        copyfile(eachfile, newPath)
        print(newPath)



def trainTestSplitNetworkAR():
    for eachfile in glob.iglob('data/AR_FaceDB_haarNose_folder/*/*.png'):
        splitt = eachfile.split('/')

        className = splitt[3][:5]
        filename = splitt[3][6:].zfill(12)

        randd = random.random()

        if (randd < 0.5):
            newPathFolder = os.path.join('data/AR_FaceDB_haarNose_network', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        elif (randd < 0.75):
            newPathFolder = os.path.join('data/AR_FaceDB_haarNose_train', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)

        else:
            newPathFolder = os.path.join('data/AR_FaceDB_haarNose_test', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        if not os.path.exists(newPathFolder):
            os.mkdir(newPathFolder)

        copyfile(eachfile, newPath)
        print(newPath)



def trainTestSplitNetworkDisguise():
    for eachfile in glob.iglob('data/disguise_haarNose_folder/*/*.png'):
        splitt = eachfile.split('/')

        print(splitt)
        className = splitt[3].split('_')[0]
        filename = splitt[3].split('_')[1].split('.')[0].zfill(9) + '.png'

        randd = random.random()

        if (randd < 0.5):
            newPathFolder = os.path.join('data/disguise_haarNose_network', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        elif (randd < 0.75):
            newPathFolder = os.path.join('data/disguise_haarNose_train', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)

        else:
            newPathFolder = os.path.join('data/disguise_haarNose_test', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        if not os.path.exists(newPathFolder):
            os.mkdir(newPathFolder)

        copyfile(eachfile, newPath)
        print(newPath)



def trainTestSplitNetworkPlastic():
    for eachfile in glob.iglob('data/plasticSurgery_haarNose_folder/*/*.png'):
        splitt = eachfile.split('/')

        print(splitt)
        className = splitt[3].split('_')[0]
        filename = splitt[3].split('_')[1].split('.')[0].zfill(9) + '.png'

        randd = random.random()

        if (int(className) < 450):
            newPathFolder = os.path.join('data/plasticSurgery_haarNose_network', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        elif (int(className) < 675):
            newPathFolder = os.path.join('data/plasticSurgery_haarNose_train', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)

        else:
            newPathFolder = os.path.join('data/plasticSurgery_haarNose_test', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        if not os.path.exists(newPathFolder):
            os.mkdir(newPathFolder)

        copyfile(eachfile, newPath)
        print(newPath)



def trainTestSplitNetworkARFace():
    for eachfile in glob.iglob('data/AR_FaceDB_haarFace_folder/*/*.png'):
        splitt = eachfile.split('/')

        className = splitt[3][:5]
        filename = splitt[3][6:].zfill(12)

        randd = random.random()

        if (randd < 0.5):
            newPathFolder = os.path.join('data/AR_FaceDB_haarFace_network', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        elif (randd < 0.75):
            newPathFolder = os.path.join('data/AR_FaceDB_haarFace_train', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)

        else:
            newPathFolder = os.path.join('data/AR_FaceDB_haarFace_test', className)
            newPath = os.path.join(newPathFolder, className + '_' + filename)
        
        if not os.path.exists(newPathFolder):
            os.mkdir(newPathFolder)

        copyfile(eachfile, newPath)
        print(newPath)


# trainTestSplitAR()
# trainTestSplitNetworkAR()
# trainTestSplitNetworkDisguise()
# trainTestSplitNetworkPlastic()
trainTestSplitNetworkARFace()