import os
from os import walk, getcwd
from PIL import Image
import cv2
import glob
from shutil import copyfile



# for eachfile in glob.iglob('data/AR_FaceDB_haarNose/data/*.png'):
#     splitt = eachfile.split('/')

#     className = splitt[3][:5]
#     filename = splitt[3][6:].zfill(12)


#     newPathFolder = os.path.join('data/AR_FaceDB_haarNose_folder', className)
#     newPath = os.path.join(newPathFolder, className + '_' + filename)
    
#     if not os.path.exists(newPathFolder):
#         os.mkdir(newPathFolder)

#     copyfile(eachfile, newPath)
#     print(eachfile, newPath)




# for eachfile in glob.iglob('data/disguise_haarNose/*.png'):
#     splitt = eachfile.split('/')

#     className = splitt[2].split('_')[0]
#     filename = splitt[2].split('_')[1].split('.')[0].zfill(9) + '.png'


#     newPathFolder = os.path.join('data/disguise_haarNose_folder', className)
#     newPath = os.path.join(newPathFolder, className + '_' + filename)
    
#     if not os.path.exists(newPathFolder):
#         os.mkdir(newPathFolder)

#     copyfile(eachfile, newPath)
#     print(eachfile, newPath)




# for eachfile in glob.iglob('data/plasticSurgery_haarNose/*.png'):
#     splitt = eachfile.split('/')

#     className = splitt[2].split('_')[0]
#     filename = splitt[2].split('_')[1].split('.')[0].zfill(9) + '.png'


#     newPathFolder = os.path.join('data/plasticSurgery_haarNose_folder', className)
#     newPath = os.path.join(newPathFolder, className + '_' + filename)
    
#     if not os.path.exists(newPathFolder):
#         os.mkdir(newPathFolder)

#     copyfile(eachfile, newPath)
#     print(eachfile, newPath)


for eachfile in glob.iglob('data/AR_FaceDB_haarFace/data/*.png'):
    splitt = eachfile.split('/')

    className = splitt[3][:5]
    filename = splitt[3][6:].zfill(12)


    newPathFolder = os.path.join('data/AR_FaceDB_haarFace_folder', className)
    newPath = os.path.join(newPathFolder, className + '_' + filename)
    
    if not os.path.exists(newPathFolder):
        os.mkdir(newPathFolder)

    copyfile(eachfile, newPath)
    print(eachfile, newPath)
