# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 02:57:45 2023

@author: Dr.Salama Ikki
"""

import cv2
from pyzbar.pyzbar import decode

import os
import numpy as np # linear algebra

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img

import pandas as pd
tf.random.set_seed(333)

tf.random.set_seed(333)

import time


# these functions are useful for turning lots of images into useful data
def genImageList(l):
    return list(filter(lambda i: i.split(".")[-1] in ["png","jpg","jpeg"], l))

def genDf(images,labels):
    df= pd.DataFrame(images,columns=["Image"])
    df["Class"] = labels
    return df

def genData(path):
    MaskPath = r''+path+'/with_mask/'
    NoMaskPath = r''+path+'/without_mask/'

    MaskImages = genImageList([MaskPath+i for i in os.listdir(MaskPath)])
    MaskLabels = np.zeros(len(MaskImages))

    NoMaskImages = genImageList([NoMaskPath+i for i in os.listdir(NoMaskPath)])
    NoMaskLabels = np.ones(len(NoMaskImages))

    MaskDf = genDf(MaskImages,MaskLabels)
    NoMaskDf = genDf(NoMaskImages,NoMaskLabels)

    Df = pd.DataFrame(shuffle(pd.concat([MaskDf,NoMaskDf])),columns=MaskDf.columns)
    return Df

def processData(df):
    x,y = df["Image"],df["Class"]
    data = []
    size = 100
    for img_path in x:
        image = load_img(img_path, target_size=(size, size))
        image = img_to_array(image)
        image = image/255
        data.append(image)

    data = np.array(data, dtype="float32")
    labels = np.array(y).reshape([-1,1])
    return data,labels

#imageInCode = cv2.imread( 'sampleQRcode.jpg' )

# this is where you can start looping

camera = cv2.VideoCapture(0)
result, img = camera.read()

# this is the path to save the picture to a specific folder
path = "Dataset\TestOneImage\with_mask" 
pathWithPicture = os.path.join( path, "test.jpg" )
cv2.imwrite(pathWithPicture, img)

path1 = "Dataset\Test" 
pathWithPicture1 = os.path.join( path1, "test.jpg" )
cv2.imwrite(pathWithPicture1, img)

imageInCode = cv2.imread( pathWithPicture )
modelledWithoutTraining = keras.models.load_model("PortableModel")

code = decode(imageInCode)
try:
    print( code[0].data )
    if code[0].data == b'exception_case':
        print("===========")
        print("success")
        print("===========")
except: # test first against files of images, then against one image in a file
    print("No Valid QR Codes present")

    # this file that testDF refers to *must* have subfolders with_mask and without_mask
    # or elsse crash
    testDF = genData("Dataset/TestOneImage")
    print(testDF)
    testx,testy = processData( testDF )
    

    
    #baseModel.predict( testx )
    # this asks if someone is *not* wearing a mask
    # high value equals high certainty they are not wearing a mask
    # low value equals low certainty of *not* wearing a mask (they are likely wearing a mask)
    #baseModel.save("PortableModel") # saves the original model, can be used here once you retrieve it too i suppose
    
    results = modelledWithoutTraining.predict( testx )
    print("{}".format( results ))
    if results[0] > 0.50:
        print("=====================")
        print("User not wearing mask")
        print("=====================")
    if results[0] <= 0.5:
        print("====================")
        print("User is wearing mask")
        print("====================")
        
        
    # load that image
    image = load_img( "Dataset/Test/test.jpg" , target_size=(100, 100))
    
    image = img_to_array(image) # turn it into an array so it can be proccessed by AI
    image = tf.expand_dims(image, axis=0)
    image = image/255 # grayscale
    
    results = modelledWithoutTraining.predict( image )
    print("{}".format( results ))
    if results[0] > 0.50:
        print("=====================")
        print("User not wearing mask")
        print("=====================")
    if results[0] <= 0.5:
        print("====================")
        print("User is wearing mask")
        print("====================")

time.sleep(10)
os.remove(pathWithPicture)
os.remove(pathWithPicture1)


# check an entire folder to see how accurate the model is
testDF = genData("Dataset/AccuracyTest/SmallerSet") # just find a different dataset to do so
print(testDF)
testx,testy = processData( testDF )
results = modelledWithoutTraining.predict( testx )

correct = 0
incorrect = 0

truePositive = 0 # is wearing mask, guess right
trueNegative = 0 # is not masking, guess right
falsePositive = 0 # is not wearing, guess wrong
falseNegative = 0 # is wearing maskm guess wrong

threshold = 0.5

for i in range( len(results)):
    if results[i] > threshold and testy[i] == 1:
        correct = correct + 1
        truePositive = truePositive + 1
        
    elif results[i] <= threshold and testy[i] == 0:
        correct = correct + 1
        trueNegative = trueNegative + 1
        
    elif results[i] > threshold and testy[i] == 0: # actually wearing mask
        incorrect = incorrect + 1
        falseNegative = falseNegative + 1
        
    elif results[i] <= threshold and testy[i] == 1: # not wearing mask
        incorrect = incorrect + 1
        falsePositive = falsePositive + 1
        
    
total = correct + incorrect

print("")
print("Test Accuracy on new Dataset")
print("Threshold : {}".format( threshold ) )
print("")
print("total number of readable files : {}".format(total) )
print("{}% correct ".format( (correct / total) * 100) )
print("{}% incorrect ".format( (incorrect / total) * 100)  )
print("")
print("{}% True Positive ".format( (truePositive / total) * 100))
print("{}% True Negative ".format( (trueNegative / total) * 100))
print("{}% False Positive ".format( (falsePositive / total) * 100))
print("{}% False Negative ".format( (falseNegative / total) * 100))









