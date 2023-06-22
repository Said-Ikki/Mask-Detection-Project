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

camera = cv2.VideoCapture(0)
result, img = camera.read()

# this is the path to save the picture to a specific folder
path = "Dataset\TestOneImage\with_mask" 
pathWithPicture = os.path.join( path, "test.jpg" )


# this is where you can start looping

cv2.imwrite(pathWithPicture, img)

imageInCode = cv2.imread( pathWithPicture )
modelledWithoutTraining = keras.models.load_model("PortableModel")

code = decode(imageInCode)
try:
    print( code[0].data )
    if code[0].data == b'exception_case':
        print("===========")
        print("success")
        print("===========")
except:
    print("No Valid QR Codes present")

    # this file that testDF refers to *must* have subfolders with_mask and without_mask
    # or elsse crash
    testDF = genData("Dataset/TestOneImage")
    testx,testy = processData( testDF )
    #baseModel.predict( testx )
    # this asks if someone is *not* wearing a mask
    # high value equals high certainty they are not wearing a mask
    # low value equals low certainty of *not* wearing a mask (they are likely wearing a mask)
    #baseModel.save("PortableModel") # saves the original model, can be used here once you retrieve it too i suppose
    
    results = modelledWithoutTraining.predict( testx )
    print("{}".format( results ))

time.sleep(10)
os.remove(pathWithPicture)
    