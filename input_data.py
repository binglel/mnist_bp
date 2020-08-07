# encoding: utf-8
# ******************************************************
# requirement:python3
# Author: chyb
# Last modified: 20181023 14:00
# Email:chyb3.14@gmail.com
# Filename:input_data.py
# Description: DataReader for mnist
# ******************************************************

import numpy as np 
import struct
import matplotlib.pyplot as plt 
import os
from PIL import Image
import cv2

class DataUtils(object):
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'    
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def getImage(self):
        binfile = open(self._filename, 'rb') 
        buf = binfile.read() 
        binfile.close()
        index = 0
        numMagic,numImgs,numRows,numCols=struct.unpack_from(self._fourBytes2,buf,index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            images.append(imgVal)
        images=np.array(images)
        return images
    def getLabel(self):
        binFile = open(self._filename,'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems= struct.unpack_from(self._twoBytes2, buf,index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2,buf,index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY):
        m, n = np.shape(arrX)
        for i in range(1):
            img = np.array(arrX[i])
            img = img.reshape(28,28)
            outfile = str(i) + "_" +  str(arrY[i]) + ".png"
            plt.figure()
            plt.imshow(img, cmap = 'binary')
            plt.savefig(self._outpath + "/" + outfile)
def get_train_data():
    trainfile_X = './mnist/train-images.idx3-ubyte'
    trainfile_y = './mnist/train-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    return train_X, train_y
def get_test_data():
    testfile_X = './mnist/t10k-images.idx3-ubyte'
    testfile_y = './mnist/t10k-labels.idx1-ubyte'
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()
    return test_X, test_y

def input_data():
    trainfile_X = './mnist/train-images.idx3-ubyte'
    trainfile_y = './mnist/train-labels.idx1-ubyte'
    testfile_X = './mnist/t10k-images.idx3-ubyte'
    testfile_y = './mnist/t10k-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()
    return train_X, train_y, test_X, test_y

if __name__ == "__main__":
    train_images,train_labels,test_images,test_labels = input_data()
    print (train_images.shape)
    print (train_labels.shape)
    pic=train_images[0].reshape(28,28)
    plt.figure(1)
    plt.imshow(pic,cmap = plt.cm.gray)
    plt.show()
    for i in range(50,51):
        im=np.array(train_images[i].reshape(28,28),dtype='uint8')
        im=Image.fromarray(im)
        im.save('%d.jpg'%(i-50))
