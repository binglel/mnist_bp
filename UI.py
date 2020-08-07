# encoding: utf-8
# ******************************************************
# requirement:python3
# Author: chyb
# Last modified: 20181023 14:00
# Email:chyb3.14@gmail.com
# Filename:UI.py
# Description:UI create by pyqt5 for network training and testing
# ******************************************************

import sys
from PyQt5.QtWidgets import (QTextEdit,QGridLayout,QGroupBox,QHBoxLayout,QLabel,QFileDialog,QWidget, QPushButton, QLineEdit, QInputDialog, QApplication)
from PyQt5.QtGui import QImage,QPixmap
import numpy as np
import math
from input_data import get_train_data,get_test_data
import cv2

class BPNetwork(object):
    def __init__(self,feature_length,input_numbers,hiden_numbers,output_kinds,base_lr1,base_lr2,base_lr3):
        self.feature_length=feature_length
        self.input_numbers=input_numbers
        self.hiden_numbers=hiden_numbers
        self.output_kinds=output_kinds
        self.hiden_lr=base_lr1
        self.out_lr=base_lr2
        self.input_lr=base_lr3
        self.w1=0.5*(np.random.random((self.feature_length, self.input_numbers))-0.5)
        self.w2=0.5*(np.random.random((self.input_numbers, self.hiden_numbers))-0.5)
        self.w3=0.5*(np.random.random((self.hiden_numbers, self.output_kinds))-0.5)
        
        self.input_offset=np.zeros(self.input_numbers)
        self.hiden_offset=np.zeros(self.hiden_numbers)
        self.output_offset=np.zeros(self.output_kinds)
    def sigmoid(self,x):
        output=[]
        for i in x:
            output.append(1/(1+math.exp(-i)))
        output=np.array(output)
        return output
    def forward(self,input_feature):
        self.input_feature=input_feature
        self.input_val=np.dot(self.input_feature,self.w1)+self.input_offset
        self.input_out=self.sigmoid(self.input_val)
        self.hiden_val=np.dot(self.input_out,self.w2)+self.hiden_offset
        self.hiden_out=self.sigmoid(self.hiden_val)
        self.out_val=np.dot(self.hiden_val,self.w3)+self.output_offset
        self.out_out=self.sigmoid(self.out_val)
        
    def backward(self,label):
        self.erro=label-self.out_out
        delta_out=self.erro*self.out_out*(1-self.out_out)
        delta_hiden=self.hiden_out*(1-self.hiden_out)*np.dot(self.w3,delta_out)
        delta_input=self.input_out*(1-self.input_out)*np.dot(self.w2,delta_hiden)
        for i in range(0,self.output_kinds):
            self.w3[:,i]+=self.hiden_lr*delta_out[i]*self.hiden_out
        for i in range(0,self.hiden_numbers):
            self.w2[:,i]+=self.out_lr*delta_hiden[i]*self.input_out
        for i in range(0,self.input_numbers):
            self.w1[:,i]+=self.input_lr*delta_input[i]*self.input_feature
        self.output_offset+=self.out_lr*delta_out
        self.hiden_offset+=self.hiden_lr*delta_hiden
        self.input_offset+=self.input_lr*delta_input
    def reduce_lr(self,lr1,lr2,lr3):
        self.hiden_lr=lr1
        self.out_lr=lr2
        self.input_lr=lr3
        
class BPUI(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.initUI()
    
    def initUI(self):
        self.gridGroupBox=QGroupBox("GridLayout")
        layout=QGridLayout()
        
        layout.setSpacing(10)
        
        self.btn=QPushButton("Import",self)
        self.btn.clicked.connect(self.showDialog)
        
        self.predict_btn=QPushButton("Predict",self)
        self.predict_btn.clicked.connect(self.predict)
        
        self.train_btn=QPushButton("Train",self)
        self.train_btn.clicked.connect(self.train)
        
        self.test_btn=QPushButton("Test",self)
        self.test_btn.clicked.connect(self.test)
        
        self.imageView=QLabel()
        layout.addWidget(self.imageView,5,1,2,2)
  
        self.imageName=QLabel("Image:")
        self.imageLineEdit=QLineEdit()
        self.testName=QLabel("Accuracy:")
        self.testLineEdit=QLineEdit()
        self.numberLabel = QLabel("number:")
        self.numberLineEdit = QLineEdit()
        self.iterName=QLabel("Iter:")
        self.iterLineEdit=QLineEdit("See command window")
        self.lossName=QLabel("Loss:")
        self.lossLineEdit=QLineEdit("See command window")
        
        layout.addWidget(self.train_btn,0,0)
        layout.addWidget(self.iterName,0,1)
        layout.addWidget(self.iterLineEdit,0,2)
        layout.addWidget(self.lossName,1,1)
        layout.addWidget(self.lossLineEdit,1,2)
        layout.addWidget(self.test_btn,2,0)
        layout.addWidget(self.testName,2,1)
        layout.addWidget(self.testLineEdit,2,2)
        layout.addWidget(self.btn,3,0)
        layout.addWidget(self.imageName,3,1)
        layout.addWidget(self.imageLineEdit,3,2)
        layout.addWidget(self.predict_btn,4,0)
        layout.addWidget(self.numberLabel,4,1)
        layout.addWidget(self.numberLineEdit,4,2)
        
        self.setLayout(layout)

        
        self.setGeometry(200,200,400,400)
        self.setWindowTitle("Predict")
        self.show()
    def showDialog(self):
        self.filename,  _ = QFileDialog.getOpenFileName(self, 'Open file', './') 
        if len(self.filename):
            self.image = QImage(self.filename)
            self.imageView.setPixmap(QPixmap.fromImage(self.image))
            self.imageLineEdit.setText(self.filename)
    
    def train(self):
        self.iterLineEdit.setText("xcz")
        sample, label= get_train_data()
        sample = np.array(sample,dtype='float') 
        sample=(sample)/256.0
        samp_num = len(sample)     
        inp_num = len(sample[0])    
        out_num = 10                
        hid_num = 15 
        loss=0
        self.BP=BPNetwork(inp_num,20,hid_num,out_num,0.1,0.1,0.1)
        for step in range(0,3):
            if step==1:
                self.BP.reduce_lr(0.01,0.01,0.01)
            elif step==2:
                self.BP.reduce_lr(0.001,0.001,0.001)
            for i in range(0,samp_num):
                train_label = np.zeros(out_num)
                train_label[label[i]] = 1
                self.BP.forward(sample[i])
                self.BP.backward(np.array(train_label))
                if i%10000==0:
                    print(str(i+60000*step))
                    error=self.BP.erro
                    loss=0
                    for j in range(0,len(error)):
                        loss=loss+abs(error[j])
                    print(loss)
        self.iterLineEdit.setText("ending")
        self.lossLineEdit.setText("ending")
        
    def test(self):
        correct=0  
        test_s, test_l = get_test_data()
        test_s = np.array(test_s,dtype='float') 
        test_s =(test_s)/256.0
        for i in range(0,len(test_s)):
            self.BP.forward(test_s[i])
            result=self.BP.out_out
            if np.argmax(result) == test_l[i]:
                correct+=1
        accuracy=float(correct)/float(len(test_s))
        self.testLineEdit.setText(str(accuracy))
    def predict(self):
        im=cv2.imread(self.filename,0)
        im=im.reshape((1,784))
        L=[]
        for i in range(0,784):
            L.append(im[0,i])
        L=np.array(L)
        L=L/256.0
        self.BP.forward(L)
        result=self.BP.out_out
        a=np.argmax(result)
        self.numberLineEdit.setText(str(a))
        
        
if __name__=="__main__":
    app=QApplication(sys.argv)
    ex=BPUI()
    sys.exit(app.exec_())
        