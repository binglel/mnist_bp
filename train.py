# encoding: utf-8
# ******************************************************
# requirement:python3
# Author: chyb
# Last modified: 20181023 14:00
# Email:chyb3.14@gmail.com
# Filename:train.py
# Description:
# ******************************************************
import numpy as np
import math
from input_data import input_data
import matplotlib.pyplot as plt

import time
from tqdm import trange
from random import random, randint


class BPNetwork(object):
    def __init__(self,feature_length,input_numbers,hiden_numbers,output_kinds,base_lr1,base_lr2,base_lr3):
        self.feature_length=feature_length
        self.input_numbers=input_numbers
        self.hiden_numbers=hiden_numbers
        self.output_kinds=output_kinds
        self.hiden_lr=base_lr1
        self.out_lr=base_lr2
        self.input_lr=base_lr3
        """
        the initialization for weights is important
        """
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
        """
        reduce learning rate during training
        """
        self.hiden_lr=lr1
        self.out_lr=lr2
        self.input_lr=lr3
if __name__=="__main__":
    print ("*******Data loading*******")
    sample, label, test_s, test_l = input_data()
    sample = np.array(sample,dtype='float') 
    sample=(sample)/256.0
    test_s = np.array(test_s,dtype='float') 
    test_s =(test_s)/256.0

    samp_num = len(sample)
    inp_num = len(sample[0])
    out_num = 10
    hid_num = 15  
    BP=BPNetwork(inp_num,30,hid_num,out_num,0.1,0.1,0.1)
    X=[]
    Loss=[]
    Loss_temp=[]
    loss=0
    logtext = open("./saved/train_test_log.txt", "w")
    print("*******Model trainng*******")
    logtext.write("*******Model trainng*******" + "\n")
    for step in range(0,1):
        if step==1:
            BP.reduce_lr(0.01,0.01,0.01)
        elif step==2:
            BP.reduce_lr(0.001,0.001,0.001)
        with trange(samp_num) as t:
            # for i in range(0,samp_num):
            for i in t:
                # 设置进度条左边显示的信息
                t.set_description("step %i" % i)

                train_label = np.zeros(out_num)
                train_label[label[i]] = 1

                BP.forward(sample[i])
                BP.backward(np.array(train_label))
                error=BP.erro
                for j in range(0,len(error)):
                        loss=loss+abs(error[j])
                Loss_temp.append(loss)
                if i%100==99 and step==0:
                    Loss.append(sum(Loss_temp)/len(Loss_temp))
                    Loss_temp=[]
                    X.append(i+step*60000)
                    # 设置进度条右边显示的信息
                    t.set_postfix(loss=loss, step=i)
                    logtext.write("step:" + str(i) + " " + "loss:" + str(loss) + "\n")
                    loss=0

    print ("*******Model Testing*******")
    logtext.write("*******Model Testing*******" + "\n")
    correct=0  
    for i in range(0,len(test_s)):
        BP.forward(test_s[i])
        result=BP.out_out
        if np.argmax(result) == test_l[i]:
            correct+=1
    print ("Predicted Correct:", correct)
    print ("Predicted Correct Rate:", float(correct)/float(len(test_s)))
    logtext.write("Predicted Correct:" + str(correct) + "\n"+ "Predicted Correct Rate:"+str(float(correct)/float(len(test_s))))
    logtext.close()
    plt.figure()
    plt.plot(X,Loss)
    plt.savefig("./saved/train_loss.jpg")
    plt.show()
    
    
    
            
        

            
            
            
        