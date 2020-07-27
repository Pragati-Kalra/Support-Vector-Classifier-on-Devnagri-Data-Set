import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
from datetime import datetime
from sklearn.svm import SVC
datadirectory="C:\\Users\\hp\\Downloads\\devanagari-character-dataset-large\\dhcd\\train"
testdirectory="C:\\Users\\hp\\Downloads\\devanagari-character-dataset-large\\dhcd\\test"
categories=["0","1","2","3","4","5","6","7","8","9"]
training_data=[]
testing_data=[]
Y_testing=[]
for category in categories:
    path=os.path.join(datadirectory,category)
    class_num=categories.index(category)
    
    for img in os.listdir(path):
        
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array=cv2.resize(img_array,(30,30))
            new_array=new_array.flatten()
            new_array=new_array.reshape(900,1)
            #print(new_array)
            #print(new_array.shape)
           
            training_data.append([new_array,class_num])
           
            #plt.imshow(new_array,cmap="gray")
            #plt.show()
           
        except Exception as e:
            pass
       

for category in categories:
    path=os.path.join(datadirectory,category)
    class_num=categories.index(category)
    
    for img in os.listdir(path):
        
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array=cv2.resize(img_array,(30,30))
            new_array=new_array.flatten()
            new_array=new_array.reshape(900,1)
            #print(new_array)
            #print(new_array.shape)
           
            testing_data.append(new_array)
            Y_testing.append(class_num)
            #plt.imshow(new_array,cmap="gray")
            #plt.show()
           
        except Exception as e:
            pass
training_data=np.array(training_data)
testing_data=np.array(testing_data)
print(testing_data.shape)
testing_data=testing_data.reshape(17000,900)



#print(training_data)
#print(type(img_array))

random.shuffle(training_data)
#random.shuffle(testing_data)

X=[]
Y=[]
for features,label in training_data:
    #print(features.shape)
    X.append(features)
    Y.append(label)

X=np.array(X).reshape(-1,900)
#Y=np.array(Y).reshape(-1,1)
#print(X)
#X=X.T

#Y=Y.T
#print(Y)
#print(Y.shape)

#print(datetime.now())

'''
print(testing_data.shape)
a=testing_data[0]
print(a.shape)

a=a.T
a=a.tolist()
A=[]
A.append(a)
'''
clf=SVC(gamma=0.0001,C=100)
clf.fit(X,Y)
l=len(training_data)
for i in range(l):
    a=testing_data[i]
    a=a.T
    a=a.tolist()
    A=[]
    A.append(a)
    print(Y_testing[i],end=" ")
    print(clf.predict(A))

a=testing_data[10800]
p=a
p=p.reshape(30,30)
print(p.shape)
a=a.T
a=a.tolist()
A=[]
A.append(a)
print(clf.predict(A))
plt.imshow(p,cmap="gray")#",interpolation="nearest")
plt.show()
