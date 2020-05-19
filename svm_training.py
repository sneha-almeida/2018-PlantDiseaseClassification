import tkinter
from tkinter import *
from PIL import Image
import os
from tkinter import filedialog
import numpy as np
import cv2
import skimage
from skimage import io
from skimage.io import imread_collection
#from PIL import image
import pandas as pd
import skimage.feature as sk
import numpy as np
import openpyxl
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import openpyxl
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier  #Library for BPNN
from sklearn.metrics import classification_report, confusion_matrix
import xlrd
import ast
import pickle



print("**************************SVM*****************************************")
filename = 'E:\plant_classification.xlsx'
data = pd.read_excel(filename)

z = data.drop('Class', axis=1)
x = z.drop('ASM', axis=1)
y = data['Class']

#print(x)
#print(y)

train_data,test_data,train_label,test_label =train_test_split(x,y,test_size = 0.1)

#print(train_data)
#print(train_label)
#print(test_data)
#print(test_label)

# training a linear SVM classifier 


print("Creating Classifier..............")
clf = SVC(kernel = 'linear', C = 1)

print("Training..................")
clf.fit(train_data,train_label)
print("Training Complete................")
filename = 'E://finalized_model_svm.sav'
pickle.dump(clf, open(filename, 'wb'))
