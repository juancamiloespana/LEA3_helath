import numpy as np

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
import cv2 ### funcion para leer imágenes
from os import listdir ### para leer rutas
from tqdm import tqdm
from os.path import  join ### para unir ruta con archivo 

from keras.models import Sequential

from sklearn import metrics 

img1 = cv2.imread('chest_xray/train/NORMAL/IM-0117-0001.jpeg')
img2 = cv2.imread('chest_xray/train/PNEUMONIA/person7_bacteria_29.jpeg')

plt.imshow(img1)
plt.title('normal')
plt.show()

plt.imshow(img2)
plt.title('pneumonia')
plt.show()


img2.shape
img2 = cv2.resize(img2 ,(100,100))
plt.imshow(img2)
plt.title('pneumonia')
plt.show()

width = 128 #set width of the image
num_classes = 2 #set class of the image
trainpath = 'chest_xray/train/'
testpath = 'chest_xray/test/'
trainImg = [trainpath+f for f in listdir(trainpath)]
testImg = [testpath+f for f in listdir(testpath)]


def img2data(path):
    rawImgs = [] 
    labels = []
    
    for imagePath in (path):
        for item in tqdm(listdir(imagePath)):
            file = join(imagePath, item)
            if file[-1] =='g': ### verificar que se imágen extensión jpg o jpeg
                img = cv2.imread(file)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) 
                img = cv2.resize(img ,(width,width)) 
                rawImgs.append(img)
                l = imagePath.split('/')[2] 
                if l == 'NORMAL':
                    labels.append([0])
                elif l == 'PNEUMONIA':
                    labels.append([1])
    return rawImgs, labels

x_train, y_train= img2data(trainImg) #Run in train
x_test, y_test = img2data(testImg) #Run in test


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

np.max(x_train)
x_train.shape
y_train.shape
x_test.shape
y_test.shape
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo


x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255

fc_model=keras.models.Sequential([
    layers.Flatten(input_shape=(128,128,3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    
])

x1=x_train[1]
x2=x1.flatten('C')
x2.shape

128*128*3


fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC'])


fc_model.fit(x_train, y_train, batch_size=20, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc, test_auc = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

unique, counts=np.unique(y_train, return_counts=True)
counts[1]/(counts[0]+counts[1])

pred_test=(fc_model.predict(x_test) > 0.5).astype('int')

cm=metrics.confusion_matrix(y_test,pred, labels=[0,1])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()

pred_train=(fc_model.predict(x_train) > 0.5).astype('int')

cm=metrics.confusion_matrix(y_train,pred, labels=[0,1])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()

print(metrics.classification_report(y_test, pred_test))

x_train[1]



#####



