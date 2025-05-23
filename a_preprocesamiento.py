import numpy as np

##!pip install opencv-python
import cv2 ### para leer imagenes jpeg


from matplotlib import pyplot as plt ## para gráfciar imágnes
import _funciones as fn#### funciones personalizadas, carga de imágenes
import joblib ### para descargar array

############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

img1=cv2.imread('data\\test\\NORMAL\\IM-0005-0001.jpeg')
img2 = cv2.imread('data\\train\\PNEUMONIA\\person7_bacteria_29.jpeg')


############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

plt.imshow(img1)
plt.title('normal')
plt.show()

plt.imshow(img2)
plt.title('pneumonia')
plt.show()

###### representación numérica de imágenes ####

img2.shape ### tamaño de imágenes
img1.shape
img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel

np.prod(img2.shape) ### 5 millones de variables representan cada imágen

#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes

img1_r = cv2.resize(img1 ,(100,100))
plt.imshow(img1_r)
plt.title('Normal')
plt.show()
np.prod(img1_r.shape)

img2_r = cv2.resize(img2 ,(100,100))
plt.imshow(img2_r)
plt.title('Normal')
plt.show()
np.prod(img2_r.shape)

################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################


width = 100 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'data/train/'
testpath = 'data/test/'

x_train, y_train, file_list= fn.img2data(trainpath, width=100) #Run in train
x_test, y_test, file_list = fn.img2data(testpath, width=100) #Run in test




#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)



x_train.shape
y_train.shape

x_train[0].shape ### ver tamaño de la primera imágen
np.prod(x_train[0].shape)



x_test.shape
y_test.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "salidas\\x_train.pkl")
joblib.dump(y_train, "salidas\\y_train.pkl")
joblib.dump(x_test, "salidas\\x_test.pkl")
joblib.dump(y_test, "salidas\\y_test.pkl")


