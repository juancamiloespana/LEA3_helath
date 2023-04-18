import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########

import tensorflow as tf
from sklearn import metrics ### para analizar modelo




### cargar bases_procesadas ####

x_train = joblib.load('x_train.pkl')
y_train = joblib.load('y_train.pkl')
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')

############################################################
################ Probar modelos de redes neuronales #########
############################################################


######  normalizar variables ######
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255

##########Definir arquitectura de la red neuronal e instanciar el modelo ##########


fc_model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    
])

##### configura el optimizador y la función para optimizar ##############
fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC'])


#####Entrenar el modelo usando el optimizador y arquitectura definidas #########
fc_model.fit(x_train, y_train, batch_size=20, epochs=10, validation_data=(x_test, y_test))

#########Evaluar el modelo ####################

test_loss, test_acc, test_auc = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

pred_test=(fc_model.predict(x_test) > 0.5).astype('int')

cm=metrics.confusion_matrix(y_test,pred_test, labels=[0,1])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()

pred_train=(fc_model.predict(x_train) > 0.5).astype('int')
cm=metrics.confusion_matrix(y_train,pred_train, labels=[0,1])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()

print(metrics.classification_report(y_test, pred_test))