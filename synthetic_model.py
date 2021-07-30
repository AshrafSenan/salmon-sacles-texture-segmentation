import tensorflow as tf
import os
import random
import numpy as np
from PIL import Image, ImageOps
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from unet_model_builder import u_net_model 
import matplotlib.pyplot as plt
import preprocessing
data_path = 'generated_images'
seed = 42
np.random.seed = seed

IMG_COUNT = 1000
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1


#Load the dataset
Xs, Ys= preprocessing.read_data_set(data_path, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, 'mask')
Ys = to_categorical(Ys,   dtype='ubyte')
    
num_of_classes = 5
num_of_unet_layers = 5
start_filter_size = 16
input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#Split the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size=0.2, random_state = 3)

synthetic_model =  u_net_model(input_size,num_of_classes).construct_model(num_of_unet_layers,start_filter_size)

#Create the callback
backup_path = 'chkps/synthetic/parameters.ckp'

callback = tf.keras.callbacks.ModelCheckpoint(filepath=backup_path,
                                                 save_weights_only=True,
                                                 save_best_model = True,
                                                 verbose=1,
                                                 batch_size=20)
                                                
                                                 
                                                 
epochs = 1000

#Train the model                  
synthetic_model.fit(X_train, Y_train, batch_size=20, 
                     validation_split = 0.1,
                    verbose=1 ,callbacks=callback, epochs=epochs)

# Print the summary
print("Accuracy: ")
print(synthetic_model.evaluate(X_test, Y_test, batch_size=20))
                    
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix

predy = np.argmax(synthetic_model.predict(X_test), axis=3).flatten()
truy = np.argmax(Y_test, axis=3).flatten()
print('\n Regularized Classification Report\n')
print(classification_report(truy, predy, target_names=['Focus', 'Summer', 'River', 'Winter', 'Background']))
print("Confustion Matrix:")
print(confusion_matrix(truy, predy))



