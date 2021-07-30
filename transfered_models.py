import tensorflow as tf
import os
import random
import numpy as np
from PIL import Image, ImageOps
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from unet_model_builder import u_net_model 
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import preprocessing

data_path = 'annotated_data'
seed = 42
np.random.seed = seed

IMG_COUNT = 16
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1                                                
epochs = 1000

#Read the dataset
scales, masks = preprocessing.read_data_set(data_path, 16, IMG_HEIGHT, IMG_WIDTH, 'label')
#Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(scales, masks, test_size=0.25, random_state=2)

#Augment the training dataset
augmented_x_train, augmented_y_train = preprocessing.augment_data_set(X_train, X_test)

#Encode the labels to categorical
categorical_y_train = to_categorical(augmented_y_train, dtype ='ubyte')
Y_test = to_categorical(Y_test, dtype='ubyte')

num_of_classes = 5
num_of_unet_layers = 5
start_filter_size = 16

#Build a u-net model models
input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

transfered_unregularized = u_net_model(input_size, num_of_classes).construct_model(num_of_unet_layers,start_filter_size)

#Files path
reg_backup_path = 'chkps/transfere_regularised/parameters.ckp'
unreg_backup_path = 'chkps/transfere_unregularised/parameters.ckp'
reg_test_result_path = 'test_results/transfere_regularised'
unreg_test_result_path = 'test_results/transfere_unregularised'
synthetic_backup_path = 'chkps/synthetic/parameters.ckp'

#Created callbaks
reg_callback = tf.keras.callbacks.ModelCheckpoint(filepath=reg_backup_path,
                                                 save_weights_only=True,
                                                 save_best_model = True,
                                                 verbose=1)
unreg_callback = tf.keras.callbacks.ModelCheckpoint(filepath=unreg_backup_path,
                                                 save_weights_only=True,
                                                 save_best_model = True,
                                                 verbose=1)
 
#Transfer the weights from the synthetic model and train on the real data                                               
transfered_unregularized.load_weights(synthetic_backup_path)

                                                 

# Train the model
transfered_unregularized.fit(augmented_x_train, categorical_y_train, batch_size=10, 
                    validation_split = 0.25,
                    verbose=1 ,callbacks=unreg_callback, epochs=epochs)

# Train the regularized model                    
transfered_regularized = u_net_model(input_size, num_of_classes).construct_regularized_model(num_of_unet_layers,start_filter_size)                    
transfered_regularized.load_weights(synthetic_backup_path)                    
transfered_regularized.fit(X_train1, categorical_y_train, batch_size=10, 
                    validation_split = 0.25,
                    verbose=1 ,callbacks=reg_callback, epochs=epochs)
                    
                    
#Show the results
print('\n Unregularized Classification Report\n')
predy = np.argmax(transfered_regularized.predict(X_test), axis=3).flatten()
truy = np.argmax(Y_test, axis=3).flatten()
print(classification_report(truy, predy, target_names=['Focus', 'Summer', 'River', 'Winter', 'Background']))
print(confusion_matrix(truy, predy))


