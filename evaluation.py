import tensorflow as tf
import os
import random
import numpy as np
from PIL import Image, ImageOps
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from unet_model_builder import u_net_model 

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



scales, masks = preprocessing.read_data_set(data_path, 16, IMG_HEIGHT, IMG_WIDTH, 'label')
X_train, X_test, Y_train, Y_test = train_test_split(scales, masks, test_size=0.25, random_state=2)
Y_test = to_categorical(Y_test, dtype='ubyte')

num_of_classes = 5
num_of_unet_layers = 5
start_filter_size = 16
input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)




transfered_reg_backup_path = 'chkps/transfere_regularised/parameters.ckp'
transfered_unreg_backup_path = 'chkps/transfere_unregularised/parameters.ckp'
from_scratch_reg_backup_path = 'chkps/from_scratch_regularized/parameters.ckp'
from_scratch_unreg_backup_path = 'chkps/from_scratch_not_regularized/from_scratch_not_regularized'

reg_test_result_path = 'test_results/transfere_regularised'
unreg_test_result_path = 'test_results/transfere_unregularised'
synthetic_backup_path = 'chkps/synthetic/parameters.ckp'

#Transfer the weights from the synthetic model and train on the real data                                               


transfered_unregularized_model = u_net_model(input_size, num_of_classes).construct_model(num_of_unet_layers,start_filter_size)
transfered_unregularized_model.load_weights(transfered_unreg_backup_path)

transfered_regularized_model = u_net_model(input_size, num_of_classes).construct_regularized_model(num_of_unet_layers,start_filter_size)     
transfered_regularized_model.load_weights(transfered_reg_backup_path)

from_scratch_unregularized_model = u_net_model(input_size, num_of_classes).construct_model(num_of_unet_layers,start_filter_size)
from_scratch_unregularized_model.load_weights(from_scratch_unreg_backup_path)

from_scratch_regularized_model = u_net_model(input_size, num_of_classes).construct_regularized_model(num_of_unet_layers,start_filter_size)  
from_scratch_regularized_model.load_weights(from_scratch_reg_backup_path)

synthetic_model = u_net_model(input_size, num_of_classes).construct_regularized_model(num_of_unet_layers,start_filter_size)                                             
synthetic_model.load_weights(synthetic_backup_path)                                                 



from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix
from PIL import Image, ImageOps
test_result_path= 'test_results/'
def save_image(img, name):
    im = Image.fromarray(np.uint8(img *63))
    path = test_result_path  + name + ".png"
    im.save(path)   
def save_test_result(truy, predy, model_name):
    for i in range(truy.shape[0]):
        save_image(truy[i], "mask_" + str(i) ) 
        save_image(predy[i], "mask_" + str(i) + model_name + "pred")
                 
def show_report(model, model_name):
    print('\n Model report:\n')
    print(model.evaluate(X_test, Y_test))
    predy = np.argmax(model.predict(X_test), axis=3)
    truy = np.argmax(Y_test, axis=3)
    save_test_result(truy, predy,model_name)
    predyf = np.argmax(model.predict(X_test), axis=3).flatten()
    truyf = np.argmax(Y_test, axis=3).flatten()
    print("Classificxation report:")
    print(classification_report(truyf, predyf, target_names=['Focus', 'Summer', 'River', 'Winter', 'Background']))
    print("Confusion Matrix: ")
    print(confusion_matrix(truyf, predyf))
    print("\n \n \n __________________________________________________________________ \n \n")
                
five_models = [transfered_unregularized_model, transfered_regularized_model, from_scratch_unregularized_model, from_scratch_regularized_model, synthetic_model]
five_models_names = ['transfered_unregularized_model', 'transfered_regularized_model', 'from_scratch_unregularized_model', 'from_scratch_regularized_model', 'synthetic_model']


for i in range(len(five_models_names)):
    print("Model: " + five_models_names[i])
    print("\n \n \n __________________________________________________________________ \n \n")
    show_report(five_models[i], five_models_names[i])


