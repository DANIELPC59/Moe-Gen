
import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
# Use only CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

print("TensorFlow version:", tf.__version__)


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import random
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.utils import to_categorical

from DataTool_Code.LoadData import *
from WF_Model.Model.DF_TF import DFNet
from WF_Model.Model.AWF_TF import AWFNet
from WF_Model.Model.Var_CNN_TF import VarCNN
from WF_Model.test_ClassfyModel import evaluate_model
from WF_Model.CFModel_Loder import Load_Classfy_Model
random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




DataSet_name="AWF900"
#  DataSet:   DF,AWF100,AWF103,AWF200,AWF500,AWF900 ;
CF_Model_name="VarCNN"
# CF_Model:   DF ; AWF ; VarCNN
description = f"Training and evaluating {CF_Model_name} model for closed-world scenario on {DataSet_name} non-defended dataset"
print (description)
# Training the DF model
NB_EPOCH = 30   # Number of training epoch
print ("Number of Epoch: ", NB_EPOCH)
BATCH_SIZE = 128 # Batch size
VERBOSE = 1 # Output display mode
FlOW_LENGTH = 2000 # Packet sequence length
OPTIMIZER = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer


if DataSet_name=="DF":
    NB_CLASSES = 95 # number of outputs = number of classes
elif DataSet_name=="AWF100":
    NB_CLASSES = 100
elif DataSet_name=="AWF103":
    NB_CLASSES = 103
elif DataSet_name=="AWF200":
    NB_CLASSES = 200
elif DataSet_name=="AWF500":
    NB_CLASSES = 500
elif DataSet_name=="AWF900":
    NB_CLASSES = 900
INPUT_SHAPE = (FlOW_LENGTH,1)
# Data: shuffled and split between train and valid sets
print ("Loading and preparing data for training, and evaluating the model")

X_train,y_train=Load_Data(DataSet_name,'train')
if DataSet_name=="DF":
    X_valid,y_valid=Load_Data(DataSet_name,'valid')
elif DataSet_name.startswith('AWF'):
    X_valid,y_valid=Load_Data(DataSet_name,'test')

y_train = to_categorical(y_train, NB_CLASSES)
y_valid= to_categorical(y_valid, NB_CLASSES)

print(X_train.shape[0], 'train samples')

print(X_valid.shape[0], 'valid samples')

# Building and training model
print (f"Building and training {CF_Model_name} model")
if CF_Model_name=="DF":
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
elif CF_Model_name=="AWF":
    model = AWFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
elif CF_Model_name=="VarCNN":
    model = VarCNN.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    
    
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
	metrics=["accuracy"])
print ("Model compiled")
model_save_path =f"WF_Model/ModelSave/DataSet_{DataSet_name}/{CF_Model_name}/{CF_Model_name}_model_in_{DataSet_name}_DataSet.h5"


checkpoint = ModelCheckpoint(model_save_path, 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=True,
                             mode='max')
print(X_train.shape, y_train.shape)
# Start training
history = model.fit(X_train, y_train,
		batch_size=BATCH_SIZE, epochs=NB_EPOCH,
		verbose=VERBOSE, validation_data=(X_valid, y_valid),
        callbacks=[checkpoint])


# Start evaluating model with validing data
score_valid = model.evaluate(X_valid, y_valid, verbose=VERBOSE)
print("validing accuracy:", score_valid[1])


model.load_weights(model_save_path)
print("Loaded best model weights from checkpoint.")

score_valid = model.evaluate(X_valid, y_valid, verbose=VERBOSE)
print("Best saved model validation accuracy:", score_valid[1])
# classification_model = Load_Classfy_Model(CF_Model_name,DataSet_name,FlOW_LENGTH)
# F1, TPR, FPR, overall_ACC,per_class_acc=evaluate_model(model,X_valid,y_valid,batch_size=BATCH_SIZE)
#
# print(f"Saved model accuracy on valid set: {overall_ACC}")

# os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
# model.save(model_save_path)
# print(f"Model saved to {model_save_path}")