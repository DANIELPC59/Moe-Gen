import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)
import numpy as np

import tensorflow as tf

from tensorflow.python.keras.layers import Dropout
from DataTool_Code.LoadData import Load_Data
from WF_Model.CFModel_Loder import Load_Classfy_Model
from WF_Model.test_ClassfyModel import evaluate_model
"""Data Loading"""
def get_class_samples(X, Y, C):
    """
    Return data of specified class C from given dataset (X,Y)
    :param X: Data traces, np.darray type
    :param Y: Corresponding labels, default is one-hot format data, np.darray type
    :param C: Specified class
    :return:
    """
    # y = np.argmax(Y, axis=1) # Convert labels from one-hot to numerical type
    ind = np.where(Y== C)
    return X[ind], Y[ind]

# Perturbation insertion function
def adjust_WF_data(x = None,perturbation = None):
    """
    Superimpose generated perturbation with original data to generate adversarial samples
    :param x: Clean samples
    :param perturbation: Perturbation amount
    :return:
    """
    perturbation = tf.expand_dims(perturbation, 2)
    perturbation = perturbation * 1.0
    advData = x + perturbation * tf.sign(x)
    return tf.round(advData)  # Round decimals for rounding


# generator model
def generator_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(2000))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(2000))
    model.add(tf.keras.layers.Activation('relu'))
    return model
def get_gen_path(base_save_path,label,CF_Model_Name):
    path=base_save_path+f'/{CF_Model_Name}/defence_train/label_{label}'
    files = [f for f in os.listdir(path) if f.endswith(".h5")]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {path}, but found {len(files)}")
    file_name = files[0]
    full_path = os.path.join(path, file_name)
    return full_path
"""Pretrained Model Loading"""
base_save_path = "Defence_Method/Alert/File_Save/Gen_BatchSize64/AWF100" # trained g

train_X = np.array([])
train_Y = np.array([])

test_X = np.array([])
test_Y = np.array([])


# batch_size_train = 1200
# batch_size_test = 300
batch_size=512
data_length = 2000
dataset='AWF100'
data, labels = Load_Data(dataset,'test')
CF_Model_List=['AWF','DF','VarCNN']

if dataset == 'AWF100':
    flow_type=100
for CF_Model_Name in CF_Model_List:
    
    for label in range(0, flow_type):
        # print(f"{label}/{flow_type}")
        print("##################################orilabel", label)
        # Training dataset
        test_X, test_Y = get_class_samples(data, labels, label)

        print("test data", test_X.shape, test_Y.shape)
        batch_len=len(test_X)
        random_noise_train = np.random.normal(size=[batch_len, data_length])


        generator = generator_model()  # Generator 1
        generator.build(input_shape=(None, data_length))
        gen_path=get_gen_path(base_save_path,label,CF_Model_Name)
        generator.load_weights(gen_path)  # Load weights
        generator.trainable = False

        adv_train_noise = generator(random_noise_train, training=False)


        adjusted_train = adjust_WF_data(test_X, adv_train_noise)
        # adjusted_train = tf.expand_dims(adjusted_train_tmp, 1)

        if label == 0:
            train_X = adjusted_train
            train_Y = test_Y
        else:
            train_X = np.concatenate([train_X, adjusted_train], axis=0)
            train_Y = np.concatenate([train_Y, test_Y], axis=0)
     # Model evaluation
    Classfy_model=Load_Classfy_Model(CF_Model_Name,dataset)
    F1, TPR, FPR, overall_ACC, per_class_acc = evaluate_model(Classfy_model, train_X, train_Y,batch_size=batch_size)
    overhead = tf.reduce_sum(tf.abs(train_X) - tf.abs(data)) / tf.reduce_sum(tf.abs(data))
    # Output results
    AvgF1=np.mean(F1)
    print(f"==> Alert Defence {CF_Model_Name} in {dataset}")
    print(f"Overall ACC: {overall_ACC}")
    print("F1:", AvgF1)
    print("Bandwidth: {:.2%}".format(overhead))
    # return overall_ACC,overhead,AvgF1
    

