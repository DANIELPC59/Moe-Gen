# This Page of Code is used to provide Interface for Alert Defence 
import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

from tensorflow.python.keras.layers import Dropout
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from DataTool_Code.LoadData import Load_Data
from WF_Model.CFModel_Loder import Load_Classfy_Model
from WF_Model.test_ClassfyModel import evaluate_model


# Perturbation insertion function
def adjust_WF_data(x = None,perturbation = None):
    """
    add perturbation to original data
    :param x: clean sample
    :param perturbation: perturbation
    :return:
    """
    perturbation = tf.expand_dims(perturbation, 2)
    perturbation = perturbation * 1.0
    advData = x + perturbation * tf.sign(x)
    return tf.round(advData)  



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

def get_gen_path(label,CF_Model_Name,base_save_path):
    path=base_save_path+f'/{CF_Model_Name}/defence_train/label_{label}'
    files = [f for f in os.listdir(path) if f.endswith(".h5")]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {path}, but found {len(files)}")
    file_name = files[0]
    full_path = os.path.join(path, file_name)
    return full_path


def Alert_def_cw(data_x, label_y ,CF_Model_Name):
    """
    Inputs:
        data_x: np.ndarray, shape=(N, data_length,1)
        label_y: np.ndarray, shape=(N,)
        CF_Model_Name: such as "DF"
    Outputs:
        adv_data: np.ndarray, shape=(N, data_length,1) 
    """
    # Basic Configuration
    data_length=2000
    base_save_path = "Defence_Method/Alert/File_Save/Epoch_30/Gen_BatchSize512/AWF100" 
    flow_type=100  
    
    print('==>Alert Begin CW Defence')
    np.random.seed(42)
    tf.random.set_seed(42)
    
    adv_data = np.zeros_like(data_x) 
    # for cls in tqdm(range(flow_type)):
    for cls in (range(flow_type)):
        # print(f'{cls}/{flow_type}')
        cls_index = np.where(label_y == cls)[0]
        if len(cls_index) == 0:
            print(f'no data for label {cls}')
            continue
            
        x_cls = data_x[cls_index]
        gen_path=get_gen_path(cls,CF_Model_Name,base_save_path)
        generator = generator_model()  # gen1
        generator.build(input_shape=(None, data_length))
        generator.load_weights(gen_path)  # Load weights
        generator.trainable = False

        # Generate perturbation
        batch_len=len(x_cls)
        random_noise_train = np.random.normal(size=[batch_len, data_length])
        adv_noise = generator(random_noise_train, training=False)

        # add perturbation
        adjusted_train = adjust_WF_data(x_cls, adv_noise)
        
        # Put data back to original position
        adv_data[cls_index] = adjusted_train

    return adv_data

def Alert_def_ow(data_x, label_y ,CF_Model_Name):
    """
    inputs:
        data_x: np.ndarray, shape=(N, data_length,1)
        label_y: np.ndarray, shape=(N,)
        CF_Model_Name: such as "DF"
    outputs:
        adv_data: np.ndarray, shape=(N, data_length,1) 
    """
    # Basic Configuration
    data_length=2000
    base_save_path = "Defence_Method/Alert/File_Save/Epoch_30/Gen_BatchSize512/AWF100" 
    flow_type=100  
    batch_size=20
    import random
    print('==>Alert Begin OW Defence')
    np.random.seed(42)
    tf.random.set_seed(42)
  
    adv_data = np.zeros_like(data_x) 
    total_len=len(data_x)
    for start in tqdm(range(0, total_len, batch_size)):
        end = min(start + batch_size, total_len)
        batch_x = data_x[start:end]
        
        rand_cls = random.randint(0, flow_type-1)
        
        gen_path=get_gen_path(rand_cls,CF_Model_Name,base_save_path)
        generator = generator_model()  # gen1
        generator.build(input_shape=(None, data_length))
        generator.load_weights(gen_path)  # Load weights
        generator.trainable = False

        # Generate perturbation
        batch_len=len(batch_x)
        random_noise_train = np.random.normal(size=[batch_len, data_length])
        adv_noise = generator(random_noise_train, training=False)

        # add perturbation
        adjusted_train = adjust_WF_data(batch_x, adv_noise)

        # Put data back to original position
        adv_data[start:end] = adjusted_train

    return adv_data

if __name__ == '__main__':
    BURST_LEN=2000
    # Model_List=['DF','AWF','VarCNN']
    Model_List=['AWF']
    DataSet_Name='AWF100'
    test_data, test_labels = Load_Data(DataSet_Name, "test")
    for CF_Model in Model_List:
        adv_data=Alert_def_cw(test_data,test_labels,CF_Model)
        model_=Load_Classfy_Model(CF_Model,DataSet_Name,BURST_LEN)
        F1, TPR, FPR, overall_ACC,per_class_acc = evaluate_model(model_, adv_data, test_labels, batch_size=128)
        # calculate bandwidth overhead
        numerator = tf.reduce_sum(tf.abs(adv_data) - tf.abs(test_data))
        denominator = tf.reduce_sum(tf.abs(test_data)) + 1e-8  # avoid division by zero
        overhead=(numerator / denominator)
        #F1 score
        AvgF1=np.mean(F1)
        print(f"==> Alert Defence {CF_Model} in {DataSet_Name}")
        print(f"Overall ACC: {overall_ACC}")
        print("F1:", AvgF1)
        print("Bandwidth: {:.2%}".format(overhead))
        print(f"per ACC: {per_class_acc}")
        
