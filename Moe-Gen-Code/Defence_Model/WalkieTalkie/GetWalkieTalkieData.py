import tensorflow as tf
import numpy as np
import argparse
import time

import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

from Tool_Code.Tool import *
from DataTool_Code.LoadData import Load_Data



def running_WalkieTalkie(test_x,test_y,decoy_x,decoy_y):
    from WalkieTalkie import WalkieTalkie
    # generate perturbations
    start_time = time.time()
    
    test_x = tf.convert_to_tensor(test_x, dtype=tf.int32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
    decoy_x = tf.convert_to_tensor(decoy_x, dtype=tf.int32)
    decoy_y = tf.convert_to_tensor(decoy_y, dtype=tf.int32)
    
    wt = WalkieTalkie()
    perturbed_x_Burst=wt.generate_walkie_talkie_samples(test_x, test_y, decoy_x, decoy_y)  
    

   
    print(f"Total running time: {time.time() - start_time:.2f} seconds")
    return perturbed_x_Burst




def addperturbation():
    
    save_directory='/root/shared-storage/file_save/Defence_Method/WalkieTalkie/CloseWorld'
    for data_name in data_name_list:
        train_data, train_labels = Load_Data(data_name,'adv')
        test_data, test_labels = Load_Data(data_name,'test')
       
        train_data=np.squeeze(train_data, axis=-1)
        test_data=np.squeeze(test_data, axis=-1)
        print(test_data.shape)
        # a=input('stop')
        perturbed_x=running_WalkieTalkie(test_data,test_labels,train_data,train_labels)
        np.savez(os.path.join(save_directory, f'{data_name}_CW_WalkieTalkie.npz'), data=perturbed_x, labels=test_labels)
        print('Success Get WalkieTalkie Data')
## main
data_name_list=['AWF100']

addperturbation()

