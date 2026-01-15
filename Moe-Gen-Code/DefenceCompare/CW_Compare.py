import os
import sys
import random
import numpy as np
import tensorflow as tf
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
random_seed=56  # 42 , 77 ,0
set_seed(random_seed)

project_root = os.getcwd()
print(project_root)
os.chdir(project_root)
sys.path.append(project_root)

from collections import defaultdict
from DataTool_Code.LoadData import Load_AWF100_cw_test100,Load_AWF_OW_data,Load_DFD_data_cw_100,Load_DFD_data_ow_100to400,Load_WalkitTalkie_data_cw_100,Load_WalkitTalkie_data_ow_100to400
from DataTool_Code.LoadData import *
from Defence_Method.Moe_Gen.Moe_Gen_Def import Moe_Gen_OW_Def,Moe_Gen_CW_Def
from Defence_Method.Alert.Alert_def import Alert_def_cw,Alert_def_ow
from Defence_Method.AWA.awa_def import Eva_awa_CW,Eva_awa_OW


from WF_Model.test_ClassfyModel import evaluate_model
from WF_Model.CFModel_Loder import Load_Classfy_Model



import logging
import sys
import os

def get_logger(filename):
    # define logger
    logger = logging.getLogger("Experiment_logger")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename, mode='a', encoding='utf-8') # mode='a' 表示追加模式
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# init logger
log_file_name='DefenceCompare/log_01-10.log'
logger = get_logger(log_file_name)


def No_def(cw_data,cw_label):
    print("No Def EXECUTING...")
    print('No_def data:',cw_data.shape)
    return cw_data,cw_label

def DFDall(DataSetName):
    dfd_data,_ = Load_DFD_data_cw(DataSetName)
    print('dfd_data:',dfd_data.shape)
    return  dfd_data


def WalkieTalkie_def():
    WalkieTalkie_data,_=Load_WalkitTalkie_data_cw()
    print('WalkieTalkie_data.shape',WalkieTalkie_data.shape)
    return  WalkieTalkie_data

def Front_def():
    front_data,_=Load_AWF100_Front_cw()
    print('front_data.shape',front_data.shape)
    return  front_data


def Awa_def(cw_data,cw_label,CF_Model_Name):
    print("Awa EXECUTING...")
    adv_cw=Eva_awa_CW(cw_data, cw_label ,CF_Model_Name)
    print('Alert data:',adv_cw.shape)
    return adv_cw

def Alert_def(cw_data,cw_label,CF_Model_Name):
    print("Alert EXECUTING...")
    Alert_data=Alert_def_cw(cw_data, cw_label ,CF_Model_Name)
    print('Alert data:',Alert_data.shape)
    return Alert_data

def Moe_Gen_def(cw_data,cw_label,CF_Model_Name):
    print("Moe_Gen EXECUTING...")
    Moe_Gen_data=Moe_Gen_CW_Def(cw_data, cw_label ,CF_Model_Name,'AWF100')
    print('Moe_Gen data:',Moe_Gen_data.shape)
    return Moe_Gen_data

DataSet_name="AWF100"
cw_x,cw_y=Load_Data(DataSet_name,'test')

print(f"==== In {DataSet_name} Evaluation=====")
Class_type=100
batch_size=128  

dfd_data=DFDall(DataSet_name)
front_data=Front_def()
WalkieTalkie_data=WalkieTalkie_def()
No_def_data,No_def_label=No_def(cw_x,cw_y)
print('label.shape:',No_def_label.shape)

CF_Model_List=['VarCNN','DF','AWF']
targetModel_Model_List=['VarCNN','DF','AWF']

for CF_Model_Name in CF_Model_List:
    print(f"Test {CF_Model_Name} model in {DataSet_name} dataset")
    logger.info(f"Test {CF_Model_Name} model in {DataSet_name} dataset ,  RandomSeed={random_seed}")
    # Load perturb data
    AWA_data=Awa_def(cw_x,cw_y, CF_Model_Name)
    Alert_data=Alert_def(cw_x,cw_y, CF_Model_Name)
    Moe_Gen_data=Moe_Gen_def(cw_x,cw_y, CF_Model_Name)
    

    for targetModel in targetModel_Model_List:
        
        # Load target model
        model_ = Load_Classfy_Model(targetModel,'AWF100')

        # Evaluate model 
        F1_Nodef,_,_,Acc_Nodef,_=evaluate_model(model_,No_def_data,No_def_label,batch_size)
        F1_DFD,_,_,Acc_DFD,_= evaluate_model(model_,dfd_data,No_def_label,batch_size)
        F1_Alert,_,_,Acc_Alert,_= evaluate_model(model_,Alert_data,No_def_label,batch_size)
        F1_AWA,_,_,Acc_AWA,_= evaluate_model(model_,AWA_data,No_def_label,batch_size)
        F1_WalkieTalkie,_,_,Acc_WalkieTalkie,_= evaluate_model(model_,WalkieTalkie_data,No_def_label,batch_size)
        F1_Front,_,_,Acc_Front,_= evaluate_model(model_,front_data,No_def_label,batch_size)
        F1_Moe_Gen,_,_,Acc_Moe_Gen,_= evaluate_model(model_,Moe_Gen_data,No_def_label,batch_size)
        
        print(f"======>In {CF_Model_Name} Attack {targetModel}")
        print(f"No Defense    Acc: {Acc_Nodef:.4f}         AvgF1: {np.mean(F1_Nodef):.4f}")
        print(f"DFD           Acc: {Acc_DFD:.4f}           AvgF1: {np.mean(F1_DFD):.4f}")
        print(f"Alert         Acc: {Acc_Alert:.4f}         AvgF1: {np.mean(F1_Alert):.4f}")
        print(f"AWA           Acc: {Acc_AWA:.4f}           AvgF1: {np.mean(F1_AWA):.4f}")
        print(f"Walkie-Talkie Acc: {Acc_WalkieTalkie:.4f}  AvgF1: {np.mean(F1_WalkieTalkie):.4f}")
        print(f"Front         Acc: {Acc_Front:.4f}         AvgF1: {np.mean(F1_Front):.4f}")
        print(f"Moe_Gen       Acc: {Acc_Moe_Gen:.4f}       AvgF1: {np.mean(F1_Moe_Gen):.4f}")
        print("="*60) 
        
        logger.info(f"======>In {CF_Model_Name} Attack {targetModel}")
        logger.info(f"No Defense    Acc: {Acc_Nodef:.4f}         AvgF1: {np.mean(F1_Nodef):.4f}")
        logger.info(f"DFD           Acc: {Acc_DFD:.4f}           AvgF1: {np.mean(F1_DFD):.4f}")
        logger.info(f"Alert         Acc: {Acc_Alert:.4f}         AvgF1: {np.mean(F1_Alert):.4f}")
        logger.info(f"AWA           Acc: {Acc_AWA:.4f}           AvgF1: {np.mean(F1_AWA):.4f}")
        logger.info(f"Walkie-Talkie Acc: {Acc_WalkieTalkie:.4f}  AvgF1: {np.mean(F1_WalkieTalkie):.4f}")
        logger.info(f"Front         Acc: {Acc_Front:.4f}         AvgF1: {np.mean(F1_Front):.4f}")
        logger.info(f"Moe_Gen       Acc: {Acc_Moe_Gen:.4f}       AvgF1: {np.mean(F1_Moe_Gen):.4f}")
        logger.info("="*60) 
    