import os
import sys
import random
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)
import tensorflow as tf
import numpy as np

def set_seed(seed=77):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)
from DataTool_Code.LoadData import Load_Data
from Defence_Method.Moe_Gen.Gen_Model.Moe_Gen import MoE_Gen 
from Tool_Code.Tool import  *
from WF_Model.test_ClassfyModel import evaluate_model
from WF_Model.CFModel_Loder import  Load_Classfy_Model
def Load_Moe_Gen(gen_path,noise_size, NB_CLASSES, flow_size, num_experts,embed_dim):
    
    generator= MoE_Gen(noise_size, NB_CLASSES, flow_size, num_experts,embed_dim)
    dummy_batch_size = 1
    dummy_label = tf.zeros((dummy_batch_size,), dtype=tf.int32)  # [1] shape
    generator(dummy_batch_size,dummy_label)
    generator.load_weights(gen_path)
    return generator

def test_generator(noise_gen, Classfy_model, test_data, test_labels):
    print("Evaluate Generator for ")
    len_test_data = len(test_data)

    adv_test_data = np.zeros_like(test_data) 
    max_noise = 0.0  
    total_samples = 0

    for i in range(0, len_test_data, batch_size):
        left = i
        right = min(i + batch_size, len_test_data)
        batch_test_data = test_data[left:right]
        batch_test_labels = test_labels[left:right]
        cur_test_batchsize = tf.shape(batch_test_data)[0]
        
       
        test_batchnoise = noise_gen(batch_size=cur_test_batchsize, batch_label=batch_test_labels, training=False)
        test_batchnoise = tf.round(test_batchnoise)

       
        test_batchsign = custom_sign(batch_test_data)
        test_batchnoise_withsign = test_batchnoise * test_batchsign
        adv_test_batchdata = batch_test_data + test_batchnoise_withsign
       
        one_noise=test_batchnoise_withsign[42]
        one_noise=tf.reshape(one_noise,[-1])
        no_zero_index=tf.where(one_noise!=0)
        # print(no_zero_index)
        non_zero_values = tf.gather_nd(one_noise, no_zero_index)
        # print(non_zero_values)
        ans_index=tf.cast(no_zero_index,tf.int32)
        ans_value=tf.cast(non_zero_values,tf.int32)
        ans_value=ans_value[:,tf.newaxis]
        ans = tf.concat([ans_index+1, ans_value], axis=1)
        # print(ans)
       

        
        adv_test_data[left:right] = adv_test_batchdata.numpy()

        
        abs_noise = tf.abs(test_batchnoise_withsign)
        flat_noise = tf.reshape(abs_noise, [cur_test_batchsize, -1])
        max_vals = tf.reduce_max(flat_noise, axis=1)  # Evalute the max noise for each sample
        max_noise = max(max_noise, tf.reduce_max(max_vals).numpy())  # Overall max noise



    # Evaluate 
    F1, TPR, FPR, overall_ACC, per_class_acc = evaluate_model(Classfy_model, adv_test_data, test_labels,batch_size=batch_size)
    overhead = tf.reduce_sum(tf.abs(adv_test_data) - tf.abs(test_data)) / tf.reduce_sum(tf.abs(test_data))

    AvgF1=np.mean(F1)
    print(f"Overall ACC: {overall_ACC}")
    print("F1:", AvgF1)
    print("Bandwidth: {:.2%}".format(overhead))
    print("Max value in perturbation: ", max_noise)
    return overall_ACC,overhead,AvgF1

def getModelPath(DataSet,CFmodel):

        print('temp path')
        if(CFmodel=='AWF'):
            ModelPath=f"Defence_Method/Moe_Gen/File_Save/GenSave/AWF100/AWF/epoch_try.h5"
        elif(CFmodel=='DF'):
            ModelPath=f"Defence_Method/Moe_Gen/File_Save/GenSave/AWF100/DF/epoch_try.h5"
        elif(CFmodel=='VarCNN'):
            ModelPath=f"Defence_Method/Moe_Gen/File_Save/GenSave/AWF100/VarCNN/epoch_try.h5"
        else:
            raise ValueError(f"Unknown CFmodel: {CFmodel}")
        return ModelPath
if __name__ == "__main__":
    noise_size=2000
    flow_size=2000
    batch_size=512
   
    DataSet_name_list=['AWF100']
    CF_Model_List=["DF","AWF","VarCNN"]

    # DataSet_name_list=['AWF100','AWF200','AWF500','AWF900']
    # CF_Model_List=["DF","AWF","VarCNN"]
    for DataSet_name in DataSet_name_list:
        test_data,test_label=Load_Data(DataSet_name,'test')
        
        if DataSet_name=="AWF100":
            NB_CLASSES = 100
            num_experts=4
            embed_dim=50
        elif DataSet_name=="AWF200":
            NB_CLASSES = 200
            num_experts=6
            embed_dim=100
        elif DataSet_name=="AWF500":
            NB_CLASSES = 500
            num_experts=12
            embed_dim=100
        elif DataSet_name=="AWF900":
            NB_CLASSES = 900
            num_experts=13
            embed_dim=100
        for CFModel in CF_Model_List:
            print(f"==> Defence {CFModel} in {DataSet_name}")
            classfy_model=Load_Classfy_Model(CFModel,DataSet_name)
            gen_path=getModelPath(DataSet_name,CFModel)
            Moe_Gen=Load_Moe_Gen(gen_path,noise_size, NB_CLASSES, flow_size, num_experts,embed_dim)
            test_generator(Moe_Gen,classfy_model,test_data,test_label)
