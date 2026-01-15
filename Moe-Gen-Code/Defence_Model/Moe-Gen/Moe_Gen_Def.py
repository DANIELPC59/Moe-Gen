import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from DataTool_Code.LoadData import Load_Data
from Defence_Method.Moe_Gen.Gen_Model.Moe_Alert_Temperature import MoE_Gen 
from Tool_Code.Tool import  *
from WF_Model.test_ClassfyModel import evaluate_model
from WF_Model.CFModel_Loder import  Load_Classfy_Model
def Load_Moe_Gen(gen_path,noise_size, NB_CLASSES, flow_size, num_experts,embed_dim):
    # 测试基于Moe_Gen结构的生成器
    generator= MoE_Gen(noise_size, NB_CLASSES, flow_size, num_experts,embed_dim)
    dummy_batch_size = 1
    dummy_label = tf.zeros((dummy_batch_size,), dtype=tf.int32)  # [1] shape
    generator(dummy_batch_size,dummy_label)
    generator.load_weights(gen_path)
    return generator

def Moe_Gen_CW_Def(test_data, test_label, cf_model_name, dataset_name, NB_CLASSES=100,noise_size=2000, flow_size=2000, batch_size=512):
    """
    输入:测试数据、标签、分类器模型名、数据集名
    输出:扰动后的样本（np.ndarray，shape与test_data一致）
    """
    print('==>Moe_Gen Begin CW Defence')
    # 1. 获取参数
    params = get_generator_params(dataset_name)
    NB_CLASSES = params["NB_CLASSES"]
    num_experts = params["num_experts"]
    embed_dim = params["embed_dim"]
    # gen_lr = params["gen_lr"]  # 如果后续要用

    # 2. 加载生成器
    gen_path = getModelPath(dataset_name, cf_model_name)
    generator = Load_Moe_Gen(gen_path, noise_size, NB_CLASSES, flow_size, num_experts, embed_dim)
    
    # 3. 批量生成扰动
    adv_data = np.zeros_like(test_data)
    len_test_data = len(test_data)

    for i in tqdm(range(0, len_test_data, batch_size)):
        left = i
        right = min(i + batch_size, len_test_data)
        batch_data = test_data[left:right]
        batch_label = test_label[left:right]
        cur_batch_size = batch_data.shape[0]
        # 生成扰动
        batch_noise = generator(batch_size=cur_batch_size, batch_label=batch_label, training=False)
        batch_noise = tf.round(batch_noise)
        batch_sign = custom_sign(batch_data)
        batch_noise_withsign = batch_noise * batch_sign
        adv_batch = batch_data + batch_noise_withsign
        adv_data[left:right] = adv_batch.numpy()
    return adv_data
def Moe_Gen_OW_Def(test_data, test_label, cf_model_name, dataset_name, NB_CLASSES=100,noise_size=2000, flow_size=2000, batch_size=512):
    """
    输入:测试数据、标签、分类器模型名、数据集名
    输出:扰动后的样本（np.ndarray，shape与test_data一致）
    """
    # 1. 获取参数
    params = get_generator_params(dataset_name)
    NB_CLASSES = params["NB_CLASSES"]
    num_experts = params["num_experts"]
    embed_dim = params["embed_dim"]
    # gen_lr = params["gen_lr"]  # 如果后续要用
    print('==>Moe_Gen Begin OW Defence')
    # 2. 加载生成器
    gen_path = getModelPath(dataset_name, cf_model_name)
    generator = Load_Moe_Gen(gen_path, noise_size, NB_CLASSES, flow_size, num_experts, embed_dim)
    
    # 3. 批量生成扰动
    adv_data = np.zeros_like(test_data)
    len_test_data = len(test_data)

    for i in tqdm(range(0, len_test_data, batch_size)):
        left = i
        right = min(i + batch_size, len_test_data)
        batch_data = test_data[left:right]
        # 随机替换为CW类别
        batch_label = test_label[left:right].copy()
        # print('ori_batch_label',batch_label)
        batch_label = np.random.randint(0, NB_CLASSES, size=batch_label.shape[0])
        # print('random_batch_label',batch_label)
        cur_batch_size = batch_data.shape[0]
        # 生成扰动
        batch_noise = generator(batch_size=cur_batch_size, batch_label=batch_label, training=False)
        batch_noise = tf.round(batch_noise)
        # print(batch_noise)
        batch_sign = custom_sign(batch_data)
        batch_noise_withsign = batch_noise * batch_sign
        adv_batch = batch_data + batch_noise_withsign
        adv_data[left:right] = adv_batch.numpy()
    return adv_data

def get_Noise_by_labels(
    labels,
    cf_model_name,
    dataset_name,
    noise_size=2000,
    batch_size=512
):
    """
    inputs: labels (shape: [N])
    ouputs: noise (shape: [N, noise_size])
    """

    # 1. load generator params
    params = get_generator_params(dataset_name)
    NB_CLASSES = params["NB_CLASSES"]
    num_experts = params["num_experts"]
    embed_dim = params["embed_dim"]

    # 2. load generator
    gen_path = getModelPath(dataset_name, cf_model_name)
    generator = Load_Moe_Gen(
        gen_path,
        noise_size,
        NB_CLASSES,
        flow_size=noise_size,   
        num_experts=num_experts,
        embed_dim=embed_dim
    )

    labels = np.asarray(labels)
    num_samples = len(labels)

    # 3. gen noise
    all_noise = []

    for i in range(0, num_samples, batch_size):
        batch_labels = labels[i:i + batch_size]
        cur_bs = len(batch_labels)

        batch_noise = generator(
            batch_size=cur_bs,
            batch_label=batch_labels,
            training=False
        )

        batch_noise = tf.round(batch_noise)
        all_noise.append(batch_noise.numpy())

    return np.concatenate(all_noise, axis=0)
def getModelPath(DataSet,CFmodel):
        ModelPath=f"Defence_Method/Moe_Gen/File_Save/GenSave/{DataSet}/{CFmodel}/epoch_try.h5"
        return ModelPath
    
def get_generator_params(dataset_name):
    # match params by dataset_name
    if dataset_name == "AWF100":
        return dict(NB_CLASSES=100, num_experts=4, embed_dim=50)
    elif dataset_name == "AWF200":
        return dict(NB_CLASSES=200, num_experts=6, embed_dim=100)
    elif dataset_name == "AWF500":
        return dict(NB_CLASSES=500, num_experts=12, embed_dim=100)
    elif dataset_name == "AWF900":
        return dict(NB_CLASSES=900, num_experts=13, embed_dim=100)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
if __name__ == "__main__":
    noise_size=2000
    flow_size=2000
    batch_size=512
    DataSet_name_list=['AWF100']
    CF_Model_List=["DF","AWF","VarCNN"]

    for DataSet_name in DataSet_name_list:
        test_data,test_label=Load_Data(DataSet_name,'test')
        for CFModel in CF_Model_List:
            print(f"==> Defence {CFModel} in {DataSet_name}")
            classfy_model=Load_Classfy_Model(CFModel,DataSet_name)
            gen_path=getModelPath(DataSet_name,CFModel)
            Moe_Gen=Load_Moe_Gen(gen_path,noise_size, NB_CLASSES, flow_size, num_experts,embed_dim)
            test_generator(Moe_Gen,classfy_model,test_data,test_label)
