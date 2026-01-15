# 训练最终的Moe-Gen
import os
import sys
import random
import time
import logging
from os import mkdir
import tensorflow as tf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses

project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)


from Tool_Code.Tool import *
from WF_Model.CFModel_Loder import  Load_Classfy_Model
from WF_Model.test_ClassfyModel import evaluate_model
from DataTool_Code.LoadData import Load_Data
from Defence_Method.Moe_Gen.Gen_Model.Moe_Gen import MoE_Gen 

def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

flow_size = 2000
batch_size = 512
epoch_num = 40
noise_size = 2000
center_threshold=10  # threshold of center insertion
max_overhead=0.1  # threshold of max overhead

def train_generator(DataSet_name,CF_Model_name):
        logger.info(f'===>In {DataSet_name} Defence  {CF_Model_name}')
        set_random_seed(42)
        print("train_generator")
        print(f"<=== Defence {CF_Model_name} in {DataSet_name} dataset ===>")
        Classfy_Moel = Load_Classfy_Model(CF_Model_name, DataSet_name, flow_size)
        # Load_Data
        train_data, train_labels = Load_Data(DataSet_name, 'adv')
        valid_data, valid_labels = Load_Data(DataSet_name, 'test')
        print("train_data shape:", train_data.shape)
        noise_generator = MoE_Gen(noise_size, params['NB_CLASSES'], flow_size, params['num_experts'],params['embed_dim'])
        optimizer = optimizers.Adam(learning_rate=params['gen_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # Optimizer

        start_time = time.time()
        overhead_record = []
        acc_record = []
        loss_record=[]
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}")
           
            len_train_data = len(train_data)
            # shuffle data
            indices = np.random.permutation(len_train_data)
            data = train_data[indices]
            labels = train_labels[indices]
            expert_usage = np.zeros(params['num_experts'], dtype=np.float32)

            for i in range(0, len_train_data, batch_size):
                left = i
                right = min(i + batch_size, len_train_data)
                with tf.GradientTape() as tape:
                    
                    batch_data = data[left:right]
                    batch_labels = labels[left:right]
                    
                    cur_batch_size = len(batch_data)
                    noise, gate_weight = noise_generator(batch_size=cur_batch_size, batch_label=batch_labels, training=True)
 
                    expert_usage += np.sum(gate_weight.numpy(), axis=0).squeeze()
                    # no round in training (because of the gradient of round is 0)
                    batch_sign = custom_sign(batch_data)

                    noise_with_sign = noise * batch_sign
                    adv_data = batch_data + noise_with_sign
                     
                   
                    # cf_model loss
                    output = Classfy_Moel(adv_data)
                    l1_norm = tf.reduce_mean(tf.norm(noise, ord=1, axis=1))  
                    max_noise = tf.reduce_max(noise)  
                    Classfy_loss = losses.sparse_categorical_crossentropy(batch_labels, output)
                    Classfy_loss = params['class_weight'] - (Classfy_loss) * params['class_weight']
                    Classfy_loss = tf.reduce_mean(Classfy_loss)  
                    # center insertion penalty
                    center_punish = tf.where(noise > center_threshold, tf.exp(noise), 0)
                    average_max_10square = tf.reduce_mean(tf.reduce_sum(center_punish, axis=1))
                    square_loss = average_max_10square * params['norm_weight']
                    # bandwidth overhead
                    overhead = tf.reduce_sum(tf.abs(adv_data) - tf.abs(batch_data)) / tf.reduce_sum(tf.abs(batch_data))
                    overhead_loss=tf.maximum(0.0, overhead - max_overhead) * params['overhead_norm']
                    # total loss
                    loss = Classfy_loss + square_loss + overhead_loss
            
                grads = tape.gradient(loss, noise_generator.trainable_variables)

                optimizer.apply_gradients(zip(grads, noise_generator.trainable_variables))
                if i / batch_size % 100 == 0:
                    print(
                        f"  batch {i / batch_size}:  loss:{loss:.4g}  class_loss:{Classfy_loss:.4g}    square_loss:{square_loss:.4g}    overweight:{overhead_loss:.4g}  noise_norm:{l1_norm:.4g}   max_noise {max_noise:.4g}")
            print("each expert usage:", expert_usage / len_train_data)
            noise_generator.trainable = False
            overall_ACC, overhead = test_generator(noise_generator, Classfy_Moel, valid_data, valid_labels)
            acc_record.append(overall_ACC)
            overhead_record.append(overhead)
            loss_record.append(loss)
            noise_generator.trainable = True
            noise_generator.save_weights(Model_Save_path + f"/epoch_{epoch}_gen.h5")
            logger.info(f"Epoch:{epoch}, acc={overall_ACC},overhead={overhead}")
        
        end_time=time.time()
        print("===>In {DataSet_name} Defence  {CF_Model_name} 时间开销：",end_time-start_time)
        min_acc = np.min(np.array(acc_record))
        
        min_indices = np.where(np.array(acc_record)==min_acc)[0]
        overhead_ = np.array(overhead_record)[min_indices]
        logger.info(f"<min_acc: {min_acc}, and overhead: {overhead_}>")
        epochs = list(range(epoch_num))
        # draw training records
        plot_training_records(epochs, overhead_record, acc_record, loss_record)
        
        
def test_generator(noise_gen, Classfy_model, test_data, test_labels):
    print("Evaluate Generator")
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

        adv_test_data[left:right] = adv_test_batchdata.numpy()

        abs_noise = tf.abs(test_batchnoise_withsign)
        flat_noise = tf.reshape(abs_noise, [cur_test_batchsize, -1])
        max_vals = tf.reduce_max(flat_noise, axis=1)  
        max_noise = max(max_noise, tf.reduce_max(max_vals).numpy()) 
    # Evaluate 
    F1, TPR, FPR, overall_ACC, per_class_acc = evaluate_model(Classfy_model, adv_test_data, test_labels,batch_size=batch_size)
    overhead = tf.reduce_sum(tf.abs(adv_test_data) - tf.abs(test_data)) / tf.reduce_sum(tf.abs(test_data))
    print(f"Overall ACC: {overall_ACC}")
    print("Per-class ACC:", per_class_acc)
    print("Bandwidth: {:.2%}".format(overhead))
    print("Max value in perturbation: ", max_noise)
    return overall_ACC,overhead
def plot_training_records(epochs, overhead_record, acc_record, loss_record):
    """
    Draw Overhead & Accuracy and Loss curves
    """
    fig, (ax_main, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------- Left：Overhead & Accuracy ----------------
    ax1 = ax_main
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Overhead (%)', color='tab:blue')

 
    ax1.plot(epochs, np.array(overhead_record) * 100, 
             color='tab:blue', label='Overhead (%)', marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:red')
    ax2.plot(epochs, np.array(acc_record) * 100, 
             color='tab:red', label='Accuracy (%)', marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))


    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

 
    lns1 = ax1.get_lines()
    lns2 = ax2.get_lines()
    ax1.legend(lns1 + lns2, [l.get_label() for l in lns1 + lns2], loc='upper left')
    ax1.set_title('Overhead & Accuracy')

    # ---------------- Loss ----------------
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')

    def smooth(values, weight=0.8):
        smoothed = []
        last = values[0]
        for v in values:
            smoothed_val = last * weight + (1 - weight) * v
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    loss_smoothed = smooth(loss_record, weight=0.8)

    ax_loss.plot(epochs, loss_smoothed, color='tab:green', label='Loss', marker='s', linestyle='-')
    ax_loss.set_title('Loss Curve')
    ax_loss.legend()
    ax_loss.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # ---------------- save figure ----------------
    plt.tight_layout()
    plt.savefig(f"{figure_path}/recordFig.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {figure_path}/recordFig.svg")

def get_experiment_params(dataset_name, model_name):
    """
    Load Basic Dataset Parameters and Basic Model Parameters
    """
    # 1. Basic Dataset Parameters
    dataset_configs = {
        "AWF100": {"NB_CLASSES": 100, "num_experts": 4,  "embed_dim": 50,  "gen_lr": 0.001},
        "AWF200": {"NB_CLASSES": 200, "num_experts": 6,  "embed_dim": 100, "gen_lr": 0.0005},
        "AWF500": {"NB_CLASSES": 500, "num_experts": 12, "embed_dim": 100, "gen_lr": 0.0001},
        "AWF900": {"NB_CLASSES": 900, "num_experts": 13, "embed_dim": 100, "gen_lr": 0.00005},
    }

    # Load Basic Dataset Parameters
    params = dataset_configs.get(dataset_name, dataset_configs["AWF100"]).copy()

    # 2. Basic Model Parameters (Default values)
    model_defaults = {
        'DF':     {'norm_weight': 0.015, 'class_weight': 3.7, 'overhead_norm': 15},
        'AWF':    {'norm_weight': 0.015, 'class_weight': 2.0, 'overhead_norm': 35}, 
        'VarCNN': {'norm_weight': 0.015, 'class_weight': 2.0, 'overhead_norm': 35}, 
    }
    
    # Load Basic Model Parameters and Update to params
    params.update(model_defaults.get(model_name, {}))

    # 3. Special Cases Overrides
    # Handle AWF100 + AWF Special Case
    if model_name == 'AWF':
        if dataset_name == 'AWF100':
            params['overhead_norm'] = 25
            params['gen_lr'] *= 0.5
        elif dataset_name == 'AWF200':
            params['overhead_norm'] = 40

    # Handle VarCNN Special Cases
    if model_name == 'VarCNN':
        if dataset_name == 'AWF100':
            params['class_weight'] = 3.7
            params['overhead_norm'] = 15
        elif dataset_name == 'AWF200':
            params['class_weight'] = 10
            params['overhead_norm'] = 1
            params['gen_lr'] = 0.0001

    return params
if __name__=="__main__":
    # DataSet_List=['AWF100','AWF200','AWF500','AWF900']
    DataSet_List=['AWF200','AWF500','AWF900']
    
    # CFModel_List=['AWF','DF','VarCNN']
    CFModel_List=['VarCNN']
    for DataSet_name in DataSet_List:
        
        for CF_Model_name in CFModel_List:
            params=get_experiment_params(DataSet_name,CF_Model_name)
            Base_Save_path='Defence_Method/Moe_Gen/File_Save'
            Model_Save_path = Base_Save_path+f'/GenSave/{DataSet_name}/{CF_Model_name}'    
            figure_path=Base_Save_path+f'/Figure/{DataSet_name}/{CF_Model_name}'


            if not os.path.exists(Model_Save_path):
                os.makedirs(Model_Save_path, exist_ok=True)  
                print(f"new create : {Model_Save_path}")

            if not os.path.exists(figure_path):
                os.makedirs(figure_path, exist_ok=True)  
                print(f"new create : {figure_path}")
            logging.basicConfig(
                format='%(message)s',
                level=logging.INFO,
                filename=Base_Save_path+f'/Log/{CF_Model_name}/{DataSet_name}.log')
            logger = logging.getLogger(__name__)
            train_generator(DataSet_name,CF_Model_name)
            

