# Generate universal perturbations

import os
import sys

project_root =os.getcwd()
os.chdir(project_root)
print(project_root)
os.chdir(project_root)
sys.path.append(project_root)

import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import GAN_utility as ganModel
from WF_Model.CFModel_Loder import Load_Classfy_Model
from DataTool_Code.LoadData import Load_Data
from Defence_Method.Alert.get_target import get_target_adv
"""Data Loading"""

VERBOSE = 1
model_list=['DF','AWF','VarCNN']
# model_list=['VarCNN']

# model_name : DF,AWF,VarCNN
dataset='AWF100'
flow_size=2000
if dataset == 'AWF100':
    flowtype = 100
time_record=[]
for model_name in model_list:
    print(f'Defince {model_name}==>')
    model=Load_Classfy_Model(model_name,dataset,flow_size)
    train_mode="defence_train"
    start_time = time.time()  
    head_threshold=0.22


    print(f"in {dataset} for{model_name} do  { train_mode}")
    # Output the second last layer of the model
    if model_name=='DF':
        FE=tf.keras.models.Model(model.input,model.get_layer('fc2').output)
    elif model_name=='AWF':
        FE=tf.keras.models.Model(model.input,model.get_layer('block3_pool').output)
    elif model_name=='VarCNN':
        FE=tf.keras.models.Model(model.input,model.get_layer('average_pool').output)

    def adjust_WF_data(x = None,perturbation = None):
        """
        Superimpose generated perturbations with original data to generate adversarial samples
        :param x: Clean samples
        :param perturbation: Perturbation amount
        :return:
        """
        perturbation = tf.expand_dims(perturbation, 2)
        perturbation = perturbation * 1.0
        advData = x + perturbation * tf.sign(x)
        return advData

    def get_class_samples(X, Y, C):
        """
        Return data of specified class C from given dataset (X,Y)
        :param X: Data traces, np.darray type
        :param Y: Corresponding labels, default is one-hot format data, np.darray type
        :param C: Specified class
        :return:
        """
        # y = np.argmax(Y, axis=1) # Convert labels from one-hot to numerical type ==> label values are numerical
        ind = np.where(Y == C)
        return X[ind], Y[ind]

    # Loss function construction
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Optimizer construction
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def loss_1(adjusted_generated_one, adjusted_generated_two):
        ####### Cosine similarity
        total = 0
        for i in range(0, adjusted_generated_one.shape[0]):
            cos_sim = tf.reduce_sum(adjusted_generated_one[i] * adjusted_generated_two[i]) / (tf.norm(adjusted_generated_one[i],ord=2) * tf.norm(adjusted_generated_two[i],ord=2))
            total = total + cos_sim
        loss = total / batch_size
        loss = 1 - loss
        return loss

    def overHead_loss(X_ori, X_adv, overHead_thresh=0.22):
        overHead = tf.reduce_sum(tf.abs(X_adv)-tf.abs(X_ori))/tf.reduce_sum(tf.abs(X_ori))
        return tf.maximum(0.0, overHead-overHead_thresh)


    """Training iterations"""
    batch_size=512
    g_iteration = 40  #
    
    data_length = 2000

    gen_loss = []
    total_loss = []
    logit_losse = []
    head_one = []

    classNum = []
    for i in range(0, flowtype):
        classNum.append(i)


    # data, labels = load_data("./dataset/Burst_Closed World/burst_tor_200w_2500tr_test.npz")
    adv_data,adv_labels = Load_Data(dataset,'adv')
    test_data,test_labels = Load_Data(dataset,'test')

   
    for label in range(0, flowtype):
        print(f'label:{label}/{flowtype}')
        adv_data_X, adv_data_Y = get_class_samples(adv_data, adv_labels, label)
        test_data_X, test_data_Y = get_class_samples(test_data, test_labels, label)
  

        print(adv_data_X.shape, adv_data_Y.shape)
        print("##############  ori_label", label)


        generator = ganModel.generator_model_5()  

        targetLabel=get_target_adv(dataset,model_name,train_mode,label)
        print("slected target label:", targetLabel)


        targe_data_X, targe_data_Y = get_class_samples(adv_data, adv_labels, targetLabel)

        indices = np.random.randint(targe_data_X.shape[0], size=batch_size)
        x_targe_batch = targe_data_X[indices]

        for iter in range(g_iteration):
            print("label :%d, iter :%d" % (label, iter))

            with tf.GradientTape() as G1_tape:
                indices = np.random.randint(adv_data_X.shape[0], size=batch_size)
                x_train_batch = adv_data_X[indices]
                y_train_batch = adv_data_Y[indices]  #

                random_noise_one = np.random.normal(size=[batch_size, data_length])
                adv_distribution = generator(random_noise_one, training=True)   


                # 生成器的损失
                generated_one = adjust_WF_data(x_train_batch, adv_distribution)
                head_loss = overHead_loss(x_train_batch, generated_one,head_threshold)
                pre_one = model(generated_one)  
                gen_logit_loss= tf.reduce_mean(tf.maximum(pre_one[:, label], 0))

                origin_target_loss = loss_1(FE(generated_one), FE(x_targe_batch))

                if iter % 10 == 0:
                    print("##############gen_logit_loss", gen_logit_loss.numpy())
                    print("##############origin_target_loss", origin_target_loss.numpy())

               
                loss = gen_logit_loss + origin_target_loss + head_loss
                total_loss.append(loss.numpy())
                gen_loss.append(origin_target_loss.numpy())
                logit_losse.append(gen_logit_loss.numpy())
                head_one.append(head_loss.numpy())


            
            gradient_gen = G1_tape.gradient(loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))

        base_save_path = f"Defence_Method/Alert/File_Save/Epoch_40/Gen_BatchSize{batch_size}/{dataset}/{model_name}/{train_mode}"
        if not os.path.exists(base_save_path):
            os.makedirs(base_save_path, exist_ok=True)  # Use makedirs to support recursive creation
            print(f"Created directory: {base_save_path}")

        gen_save_path = f'{base_save_path}/label_{label}'
        if not os.path.exists(gen_save_path):
            os.makedirs(gen_save_path)
        generator.save_weights(gen_save_path+ '/'+f'ori_{label}_to{targetLabel}' + ".h5")

        plt.figure()
        plt.plot(head_one, label="overhead")
        plt.plot(total_loss, label="total loss")
        plt.plot(logit_losse, label="logit_loss")
        plt.plot(gen_loss, label="origin_target_loss")
        # plt.plot(dis_loss, label="dis_loss")
        plt.title(" Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(gen_save_path+'/' + str(label) + "_loss.png")
        plt.close()
        head_one = []
        total_loss = []
        logit_losse = []
        gen_loss = []

    end_time = time.time()


    elapsed_time = end_time - start_time
    print(f"Cur TIME: {elapsed_time:.4f} 秒")
    time_record.append(elapsed_time)
print('All Time cost:',time_record)