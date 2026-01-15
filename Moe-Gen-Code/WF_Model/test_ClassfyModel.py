import os
import sys
project_root =os.getcwd()
os.chdir(project_root)
sys.path.append(project_root)

import  tensorflow as tf
from tqdm import tqdm

from Tool_Code import Metrics
from DataTool_Code import LoadData
from WF_Model.CFModel_Loder import Load_Classfy_Model


def evaluate_model(model, test_data, test_labels,batch_size):

    predicted_labels_list = []
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch_data = test_data[i:i + batch_size]
        # print(batch_data.shape)
        test_outputs= model(batch_data)
        predicted=tf.argmax(test_outputs,1)
        predicted_labels_list.append(predicted.cpu())
    
    predicted_labels= tf.concat(predicted_labels_list, axis=0).numpy()
    F1, TPR, FPR, overall_ACC,per_class_acc = Metrics.get_metrics(test_labels, predicted_labels)
    return F1, TPR, FPR, overall_ACC,per_class_acc


if __name__ == '__main__':
    

    batch_size = 128
    burst_length = 2000
    DataSet='AWF100'                    # AWF100, AWF200, AWF500,AWF900
    model_name = 'AWF'            # VarCNN,DF,AWF
    print(f"<===  Evaluate {model_name} in {DataSet} dataset  ===>")
    # test_data,test_label = LoadData.Load_Data(DataSet, 'test')
    test_data,test_label = LoadData.Load_AWF100_Front_cw()

    classification_model = Load_Classfy_Model(model_name,DataSet,burst_length)
    

    F1,_,_,overall_ACC,per_class_acc=evaluate_model(classification_model, test_data, test_label,batch_size)
    avg_F1=F1.mean()
    print(f"F1: {avg_F1}")
    print(f" Overall ACC: {overall_ACC}")

    print("per class acc: ", per_class_acc)