import numpy as np
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import tqdm
project_root = os.getcwd()
print(project_root)
os.chdir(project_root)
sys.path.append(project_root)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 禁用 GPU

from DataTool_Code.LoadData import *
from Defence_Method.Moe_Gen.Moe_Gen_Def import Moe_Gen_OW_Def,Moe_Gen_CW_Def
from Defence_Method.Alert.Alert_def import Alert_def_cw,Alert_def_ow
from Defence_Method.AWA.awa_def import Eva_awa_CW,Eva_awa_OW


from WF_Model.test_ClassfyModel import evaluate_model
from WF_Model.CFModel_Loder import Load_Classfy_Model






def No_def(cw_data,cw_label,ow_data,ow_label):
    print("No Def EXECUTING...")
    No_def_data = np.concatenate([cw_data, ow_data], axis=0)
    No_def_label = np.concatenate([cw_label, ow_label], axis=0)
    print('No_def data:',No_def_data.shape)
    return No_def_data,No_def_label
def DFDall(MutiNum):
    print("DFD EXECUTING...")
    cw_data,_=Load_DFD_data_cw_100()
    ow_data,_=Load_DFD_data_ow_100to400()
    ow_data=ow_data[:len(cw_data)*MutiNum]
    dfd_data = np.concatenate([cw_data, ow_data], axis=0)
    print('dfd_data:',dfd_data.shape)
    return  dfd_data

def WalkieTalkie_def(MutiNum):
    print("WalkieTalkie EXECUTING...")
    cw_data,_=Load_WalkitTalkie_data_cw_100()
    ow_data,_=Load_WalkitTalkie_data_ow_100to400()
    ow_data=ow_data[:len(cw_data)*MutiNum]
    WalkieTalkie_data = np.concatenate([cw_data, ow_data], axis=0)
    print('WalkieTalkie_data:',WalkieTalkie_data.shape)
    return  WalkieTalkie_data


def Awa_def(cw_data,cw_label,ow_data,ow_label,CF_Model_Name):
    print("Awa EXECUTING...")
    adv_ow=Eva_awa_OW(ow_data ,CF_Model_Name)
    
    adv_cw=Eva_awa_CW(cw_data, cw_label ,CF_Model_Name)
    Alert_data = np.concatenate([adv_cw, adv_ow], axis=0)
    print('Alert data:',Alert_data.shape)
    return Alert_data

def Alert_def(cw_data,cw_label,ow_data,ow_label,CF_Model_Name):
    print("Alert EXECUTING...")
    adv_cw=Alert_def_cw(cw_data, cw_label ,CF_Model_Name)
    adv_ow=Alert_def_ow(ow_data, ow_label ,CF_Model_Name)
    Alert_data = np.concatenate([adv_cw, adv_ow], axis=0)
    print('Alert data:',Alert_data.shape)
    return Alert_data

def Moe_Gen_def(cw_data,cw_label,ow_data,ow_label,CF_Model_Name):
    print("Moe_Gen EXECUTING...")
    adv_cw=Moe_Gen_CW_Def(cw_data, cw_label ,CF_Model_Name,'AWF100')
    adv_ow=Moe_Gen_OW_Def(ow_data ,ow_label,CF_Model_Name,'AWF100')
    Moe_Gen_data = np.concatenate([adv_cw, adv_ow], axis=0)
    print('Moe_Gen data:',Moe_Gen_data.shape)
    return Moe_Gen_data



def model_predict(model, X_test,Type_num,example_num):
    predicted_labels=np.empty((0,Type_num))
    for i in tqdm.tqdm(range(Type_num)):
        predicted_label = model(X_test[i*example_num*(1+MutiNum):(i+1)*example_num*(1+MutiNum)])
        predicted_labels=np.concatenate((predicted_labels,predicted_label),axis=0)
    print('predicted_labels.shape:',predicted_labels.shape)
    return np.max(predicted_labels, axis=1)

def Draw_PT():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(8, 4.5))  

    # Draw random reference line (positive example ratio)
    pos_ratio = np.mean(No_def_label)
    plt.axhline(y=pos_ratio, color='k', linestyle='--', label=f'Random (POS-RATIO={pos_ratio:.2f})')

    # Prediction results of each model
    models = {
        'Origin': predict_labels_Nodef,
        'DFD': predict_labels_DFD,
        'Walkie-Talkie': predict_labels_WalkieTalkie,
        'AWA': predict_labels_AWA,
        'ALERT': predict_labels_Alert,
        'Moe-Gen': predict_labels_Moe_Gen,           
    }

    # Define color palette
    color_map = plt.cm.get_cmap('tab10')

    # Draw P-T curve for each model
    for idx, (name, preds) in enumerate(models.items()):
        precision, _, thresholds = precision_recall_curve(No_def_label, preds)
        plt.plot(thresholds, precision[:-1], 
                 label=name, linewidth=3, color=color_map(idx))

    # Set figure properties
    plt.grid()
    plt.xlabel('Classification Threshold', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'WF-model: {CF_Model_Name}', fontsize=16)
    plt.legend(loc='upper left', fontsize=9,ncol=2)
    plt.subplots_adjust(left=0.2, bottom=0.15)

    # Save figure
    Save_Path='/home/xuke/zpc/Code/Moe_Gen/Open_World/PT_Save'
    os.makedirs(Save_Path,exist_ok=True)
    plt.savefig(f'{Save_Path}/{CF_Model_Name}_in_{DataSet_name}_1to{MutiNum}_PT.pdf',
                dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Success saved P-T comparison plot for {DataSet_name} dataset.")

def Drew_ROC():
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(8, 4.5))
    plt.plot([0, 1], [0, 1], 'k--')

    # Define model names and corresponding prediction results
    models = {
        'Origin': predict_labels_Nodef,
        'DFD': predict_labels_DFD,
        'Walkie-Talkie': predict_labels_WalkieTalkie,
        'AWA': predict_labels_AWA,
        'ALERT': predict_labels_Alert,
        'Moe-Gen': predict_labels_Moe_Gen,          
    }

    # Define color palette (tab10 has 10 standard colors)
    color_map = plt.cm.get_cmap('tab10')

    # Iterate through models for plotting
    for idx, (name, preds) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(No_def_label, preds)
        aucval = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            label=f'{name} (AUC={round(aucval, 4)})',
            linewidth=3,
            color=color_map(idx)
        )

    plt.grid()
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'WF-model: {CF_Model_Name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=9,ncol=2)
    plt.subplots_adjust(left=0.2, bottom=0.15)


    Save_Path='Moe_Gen/Open_World/ROC_Save'
    os.makedirs(Save_Path,exist_ok=True)
    plt.savefig(f'{Save_Path}/{CF_Model_Name}_in_{DataSet_name}_1to{MutiNum}_ROC.pdf',
                dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Success saved ROC comparison plot for {DataSet_name} dataset.")

def Draw_PR():
    if MutiNum==1:
        legend_Loc='lower right'
    elif MutiNum==4:
        legend_Loc='upper right'
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc

    
    plt.figure(figsize=(8, 4.5))

    
    pos_ratio = np.mean(No_def_label)
    plt.axhline(y=pos_ratio, color='k', linestyle='--',
                label=f'Random (POS-RATIO={pos_ratio:.2f})')

    
    models = {
        'Origin': predict_labels_Nodef,
        'DFD': predict_labels_DFD,
        'Walkie-Talkie': predict_labels_WalkieTalkie,
        'AWA': predict_labels_AWA,
        'ALERT': predict_labels_Alert,
        'Moe-Gen': predict_labels_Moe_Gen,           
    }

    
    color_map = plt.cm.get_cmap('tab10')

    
    for idx, (name, preds) in enumerate(models.items()):
        precision, recall, _ = precision_recall_curve(No_def_label, preds)
        pr_auc = auc(recall, precision)
        plt.plot(
            recall,
            precision,
            label=f'{name} (PR-AUC={pr_auc:.4f})',
            linewidth=3,
            color=color_map(idx)
        )

    plt.grid()
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'WF-model: {CF_Model_Name}', fontsize=16)
    plt.legend(loc=legend_Loc, fontsize=9,ncol=2)
    plt.subplots_adjust(left=0.2, bottom=0.15)
    Save_Path='Moe_Gen/Open_World/PR_Save'
    os.makedirs(Save_Path,exist_ok=True)
    
    plt.savefig(
        f'{Save_Path}/{CF_Model_Name}_in_{DataSet_name}_1to{MutiNum}_PR.pdf',
        dpi=800, bbox_inches='tight', pad_inches=0
    )
    plt.close()

    print(f"Success saved PR comparison plot for {DataSet_name} dataset.")





MutiNum_list={4,1}   
for MutiNum in MutiNum_list:
    Save_Path=f'OpenWorld/SaveFile/1to{MutiNum}'

    DataSet_name="AWF100"
    cw_x,cw_y=Load_AWF100_cw_test100()
    ow_x,ow_y=Load_AWF_OW_data()
    ow_x=ow_x[:len(cw_x)*MutiNum]
    ow_y=ow_y[:len(cw_y)*MutiNum]
    print(f"==== In {DataSet_name} Evaluation=====")
    Class_type=100
    example_num=100  

    dfd_data=DFDall(MutiNum)
    WalkieTalkie_data=WalkieTalkie_def(MutiNum)
    No_def_data,No_def_label=No_def(cw_x,cw_y,ow_x,ow_y)
    No_def_label=[1 if i<Class_type else 0 for i in No_def_label]
    No_def_label=np.array(No_def_label)
    print(No_def_label.shape)
    CF_Model_List=['VarCNN','DF','AWF']

    for CF_Model_Name in CF_Model_List:
        print(f"Test {CF_Model_Name} model in {DataSet_name} dataset")

        # Load Defense Data
        AWA_data=Awa_def(cw_x,cw_y,ow_x,ow_y,CF_Model_Name)
        Alert_data=Alert_def(cw_x,cw_y,ow_x,ow_y,CF_Model_Name)
        Moe_Gen_data=Moe_Gen_def(cw_x,cw_y,ow_x,ow_y,CF_Model_Name)
        # AWA_data=dfd_data
        # Alert_data=dfd_data
        # Moe_Gen_data=dfd_data

        #Load Classfy Model
        model_ = Load_Classfy_Model(CF_Model_Name,'AWF100')

        #Predict Defense Data
        predict_labels_Nodef=model_predict(model_,No_def_data,Class_type,example_num)
        
        predict_labels_DFD=model_predict(model_,dfd_data,Class_type,example_num)
        predict_labels_WalkieTalkie=model_predict(model_,WalkieTalkie_data,Class_type,example_num)
        
        predict_labels_Alert=model_predict(model_,Alert_data,Class_type,example_num)
        predict_labels_AWA=model_predict(model_,AWA_data,Class_type,example_num)
        predict_labels_Moe_Gen=model_predict(model_,Moe_Gen_data,Class_type,example_num)
        Draw_PR()
        Draw_PT()
        Drew_ROC()  










