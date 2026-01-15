import os
from WF_Model.Model.DF_TF import DFNet
from WF_Model.Model.AWF_TF import  AWFNet
from WF_Model.Model.Var_CNN_TF import VarCNN
def Load_Classfy_Model(model_name,dataset,flow_size=2000):
    """
    Load the Classify model from the specified path.
    Args:
        model_path (str): The path to the model file.
    Returns:
        model: The loaded model.
    """
    INPUT_SHAPE = (flow_size, 1)
    model_path = get_Model_Path(model_name,dataset,flow_size=flow_size)
    if(dataset=='DF'):
        flow_type=95
    elif(dataset=='AWF100'):
        flow_type=100
    elif(dataset=='AWF103'):
        flow_type=103
    elif(dataset=='AWF200'):
        flow_type=200
    elif dataset == "AWF500":
        flow_type = 500
    elif dataset == "AWF900":
        flow_type = 900
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 加载识别模型
    if model_name == 'DF':
        Classfy_Model=DFNet.build(INPUT_SHAPE,flow_type)
    elif model_name == 'AWF':
        Classfy_Model=AWFNet.build(INPUT_SHAPE,flow_type)
    elif model_name == 'VarCNN':
        Classfy_Model=VarCNN.build(INPUT_SHAPE,flow_type)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    
    Classfy_Model.load_weights(model_path)
    
    Classfy_Model.trainable = False
    
    return Classfy_Model

def get_Model_Path(model_name,DataSet_name,flow_size=2000):
    Base_Path = ''
    Model_Path = Base_Path+f'/Moe_Gen/WF_Model/ModelSave/DataSet_{DataSet_name}/{model_name}'

    if not os.path.exists(Model_Path):
        print(f' Path: {Model_Path} not exist ')
        return None
    
    files = os.listdir(Model_Path)

    
    if len(files) == 0:
        print(f'No {model_name} model found in flow_size={flow_size} ')
        return None
    file_name = files[0]
    
    model_path = os.path.join(Model_Path, file_name)
    return model_path



if __name__ == '__main__':
    Load_Classfy_Model('DF','DF',flow_size=2000)
                   