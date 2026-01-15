import os
import sys

# Add project path to sys.path
project_path = os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

import numpy as np



from DataTool_Code import LoadData
from Defence_Method.AWA.awa_class import AWA_Class
from Defence_Method.AWA import test_awa


def Get_keycup(DataSet,CFmodel):
    npz_path = f'Defence_Method/AWA/File_Save/Gen_Save/{DataSet}/{CFmodel}'
    files = [f for f in os.listdir(npz_path) if f.endswith(".npz")]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {path}, but found {len(files)}")
    file_name = files[0]
    full_path = os.path.join(npz_path, file_name)
    data = np.load(full_path)
    print("Keys in npz:", data.files)
    key1 = data['key1']
    key2 = data['key2']
    return key1,key2



# Load model and perturb specified class data
def gen_adv_for_class(data_x, data_y, label, gen_path):
    
    inds = np.where(data_y == label)[0]
    if len(inds) == 0:
        return np.empty((0, data_x.shape[1])), np.empty((0,))
    x_cls = data_x[inds]
    y_cls = data_y[inds]
    
    noise = np.random.normal(size=x_cls.shape)
    
    model = generator_model((x_cls.shape[1],))
    model.load_weights(gen_path)
    model.trainable = False
    
    pert = model(noise, training=False).numpy()
    adv_x = x_cls + pert * np.sign(x_cls)
    adv_x = np.round(adv_x)
    return adv_x, y_cls,pert



if __name__ == "__main__":
    print("eva awa")
    dataset_name = "AWF100"
    CF_Model="AWF"
    root_dir = f"Defence_Method/AWA/File_Save/Gen_Save/{dataset_name}/{CF_Model}"

    data_x, data_y = LoadData.Load_Data(dataset_name, 'test')
    print(data_y.dtype)
    print(data_x.dtype)


    adv_data=test_awa.Eva_awa_CW(data_x,data_y,'DF')
    print(np.max(np.abs(adv_data-data_x)))
    