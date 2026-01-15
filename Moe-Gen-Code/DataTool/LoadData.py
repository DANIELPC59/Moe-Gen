import numpy as np

Base_Path='/Website_Fingerprinting/DataSet'
def get_tar_class_data(datas,labels,target_c):

    index=np.where(labels==target_c)[0]
    target_data=datas[index]
    target_label=labels[index]
    return target_data,target_label

def getData_Path(DataSet,datatype):
    if DataSet=='DF':
        data_path=Base_Path+f"/DF/close-world/Nodef_AWA/{datatype}.npz"
    elif DataSet.startswith('AWF'):
        data_path=Base_Path+f"/AWF/{DataSet}/NoDef_Burst/{datatype}.npz"
    else:
        raise ValueError(f"Unknown DataSet: {DataSet}")
    return data_path

# Load Original Data
def Load_Data(DataSet,datatype):
    datapath=getData_Path(DataSet,datatype)
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['label']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')

# Load AWF100 Data for OpenWorld Testing(100*100=10000)
# Every class has 100 samples
def Load_AWF100_cw_test100():
    datapath=Base_Path+'/AWF/AWF100/NoDef_Burst/AWF100_CW_test_100.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['label']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')

# Load AWF100 OpenWorld Data for OpenWorld Testing(100*100*4=40000)
def Load_AWF_OW_data():
    datapath=Base_Path+'/AWF/OpenWorld/AWF100_OW_100to400_burst.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['label']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]    
    return x_data.astype('float32'), y_data.astype('float32')

# ToolFunc for WalkitTalkie 
def Load_AWF_youer_data():
    datapath=Base_Path+'/OpenWorld/AWF100_OW_1000.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['label']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]    
    return x_data.astype('float32'), y_data.astype('float32')

# Load WalkitTalkie Data
def Load_WalkitTalkie_data_cw():
    datapath=Base_Path+'/AWF/AWF100/WalkieTalkie/AWF100_CW_WalkieTalkie.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['labels']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')
# Load WalkitTalkie Data (EVERY CLASS 100 SAMPLES)
def Load_WalkitTalkie_data_cw_100():
    datapath=Base_Path+'/AWF/AWF100/WalkieTalkie/AWF100_CW_WalkieTalkie_100.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['labels']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')
# Load WalkitTalkie OPEN-WORLD Data for OpenWorld Testing(100*100*4=40000)
def Load_WalkitTalkie_data_ow_100to400():
    datapath=Base_Path+'/AWF/AWF100/WalkieTalkie/AWF100_OW_WalkieTalkie_100to400.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['labels']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')


# load DFD Data
def Load_DFD_data_cw():
    datapath=Base_Path+'/AWF/AWF100/DFD/CloseWorld/AWF100_CW_DFD_all.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['labels']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')
# Load DFD Data (EVERY CLASS 100 SAMPLES)
def Load_DFD_data_cw_100():
    datapath=Base_Path+'/AWF/AWF100/DFD/CloseWorld/AWF_CW_DFD_100.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['label']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')
# Load DFD OPEN-WORLD Data for OpenWorld Testing(100*100*4=40000)
def Load_DFD_data_ow_100to400():
    datapath=Base_Path+'/AWF/AWF100/DFD/CloseWorld/AWF_OW_DFD_100to400.npz'
    data_combine=np.load(datapath)
    x_data=data_combine['data']
    y_data=data_combine['label']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')

if __name__=="__main__":
    data,label=Load_Data('AWF100','adv')
    print(data.shape)
    print(label.shape)
