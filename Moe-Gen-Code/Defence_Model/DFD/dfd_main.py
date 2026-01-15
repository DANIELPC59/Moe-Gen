import os
import sys
project_path=os.getcwd()
print("Project path:", project_path)
sys.path.append(project_path)
import time
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from DataTool_Code.LoadData import Load_Data
def dfd_up(ori_burst, rate):
    """
    上行方向插入扰动数据包
    rate: float，插入比例
    """
    burst = tf.identity(ori_burst) # 复制一个新 tensor，防止原地修改
    burst_len = burst.shape[1]
    # 上行突发的所有索引，2,4,6...
    idx = tf.range(2, burst_len, delta=2) 
    # 取出所有要被修改的 burst[:, idx]
    
    update = tf.gather(burst, idx - 2, axis=1) * rate
    update = tf.round(update) # 保证为整数
    # 只在偶数位置（上行）插入
    # 把 update 累加到 burst 的对应位置
    mask = tf.one_hot(idx, burst_len, on_value=1.0, off_value=0.0) # (num_idx, burst_len)
    update_pad = update[:, :, tf.newaxis] * mask # (batch, num_idx, burst_len)
    update_sum = tf.reduce_sum(update_pad, axis=1) # (batch, burst_len)
    # burst = burst + update_sum
    
    zero_mask = tf.cast(burst != 0, burst.dtype)  # 非0位置为1，0位置为0
    update_sum = update_sum * zero_mask
    
    # 累加回原 burst
    burst = burst + update_sum
    return burst

def dfd_all(ori_burst, rate):
    """
    在所有位置插入扰动（奇数位置保留负号）
    ori_burst: tf.Tensor, shape (batch_size, burst_len)
    rate: float
    """
    burst = tf.identity(ori_burst)
    burst_len = burst.shape[1]

    # 需要插入的位置（从第2个开始，因为要用 idx-2）
    idx = tf.range(2, burst_len)

    # 从 idx-2 的位置取数据作为扰动来源
    update = tf.gather(burst, idx - 2, axis=1) * rate
    update = tf.round(update)

    # one-hot 掩码
    mask = tf.one_hot(idx, burst_len, on_value=1.0, off_value=0.0)
    update_pad = update[:, :, tf.newaxis] * mask
    update_sum = tf.reduce_sum(update_pad, axis=1)

    # zero padding 不插入
    zero_mask = tf.cast(burst != 0, burst.dtype)
    update_sum = update_sum * zero_mask

    # 累加回 burst
    burst = burst + update_sum
    return burst


# # 用法示例
dataset_name='AWF100'
data_type='test'
data_x,label_y=Load_Data(dataset_name,data_type)
data_x = np.squeeze(data_x, axis=-1) # 维度从(datanum,burst_len,1) => (datanum,burst)
print('data_shape',data_x.shape)
rate=0.75
# 自动用 GPU（如果有）
Save_path="Zpc/DataSet/AWF/AWF100/DFD/CloseWorld"

batch_size = 64

results = []
norm_original_total = 0
norm_result_total = 0
begin_time=time.time()
with tf.device('/GPU:0'):
     for start in tqdm(range(0, len(data_x), batch_size), desc="Processing batches"):
        end = start + batch_size
        batch_data = data_x[start:end]
        print('batch_data:',batch_data.shape)
        tf_burst = tf.constant(batch_data, dtype=tf.float32)
        result = dfd_all(tf_burst, rate)

        # 累加结果和带宽统计
        results.append(result.numpy())
        norm_original_total += tf.reduce_sum(tf.abs(tf_burst)).numpy()
        norm_result_total += tf.reduce_sum(tf.abs(result)).numpy()

# 拼接所有 batch 的结果
final_result = np.vstack(results)
end_time=time.time()

# 计算带宽开销比值
bandwidth = norm_result_total / norm_original_total
print('带宽开销比值：', bandwidth)
print('result.shape',final_result.shape)
# 保存结果
os.makedirs(Save_path, exist_ok=True)
np.savez(os.path.join(Save_path, f'{dataset_name}_CW_DFD_all.npz'), data=final_result, labels=label_y)
print('处理完成，保存路径：', Save_path)
print('时间开销:',end_time-begin_time)


# 测试数据
# burst = np.array([
#     [ 15,  -7,  11,  -4,   0,  -2,   6,   0,   1, -12],
#     [  0, -10,   0, -15,  15, -12,   3,  -7,   0,   0],
#     [ 14,  -9,   0, -15,  12, -15,   0, -18,   2, -10],
#     [ 18,  -2,   4, -16,  14,   0,   0, -16,  15, -13],
#     [  0,  -1,   0,  -1,   0,   0,  17,  -3,   5,  -9],
#     [  9,  -2,   0,  -5,   8,  -3,   5,  -3,  14,  -9],
#     [ 10, -19,   0,  -4,   7, -11,  13,   0,   2,  -6],
#     [ 12,   0,  11,  -7,   1, -13,   7,   0,   5, -10],
#     [ 15,   0,   9, -17,  12,  -2,   5, -17,   2,  -5],
#     [  1,  -2,   6,  -4,   0, -17,   5,  -6,  16,  -9]
# ], dtype=np.int32)
# print(burst)
# tf_burst = tf.constant(burst, dtype=tf.float32)
# result=dfd_all(tf_burst,1)
# print(result)