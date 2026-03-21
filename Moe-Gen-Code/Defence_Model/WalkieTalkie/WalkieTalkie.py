import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def label_accuracy(pred, target):
    """Computes the accuracy of model predictions matching the target labels"""
    batch_size = target.shape[0]
    correct = np.sum(pred == target)
    accuracy = correct / batch_size * 100.0
    return accuracy



class WalkieTalkie:

    def __init__(self):
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def generate_walkie_talkie_samples(self, test_bursts, test_y, train_bursts, train_y):
        """
        Walkie-Talkie

        Input:
        test_bursts: Tensor, shape=[batch_size, seq_len], 
        test_y: Tensor, shape=[batch_size], 
        train_bursts: Tensor, shape=[train_size, seq_len],  train_burst play the role of  decoy
        train_y: Tensor, shape=[train_size], 
        Return：
        perturbed_bursts: Tensor, shape=[batch_size, seq_len]
        """
     


        batch_size = tf.shape(test_bursts)[0]
        train_size = tf.shape(train_bursts)[0]

      
        perturbed_bursts = []
        virtual_packet_counts = []

        print("Begin generate perturbed x")
        for i in tf.range(batch_size):
            print(f'{i}/{batch_size}')
            current_label = test_y[i]

            different_class_indices = tf.where(tf.not_equal(train_y, current_label))
            different_class_indices = tf.reshape(different_class_indices, [-1])

            random_idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(different_class_indices)[0], dtype=tf.int32)
            decoy_idx = different_class_indices[random_idx]

            test_burst = test_bursts[i]
            decoy_burst = train_bursts[decoy_idx]

            test_abs = tf.abs(test_burst)
            decoy_abs = tf.abs(decoy_burst)
            max_abs = tf.maximum(test_abs, decoy_abs)
            test_sign = tf.sign(test_burst)

            supersequence = test_sign * max_abs
            perturbed_bursts.append(supersequence)
        
        perturbed_bursts = tf.stack(perturbed_bursts)
        

        return perturbed_bursts


  
    def eval_performance(self, eval_x, eval_y, train_x, train_y):
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataset.setXY(eval_x, eval_y)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(ori_dataset.get_dataset())
        print('Performance before attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')

        x = eval_x
        assert len(x.shape) == 2, f'x.shape={x.shape}, not [batch, Feat].'
        print('x.shape:', x.shape)
        perturbed_x = self.generate_walkie_talkie_samples(
            x, eval_y, train_x, train_y, max_burst_len=4000, max_dir_len=5000)

        pert_dataset = DynamicDataset(perturbed_x, eval_y)
        pert_dataset.setX(perturbed_x)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(pert_dataset.get_dataset())
        print('Performance after attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')


    def eval_performance_for_batch_baseline(self, eval_x, eval_y, train_x, train_y):
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataset.setXY(eval_x, eval_y)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(ori_dataset.get_dataset())
        print('Performance before attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')

        x = eval_x
        assert len(x.shape) == 2, f'x.shape={x.shape}, not [batch, Feat].'
        print('x.shape:', x.shape)
        if hasattr(self, 'perturbed_x') and self.perturbed_x.shape == eval_x.shape:
            print('Reuse previous perturbed x') # reuse perturbed_x
        else:
            self.perturbed_x = self.generate_walkie_talkie_samples(
                x, eval_y, train_x, train_y, max_burst_len=4000, max_dir_len=5000)

        pert_dataset = DynamicDataset(self.perturbed_x, eval_y)
        pert_dataset.setX(self.perturbed_x)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(pert_dataset.get_dataset())
        print('Performance after attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')



