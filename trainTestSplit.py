import numpy as np
from scipy.io import loadmat
import os
import random
from scipy import io
# get train test split
keys = {'PaviaU':'paviaU_gt',
        'Salinas':'salinas_gt',
        'KSC':'KSC_gt',
        'Houston':'Houston2018_gt',
        'gf5': 'gf5_gt',
        'Xiongan': 'xiongan_gt'}
TRAIN_SIZE = [10]
RUN = 10


def sample_gt(gt, train_size, mode='fixed_withone'):
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)
        if mode == 'random':
            train_size = float(train_size) / 100  # dengbin:20181011

    if mode == 'random_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            train_len = int(np.ceil(train_size * len(X)))
            train_indices += random.sample(X, train_len)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0

    elif mode == 'fixed_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features

            train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


# 保存样本
def save_sample(train_gt, test_gt, dataset_name, sample_size, run):
    sample_dir = './trainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})


def load(dname):
    path = os.path.join(dname,'{}_gt.mat'.format(dname))
    dataset = loadmat(path)
    key = keys[dname]
    gt = dataset[key]
    # # 采样背景像素点
    # gt += 1
    return gt


def TrainTestSplit(datasetName):
    gt = load(datasetName)
    for size in TRAIN_SIZE:
        for r in range(RUN):
            train_gt, test_gt = sample_gt(gt, size)
            save_sample(train_gt, test_gt, datasetName, size, r)
    print('Finish split {}'.format(datasetName))


if __name__ == '__main__':
    dataseteName = ['Xiongan']
    for name in dataseteName:
        TrainTestSplit(name)
    print('*'*8 + 'FINISH' + '*'*8)