'''Ensemble learning'''
from scipy.io import loadmat, savemat
import argparse
import numpy as np
import os
from tqdm import tqdm
from configparser import ConfigParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='INTEGRATED LEARNING')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--run', type=int, default=10,
                        help='RUN TIMES')
    parser.add_argument('--spc', type=int, default=10,
                        help='SAMPLE EACH CLASS')
    arg = parser.parse_args()
    config = ConfigParser()
    config.read('dataInfo.ini')
    block = [50, 100, 150, 200]
    save_root = 'prediction/{}/{}/integratedLearning_l1_clip'.format(arg.name, arg.spc)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for r in tqdm(range(arg.run)):
        res = []
        for b in block:
            path = 'prediction/{}/{}/{}_overall_skip_2_SGConv_l1_clip/{}.mat'.format(arg.name, arg.spc, b, r)
            res.append(loadmat(path)['pred'])
        # fr = np.where(res[0] == res[1], res[0], np.full(res[0].shape, -1))
        # fr = np.where(res[1] == res[2], res[1], fr)
        # fr = np.where(res[0] == res[2], res[0], fr)
        # fr = np.where(fr == -1, res[1], fr)
        # res -> h x w x len(block)
        res = np.stack(res, axis=-1)
        h, w = res.shape[:2]
        nc = config.getint(arg.name, 'nc')
        res = res.reshape((h * w, -1))
        fr = [np.bincount(x, minlength=nc) for x in res]
        fr = np.stack(fr, axis=0)
        fr = fr.reshape((h, w, -1))
        fr = np.argmax(fr, axis=-1)
        savemat(os.path.join(save_root, '{}.mat'.format(r)), {'pred': fr})
    print('*'*5 + 'FINISH' + '*'*5)
