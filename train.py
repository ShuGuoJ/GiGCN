'''Training'''
from scipy.io import loadmat
import numpy as np
import argparse
import configparser
import torch
from torch import nn
from skimage.segmentation import slic
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import scale, minmax_scale
import os
from PIL import Image
from utils import get_graph_list, split, get_edge_index
import math
from Model.module import SubGcnFeature, GraphNet
from Trainer import JointTrainer
from Monitor import GradMonitor
from visdom import Visdom
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100,
                        help='BLOCK SIZE')
    parser.add_argument('--epoch', type=int, default=1,
                        help='ITERATION')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID')
    parser.add_argument('--comp', type=int, default=10,
                        help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64,
                        help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=10,
                        help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=10,
                        help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=128,
                        help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0.,
                        help='WEIGHT DECAY')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    viz = Visdom(port=17000)

    # Data processing
    # Reading hyperspectral image
    data_path = 'data/{0}/{0}.mat'.format(arg.name)
    m = loadmat(data_path)
    data = m[config.get(arg.name, 'data_key')]
    gt_path = 'data/{0}/{0}_gt.mat'.format(arg.name)
    m = loadmat(gt_path)
    gt = m[config.get(arg.name, 'gt_key')]
    # Normalizing data
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = data.astype(np.float)
    if arg.name == 'Xiongan':
        minmax_scale(data, copy=False)
    data_normalization = scale(data).reshape((h, w, c))

    # Superpixel segmentation
    seg_root = 'data/rgb'
    seg_path = os.path.join(seg_root, '{}_seg_{}.npy'.format(arg.name, arg.block))
    if os.path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        rgb_path = os.path.join(seg_root, '{}_rgb.jpg'.format(arg.name))
        img = Image.open(rgb_path)
        img_array = np.array(img)
        # The number of superpixel
        n_superpixel = int(math.ceil((h * w) / arg.block))
        seg = slic(img_array, n_superpixel, arg.comp)
        # Saving
        np.save(seg_path, seg)

    # Constructing graphs
    graph_path = 'data/{}/{}_graph.pkl'.format(arg.name, arg.block)
    if os.path.exists(graph_path):
        graph_list = torch.load(graph_path)
    else:
        graph_list = get_graph_list(data_normalization, seg)
        torch.save(graph_list, graph_path)
    subGraph = Batch.from_data_list(graph_list)

    # Constructing full graphs
    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)
        np.save(full_edge_index_path, edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
    fullGraph = Data(None,
                     edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                     seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

    for r in range(arg.run):
        print('*'*5 + 'Run {}'.format(r) + '*'*5)
        # Reading the training data set and testing data set
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
        tr_gt, te_gt = m['train_gt'], m['test_gt']
        tr_gt_torch, te_gt_torch = torch.from_numpy(tr_gt).long(), torch.from_numpy(te_gt).long()
        fullGraph.tr_gt, fullGraph.te_gt = tr_gt_torch, te_gt_torch

        gcn1 = SubGcnFeature(config.getint(arg.name, 'band'), arg.hsz)
        gcn2 = GraphNet(arg.hsz, arg.hsz, config.getint(arg.name, 'nc'))
        optimizer = torch.optim.Adam([{'params': gcn1.parameters()},
                                      {'params': gcn2.parameters()}],
                                     weight_decay=arg.wd)
        criterion = nn.CrossEntropyLoss()
        trainer = JointTrainer([gcn1, gcn2])
        monitor = GradMonitor()

        # Plotting a learning curve and gradient curve
        viz.line([[0., 0., 0.]], [0], win='{}_train_test_acc_{}'.format(arg.name, r),
                 opts={'title': '{} train&test&acc {}'.format(arg.name, r),
                       'legend': ['train', 'test', 'acc']})
        viz.line([[0., 0.]], [0], win='{}_grad_{}'.format(arg.name, r), opts={'title': '{} grad {}'.format(arg.name, r),
                                                                              'legend': ['internal', 'external']})

        device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        max_acc = 0
        save_root = 'models/{}/{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, arg.block)
        pbar = tqdm(range(arg.epoch))
        # Training
        for epoch in pbar:
            pbar.set_description_str('Epoch: {}'.format(epoch))
            tr_loss = trainer.train(subGraph, fullGraph, optimizer, criterion, device, monitor.clear(), is_l1=True, is_clip=True)
            te_loss, acc = trainer.evaluate(subGraph, fullGraph, criterion, device)
            pbar.set_postfix_str('train loss: {} test loss:{} acc:{}'.format(tr_loss, te_loss, acc))
            viz.line([[tr_loss, te_loss, acc]], [epoch], win='{}_train_test_acc_{}'.format(arg.name, r), update='append')
            viz.line([monitor.get()], [epoch], win='{}_grad_{}'.format(arg.name, r), update='append')

            if acc > max_acc:
                max_acc = acc
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                trainer.save([os.path.join(save_root, 'intNet_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'extNet_best_{}_{}.pkl'.format(arg.spc, r))])
    print('*'*5 + 'FINISH' + '*'*5)

