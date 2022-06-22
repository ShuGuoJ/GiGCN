import torch
from torch_geometric.data import Data, Batch
from torch.optim import optimizer as optimizer_
from torch_geometric.utils import accuracy
from torch_geometric.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
import time


class JointTrainer(object):
    r'''Joint trainer'''
    def __init__(self, models: list):
        super().__init__()
        self.models = models

    def train(self, subGraph: Batch, fullGraph: Data, optimizer, criterion, device, monitor = None, is_l1=False, is_clip=False):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        intNet.train()
        extNet.train()
        intNet.to(device)
        extNet.to(device)
        criterion.to(device)
        # Internal graph features
        # if subGraph.num_graphs < 5000:
        #     subGraph = subGraph.to(device)
        #     fe = intNet(subGraph)
        # else:
        #     batchsz = 5000
        #     fe = []
        #     n_iteration = subGraph.num_graphs // batchsz
        #     graph_list = subGraph.to_data_list()
        #     for n in range(n_iteration):
        #         batch = Batch.from_data_list(graph_list[n * batchsz: (n + 1) * batchsz])
        #         batch = batch.to(device)
        #         f = intNet(batch)
        #         fe.append(f)
        #     if subGraph.num_graphs % batchsz != 0:
        #         batch = Batch.from_data_list(graph_list[(n + 1) * batchsz:])
        #         batch = batch.to(device)
        #         f = intNet(batch)
        #         fe.append(f)
        #     fe = torch.cat(fe, dim=0)
        # subGraph = subGraph.to(device)
        fe = intNet(subGraph.to_data_list())

        # External graph features
        fullGraph.x = fe
        fullGraph = fullGraph.to(device)
        logits = extNet(fullGraph)
        indices = torch.nonzero(fullGraph.tr_gt, as_tuple=True)
        y = fullGraph.tr_gt[indices].to(device) - 1
        node_number = fullGraph.seg[indices]
        pixel_logits = logits[node_number]
        loss = criterion(pixel_logits, y)
        # l1 norm
        if is_l1:
            l1 = 0
            for p in intNet.parameters():
                l1 += p.norm(1)
            loss += 1e-4 * l1
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        # Clipping gradient
        if is_clip:
            # External gradient
            clip_grad_norm_(extNet.parameters(), max_norm=2., norm_type=2)
            # Internal gradient
            clip_grad_norm_(intNet.parameters(), max_norm=3., norm_type=2)
        optimizer.step()

        if monitor is not None:
            monitor.add([intNet.parameters(), extNet.parameters()], ord=2)
        return loss.item()

    def evaluate(self, subGraph, fullGraph, criterion, device):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        intNet.eval()
        extNet.eval()
        intNet.to(device)
        extNet.to(device)
        criterion.to(device)
        with torch.no_grad():
            # subGraph = subGraph.to(device)
            fe = intNet(subGraph.to_data_list())
            # if subGraph.num_graphs < 5000:
            #     subGraph = subGraph.to(device)
            #     fe = intNet(subGraph)
            # else:
            #     batchsz = 5000
            #     fe = []
            #     n_iteration = subGraph.num_graphs // batchsz
            #     graph_list = subGraph.to_data_list()
            #     for n in range(n_iteration):
            #         batch = Batch.from_data_list(graph_list[n * batchsz: (n + 1) * batchsz])
            #         batch = batch.to(device)
            #         f = intNet(batch)
            #         fe.append(f)
            #     if subGraph.num_graphs % batchsz != 0:
            #         batch = Batch.from_data_list(graph_list[(n + 1) * batchsz:])
            #         batch = batch.to(device)
            #         f = intNet(batch)
            #         fe.append(f)
            #     fe = torch.cat(fe, dim=0)
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            logits = extNet(fullGraph)
            pred = torch.argmax(logits, dim=-1)
            indices = torch.nonzero(fullGraph.te_gt, as_tuple=True)
            y = fullGraph.te_gt[indices].to(device) - 1
            node_number = fullGraph.seg[indices]
            pixel_pred = pred[node_number]
            pixel_logits = logits[node_number]
            loss = criterion(pixel_logits, y)
        return loss.item(), accuracy(pixel_pred, y)

    # Getting prediction results
    def predict(self, subGraph, fullGraph, device: torch.device):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        intNet.eval()
        extNet.eval()
        intNet.to(device)
        extNet.to(device)
        begin_time = time.time()
        with torch.no_grad():
            # Internal graph features
            fe = intNet(subGraph.to_data_list())

            # External graph features
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            logits = extNet(fullGraph)
        pred = torch.argmax(logits, dim=-1)
        end_time = time.time()
        print(f"time: {end_time - begin_time}")
        exit(0)
        # indices = torch.nonzero(fullGraph, as_tuple=True)
        # node_number = fullGraph.seg[indices]
        # pixel_pred = pred[node_number]

        return pred

    # Getting hidden features
    def getHiddenFeature(self, subGraph, fullGraph, device, gt = None, seg = None):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        intNet.eval()
        extNet.eval()
        intNet.to(device)
        extNet.to(device)
        with torch.no_grad():
            fe = intNet(subGraph.to_data_list())
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            fe = extNet(fullGraph)
        if gt is not None and seg is not None:
            indices = torch.nonzero(gt, as_tuple=True)
            gt = gt[indices] - 1
            node_number = seg[indices].to(device)
            fe = fe[node_number]
            return fe.cpu(), gt
        else:
            return fe.cpu()

    def get_parameters(self):
        return self.models[0].parameters(), self.models[1].parameters()

    def save(self, paths):
        torch.save(self.models[0].cpu().state_dict(), paths[0])
        torch.save(self.models[1].cpu().state_dict(), paths[1])



