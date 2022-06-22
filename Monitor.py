"""
Monitoring the change of gradient
"""
import torch
import numpy as np


class GradMonitor(object):
    def __init__(self):
        self.grads = []

    def clear(self):
        self.grads.clear()
        return self

    def add(self, parameters_list: list, ord=1):
        grad = []
        for parameters in parameters_list:
            grad_norm = []
            for p in parameters:
                if p.requires_grad and p.grad is not None:
                    norm = p.grad.norm(ord)
                    grad_norm.append(norm)
                else:
                    continue
            grad_norm = torch.tensor(grad_norm, dtype=torch.float)
            grad.append(grad_norm.norm(ord).item())
        self.grads.append(grad)

    def get(self):
        if len(self.grads) == 0:
            return None
        else:
            # return float(np.mean(self.grads))
            return np.mean(self.grads, axis=0).tolist()


