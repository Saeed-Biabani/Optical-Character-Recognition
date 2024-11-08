import matplotlib.pyplot as plt
from typing import Mapping
from PIL import Image
import numpy as np
import torch

class Visualizer:
    def __init__(self, patterns : tuple[str]) -> None:
        self.patterns = patterns
        self.configs = {
            'loss' : {
                'figsize' : (15, 8),
                'title' : "Loss Curve",
                'fname' : "lossCurve.png",
                'legend_loc' : "upper right",
                'xlabel' : 'Iteration',
                'ylabel' : 'CrossEntropy Loss'
            },
            'acc' : {
                'figsize' : (15, 8),
                'title' : "Accuracy Curve",
                'fname' : "accCurve.png",
                'legend_loc' : "lower right",
                'xlabel' : 'Iteration',
                'ylabel' : 'Accuracy'
            },
            'lr' : {
                'figsize' : (15, 8),
                'title' : "Learning Rate Curve",
                'fname' : "lrCurve.png",
                'legend_loc' : "upper right",
                'xlabel' : 'Iteration',
                'ylabel' : 'Learning Rate'
            }
        }
    
    def __draw__(self, log : Mapping[str, tuple[float]], **kwargs) -> None:
        plt.figure(figsize = kwargs['figsize'])
        
        for key in log:
            plt.plot(log[key])
        
        plt.legend(list(log.keys()), loc = kwargs['legend_loc'])
        
        plt.xlabel(kwargs['xlabel'])
        plt.ylabel(kwargs['ylabel'])
        
        plt.title(kwargs['title'])
        plt.savefig(kwargs['fname'])
        plt.close()
    
    def __call__(self, log : Mapping[str, tuple[float]]) -> None:
        for patern in self.patterns:
            build_log = {}
            for key in log.keys():
                if patern in key:
                    build_log[key] = log[key]
            self.__draw__(build_log, **self.configs[patern])

def countParams(model : torch.nn.Module) -> int:
    return sum([
        np.prod(p.size()) for p in filter(
            lambda p: p.requires_grad, model.parameters()
        )
    ])

def updateLog(
    from_ : Mapping[str, float],
    to_ : Mapping[str, float]
) -> None:
    for key in from_:
        if key in to_.keys():
            to_[key].append(from_[key])

def loadImage(fname):
    return Image.open(fname).convert('L')

def visGrads(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.mean().item())

    plt.plot(grads)
    plt.savefig('grad.png')
    plt.close()