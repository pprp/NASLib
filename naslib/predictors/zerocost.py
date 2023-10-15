"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip
based on https://github.com/mohsaied/zero-cost-nas
"""
import torch
import logging
import math
import numpy as np 

from naslib.predictors.predictor import Predictor
from naslib.predictors.utils.pruners import predictive
from naslib import utils 

logger = logging.getLogger(__name__)


class ZeroCost(Predictor):
    def __init__(self, config, method_type="jacov"):
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.method_type = method_type
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        self.device = torch.device("cpu") #("cuda:0" if torch.cuda.is_available() else "cpu")
        _, dataloader, _ = self.build_search_dataloaders(config)
        self.dataloader = dataloader 
    
    def build_search_dataloaders(self, config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="train"
        )
        return train_queue, valid_queue, _  # test_queue is not used in search currently

    def query(self, graph, dataloader=None, info=None):
        if dataloader is None: 
            dataloader = self.dataloader 

        if isinstance(graph, list):
            loss_fn = graph[0].get_loss_fn()
            n_classes = graph[0].num_classes
        else: 
            loss_fn = graph.get_loss_fn() 
            n_classes = graph.num_classes

        score_list = [] # store score for each graph
        for g in graph:
            # if torch.cuda.is_available():
            #     g = g.to(self.device) 
            #     import pdb; pdb.set_trace()
            score = predictive.find_measures(
                    net_orig=g,
                    dataloader=dataloader,
                    dataload_info=(self.dataload, self.num_imgs_or_batches, n_classes),
                    device=self.device,
                    loss_fn=loss_fn,
                    measure_names=[self.method_type],
                )

            if math.isnan(score) or math.isinf(score):
                score = -1e8

            if self.method_type == 'synflow':
                if score == 0.:
                    return score

                score = math.log(score) if score > 0 else -math.log(-score)

            score_list.append(score)

        return np.array(score_list)
