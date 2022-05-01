import os
import random
from turtle import forward
import numpy as np
import os.path as osp
from tqdm import tqdm
import argparse
# Torch-related imports
import torch
import torch.nn as nn
# from torch import optim
import torch.backends.cudnn as cudnn
# from torch.optim.lr_scheduler import MultiStepLR
import sys
sys.path.append('../')
from models.resnet import ResNet18
# From magnet_loss
# from utils.magnet_loss import MagnetLoss
# from utils.attacks import final_attack_eval
# from utils.magnet_training import magnet_epoch_wrapper
# from utils.setups import magnet_assertions, get_batch_builders, get_magnet_data
from utils.utils import get_softmax_probs
# from utils.logging import (report_epoch_and_save, print_to_log, update_log,
    # print_training_params, check_best_model, copy_best_checkpoint)
from utils.utils import eval_model, compute_real_losses, copy_pretrained_model
# from utils.train_settings import parse_settings
# from datasets.load_dataset import load_dataset
from robustbench.data import load_cifar10
from autoattack import AutoAttack

# For deterministic behavior
cudnn.deterministic = True
cudnn.benchmark = False


class model_for_robustbench(nn.Module):
    def __init__(self, model, magnet_data):
        super(model_for_robustbench, self).__init__()
        self.model = model
        self.magnet_data = magnet_data

    def forward(self, x):
        logits, embedding = self.model(x)
        if(self.magnet_data is None):
            return(logits)
        else:
            return(get_softmax_probs(embedding, self.magnet_data, return_scores=True))


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=99,
                        help='random seed for reproducibility')
    parser.add_argument('--model_path', 
                        help='path to model')

    args = parser.parse_args()
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # Decide device to use
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = ResNet18(num_classes=10).to(device)
    if args.model_path is not None:
        model = copy_pretrained_model(model, args.model_path)
    
    ckpt = torch.load(args.model_path)
    try:
        magnet_data = {
            'cluster_classes'   : ckpt['cluster_classes'],
            'cluster_centers'   : ckpt['cluster_centers'],
            'variance'          : ckpt['variance'],
            'L'                 : ckpt['L'],
            # 'L'                 : 10,
            'K'                 : ckpt['K'],
            'normalize_probs'   : ckpt['normalize_probs'],
            # 'normalize_probs'   : True,
        }
        print('Succesfully loaded magnet_data from checkpoint')
    except:
        magnet_data = None
        print('Unable to load magnet_data from checkpoint. '
            'Regular training is inferred')
    eval_model = model_for_robustbench(model, magnet_data)
    x_test, y_test = load_cifar10(n_examples=100)
    # threat_model='Linf'
    # adversary = AutoAttack(eval_model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    # adversary = AutoAttack(eval_model, norm='Linf', eps=8/255)
    adversary = AutoAttack(eval_model, norm='L2', eps=0.5)
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    # print(x_adv)

if __name__ == '__main__':
    main()