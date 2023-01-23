from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
import os
from model import LES
from torch.autograd import Variable
from penalty import DivergenceLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
import argparse
from sklearn.model_selection import train_test_split

def get_data_files(directory,file_ext=".pt",file_prefix='sample'):
    return [os.path.join(directory,file) for file in os.listdir(directory) if file.endswith(file_ext) and file.startswith(file_prefix)]

parser = argparse.ArgumentParser(description='Run turbulent flow net model')

parser.add_argument('-d',
                        '--data-directory',
                        metavar='data_directory',
                        type=str,
                        help='Dataset directory for train,test, and val'
                    )

parser.add_argument('-train',
                        '--train-ratio',
                        metavar='train_ratio',
                        type=float,
                        default=0.7,
                        help='Train ratio'
                    )               

parser.add_argument('-test',
                        '--test-ratio',
                        metavar='test_ratio',
                        type=float,
                        default=0.10,
                        help='Train ratio'
                    )

parser.add_argument('-val',
                        '--val-ratio',
                        metavar='val_ratio',
                        type=float,
                        default=0.15,
                        help='Valation ratio'
                    )

parser.add_argument('-o',
                        '--output-model',
                        metavar='output_model',
                        type=str,
                        default="model.pth",
                        help='Output model'
                    )

args = parser.parse_args()

data_directory=args.data_directory
output_model=args.output_model

train_ratio = args.train_ratio
test_ratio = args.test_ratio
val_ratio = args.val_ratio

#best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
min_mse = 1
time_range  = 6
output_length = 4
input_length = 26
learning_rate = 0.001
dropout_rate = 0
kernel_size = 3
batch_size = 32

data_files = get_data_files(data_directory)
#split valid and test
train_indices,test_indices = train_test_split(data_files, test_size=1 - train_ratio)
valid_indices,test_indices = train_test_split(test_indices, test_size=test_ratio/(test_ratio + val_ratio)) 

model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
            dropout_rate = dropout_rate, time_range = time_range).to(device)
model = nn.DataParallel(model)

train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, True)
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_fun = torch.nn.MSELoss()
regularizer = DivergenceLoss(torch.nn.MSELoss())
coef = 0

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)

train_mse = []
valid_mse = []
test_mse = []
for i in range(100):
    start = time.time()
    torch.cuda.empty_cache()
    scheduler.step()
    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun, coef, regularizer))#
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model 
        torch.save(best_model, output_model)
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))
print(time_range, min_mse)


loss_fun = torch.nn.MSELoss()
best_model = torch.load(output_model)
test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "loss_curve": loss_curve}, 
            "results.pt")
