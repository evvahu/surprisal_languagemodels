import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def read_train(path, lang='HI'):
    pass


class VerbPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate=0, non_lin=True, function='sigmoid', layers=1):
        super(VerbPredictor, self).__init__()
        dropout_layer = nn.Dropout(dropout_rate)
        hiddenlayer = nn.Linear(input_dim, hidden_dim)
        outlayer = nn.Linear(hidden_dim, input_dim)
        self._linear = nn.Sequential(dropout_layer, transform)
    

    def forward(self, batch)



