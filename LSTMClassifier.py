# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:59:51 2024

@author: anash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics

class LSTMClassifierTrain(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.apply(self._init_weights)
        self.save_hyperparameters()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.train_acc(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.valid_acc(outputs, labels)
        self.log('val_acc', self.valid_acc)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm == 0:
                    print(f'Zero gradient for {name}')
                    