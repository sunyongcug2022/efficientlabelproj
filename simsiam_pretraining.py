# Import the Python frameworks we need for this tutorial.
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor

from lightly.data import LightlyDataset
from lightly.transforms import SimSiamTransform, utils

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.utils.dist import print_rank_zero
from lightly.utils.benchmarking import MetricCallback

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import numpy as np

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices_num = 1


class SimSiam(pl.LightningModule):

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = list(resnet.children())[-1].in_features
        self.projection_head = SimSiamProjectionHead(feature_dim, feature_dim,
                                                     proj_hidden_dim)
        self.prediction_head = SimSiamPredictionHead(proj_hidden_dim,
                                                     pred_hidden_dim, out_dim)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch[0]
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


def create_data_loader_ssl(path_to_train_data, path_to_test_data, input_size, batch_size):
    # define the augmentations for self-supervised learning
    SSLTransform = SimSiamTransform(input_size=input_size)

    # create a lightly dataset for training with augmentations
    dataset_train_ssl = LightlyDataset(input_dir=path_to_train_data,
                                       transform=SSLTransform)

    # create a dataloader for training
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        num_workers=num_workers,
        # sampler=train_sampler,
    )

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=utils.IMAGENET_NORMALIZE["mean"],
                                         std=utils.IMAGENET_NORMALIZE["std"])
    ])

    # create a lightly dataset for embedding
    dataset_test_ssl = LightlyDataset(input_dir=path_to_test_data,
                                      transform=test_transforms)

    # create a dataloader for embedding
    dataloader_test_ssl = torch.utils.data.DataLoader(
        dataset_test_ssl,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=valid_sampler,
    )

    return dataloader_train_ssl, dataloader_test_ssl


def trained_for_ssl(ssl_model, input_size, batch_size, path_to_data,
                    max_epochs):
    dataloader_train_ssl, dataloader_test_ssl = create_data_loader_ssl(
        input_size, batch_size)
    trainer = pl.Trainer(max_epochs=max_epochs,
                         devices=1,
                         accelerator=accelerator)
    trainer.fit(ssl_model, dataloader_train_ssl)
    print(logged_metrics)

    return ssl_model


def get_pretrain_model(input_size, batch_size, path_to_data, max_epochs):
    ssl_model = SimSiam()
    trained_ssl_model = trained_for_ssl(ssl_model, input_size, batch_size,
                                        path_to_data, max_epochs)
    return trained_ssl_model