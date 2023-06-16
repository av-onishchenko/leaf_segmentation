import os
import cv2
import torch
from torch import nn
import numpy as np
from semantic.SemanticSegmentationModules import ResBlock
from functools import lru_cache


class LeavesRegressorDataset:
    def __init__(self, root, images_data, transform=None):
        self.root = root
        self.images_data = images_data
        self.transform = transform

    def __len__(self):
        return len(self.images_data.index)

    @lru_cache(maxsize=300)
    def get_image(self, idx):
        image_filepath = self.images_data.iloc[idx]["img_path"]
        centers_filepath = image_filepath[:-7] + "centers.png"
        image = cv2.imread(os.path.join(self.root, image_filepath))
        centers = cv2.imread(os.path.join(self.root, centers_filepath))
        target = np.sum(centers / 255) // 3
        return image, target

    def __getitem__(self, idx):
        image, target = self.get_image(idx)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"].T
        return image, target


class RegressorModel(nn.Module):
    def __init__(self, backbone, features):
        super(RegressorModel, self).__init__()

        self.backbone = backbone
        self.encoder4 = ResBlock(features * 8, features * 12)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = ResBlock(features * 12, features * 16)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder6 = ResBlock(features * 16, features * 20)
        self.pool6 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.bn = nn.BatchNorm1d(features * 20)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(features * 20, 1)

    def forward(self, x):
        pooled0 = self.pool4(self.encoder4(self.backbone(x)))
        pooled1 = self.pool5(self.encoder5(pooled0))
        pooled2 = self.flatten(self.pool6(self.encoder6(pooled1)))
        out = self.linear(self.dropout(self.bn(pooled2)))
        return out
