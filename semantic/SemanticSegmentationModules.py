import os
import cv2
import torch
from torch import nn
from functools import lru_cache


class LeavesSemSegDataset:
    def __init__(self, root, images_data, transform=None):
        self.root = root
        self.images_data = images_data
        self.transform = transform

    def __len__(self):
        return len(self.images_data.index)

    @lru_cache(maxsize=300)
    def get_image(self, idx):
        image_filepath = self.images_data.iloc[idx]
        image = cv2.imread(os.path.join(self.root, image_filepath["img_path"]))
        mask = cv2.imread(os.path.join(self.root, image_filepath["sem_path"]))[:, :, :1]
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.get_image(idx)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].T
            mask = transformed["mask"].T

        return image, mask


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.transform = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn_transform = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        transform_x = self.bn_transform(self.transform(x))
        return self.relu2(out + transform_x)


class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, init_features):
        super(Backbone, self).__init__()

        features = init_features
        self.encoder1 = ResBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features):
        super(UNet, self).__init__()

        features = init_features
        self.backbone = Backbone(in_channels, out_channels, init_features)

        self.middle = ResBlock(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ResBlock((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ResBlock((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ResBlock((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ResBlock(features * 2, features)
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        encoder1 = self.backbone.encoder1(x)
        encoder2 = self.backbone.pool1(encoder1)
        encoder2 = self.backbone.encoder2(encoder2)
        encoder3 = self.backbone.pool2(encoder2)
        encoder3 = self.backbone.encoder3(encoder3)
        encoder4 = self.backbone.pool2(encoder3)
        encoder4 = self.backbone.encoder4(encoder4)
        encoder5 = self.backbone.pool4(encoder4)

        middle = self.middle(encoder5)
        decoder4 = self.upconv4(middle)
        decoder4 = self.decoder4(torch.cat((decoder4, encoder4), dim=1))
        decoder3 = self.upconv3(decoder4)
        decoder3 = self.decoder3(torch.cat((decoder3, encoder3), dim=1))
        decoder2 = self.upconv2(decoder3)
        decoder2 = self.decoder2(torch.cat((decoder2, encoder2), dim=1))
        decoder1 = self.upconv1(decoder2)
        decoder1 = self.decoder1(torch.cat((decoder1, encoder1), dim=1))
        return self.conv(decoder1)
