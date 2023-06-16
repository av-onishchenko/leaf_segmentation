import os
import cv2
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.Metrics import SymmetricBestDice
from functools import lru_cache


class LeavesIntanceSegmDataset:
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
        mask = cv2.imread(os.path.join(self.root, image_filepath["inst_path"]))
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.get_image(idx)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"].T
            mask = transformed["mask"].T
        return image, mask


def add_xy_channels(mask):
    x_len = mask.shape[1]
    y_len = mask.shape[2]
    x_chanel = np.arange(x_len).reshape(-1, 1)
    y_chanel = np.arange(y_len).reshape(1, -1)

    x_chanel = np.expand_dims(np.repeat(x_chanel, y_len, axis=1), 0)
    y_chanel = np.expand_dims(np.repeat(y_chanel, x_len, axis=0), 0)
    return np.concatenate((mask, x_chanel, y_chanel), axis=0)


def InstanseSegmentation(x, model, reg, device="cpu", threshold=0.6, plt_show=False):
    model.eval()
    reg.eval()

    # Предсказываем количество листьев
    img = torch.unsqueeze(torch.FloatTensor(x[0]), 0).to(device)
    reg_result = int(np.round(reg(img).detach().cpu().numpy()[0])[0])

    # feature-extractor для картинки
    sig = nn.Sigmoid()
    res = sig(model(img)).detach().cpu().numpy()[0]
    features = sig(model.res_decoder1 / 1).detach().cpu().numpy()[0]

    # Добавляем каналы с координатами
    features = add_xy_channels(features)

    # Выбираем сегментированые пиксели
    good_pixels = []
    for i in range(res.shape[1]):
        for j in range(res.shape[2]):
            if res[0, i, j] > threshold:
                good_pixels.append(features[:, i, j][-8:])

    # Кластеризуем с предсказанным количеством листьев
    kmeans = KMeans(n_clusters=reg_result, n_init=10)
    kmeans.fit(good_pixels)
    labels = kmeans.labels_

    # Заполненяем маски
    cnt = 0
    for i in range(res.shape[1]):
        for j in range(res.shape[2]):
            if res[0, i, j] > threshold:
                res[0, i, j] = labels[cnt] + 1
                cnt += 1
            else:
                res[0, i, j] = 0

    # Собираем картинку
    new_pic = x[1][0, :, :] + x[1][1, :, :] * 256 + x[1][2, :, :] * 256 * 256
    a = np.unique(new_pic)
    for i in range(a.shape[0]):
        new_pic[new_pic == a[i]] = i

    if plt_show:
        plt.imshow(res.T / reg_result)
        plt.show()

    return SymmetricBestDice(new_pic, res, len(a) - 1, reg_result)
