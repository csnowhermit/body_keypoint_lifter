import os
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

'''
    带时序的数据集
'''
class Keypoint_Dataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.label = np.load(label_path)

        assert self.data.shape[0] == self.label.shape[0]    # 数据和标签文件数是一一对应的

    def __getitem__(self, index):
        curr_data = self.data[index]
        curr_label = self.label[index] / 1000.0

        # 人体做归一化
        # 归一化
        min_width = np.min(curr_data[:, 0])
        min_height = np.min(curr_data[:, 1])

        curr_width = np.max(curr_data[:, 0]) - np.min(curr_data[:, 0])
        curr_height = np.max(curr_data[:, 1]) - np.min(curr_data[:, 1])

        if float(curr_width) == 0.0:
            curr_width = 1.0
        if float(curr_height) == 0.0:
            curr_height = 1.0

        curr_data[:, 0] = (curr_data[:, 0] - min_width) / curr_width
        curr_data[:, 1] = (curr_data[:, 1] - min_height) / curr_height

        # # 归一化之后可视化下看对不对
        # xp = curr_data.T[0].T
        # yp = curr_data.T[1].T
        # plot_line_chart_part_2d(xp, yp, "%s_2d" % (index), mode='label')

        # 对于标签，将人移到原点附近（两髋关节的中心点为原点）
        left_hip = curr_label[23]
        right_hip = curr_label[24]

        center_x = (left_hip[0] + right_hip[0]) / 2.0
        center_y = (left_hip[1] + right_hip[1]) / 2.0
        center_z = (left_hip[2] + right_hip[2]) / 2.0

        # 将人移到中心
        curr_label[:, 0] = curr_label[:, 0] - center_x
        curr_label[:, 1] = curr_label[:, 1] - center_y
        curr_label[:, 2] = curr_label[:, 2] - center_z

        # 对3d关键点做归一化
        min_x = np.min(curr_label[:, 0])
        min_y = np.min(curr_label[:, 1])
        min_z = np.min(curr_label[:, 2])

        curr_x = np.max(curr_label[:, 0]) - np.min(curr_label[:, 0])
        curr_y = np.max(curr_label[:, 1]) - np.min(curr_label[:, 1])
        curr_z = np.max(curr_label[:, 2]) - np.min(curr_label[:, 2])

        if float(curr_x) == 0.0:
            curr_x = 1.0
        if float(curr_y) == 0.0:
            curr_y = 1.0
        if float(curr_z) == 0:
            curr_z = 1.0

        curr_label[:, 0] = (curr_label[:, 0] - min_x) / curr_x
        curr_label[:, 1] = (curr_label[:, 1] - min_y) / curr_y
        curr_label[:, 2] = (curr_label[:, 2] - min_z) / curr_z

        # # 可视化看下对不对
        # xp = curr_label.T[0].T
        # yp = curr_label.T[1].T
        # zp = curr_label.T[2].T
        # plot_line_chart_part(xp, yp, zp, "%s_3d" % (index), mode='label')

        curr_data = curr_data.flatten()
        curr_label = curr_label.flatten()
        return torch.tensor(curr_data, dtype=torch.float32), torch.tensor(curr_label, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    root_path = "./data/"
    train_dataset = Keypoint_Dataset(root_path + "data_train.npy", label_path=root_path + "label_train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = Keypoint_Dataset(root_path + "data_val.npy", label_path=root_path + "label_val.npy")
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)

    for idx, (data, label) in enumerate(val_dataloader):
        print(idx, data.shape, label.shape)





