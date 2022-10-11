import os
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

'''
    关键点数据集
    2D关键点为mediapipe检测到的33个点
    3D关键点为smplx的joint
'''
class Keypoint_Dataset2(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.label = np.load(label_path)

        assert self.data.shape[0] == self.label.shape[0]    # 数据和标签文件数是一一对应的

        self.smplx_body_joint_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 62, 65]  # 60、61左大小脚趾；63、64右大小脚趾；10、11为left_foot right_foot

    def __getitem__(self, index):
        curr_data = self.data[index]
        curr_label = self.label[index]

        # 标签只要人体的
        new_label = []
        for i in range(144):
            if i in self.smplx_body_joint_index:
                new_label.append(curr_label[i])

        curr_data = curr_data.flatten()
        new_label = np.array(new_label).flatten()
        return torch.tensor(curr_data, dtype=torch.float32), torch.tensor(new_label, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    root_path = "D:/workspace/workspace_python/verify_body/data/"
    train_dataset = Keypoint_Dataset2(root_path + "data_train.npy", label_path=root_path + "label_train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # val_dataset = Keypoint_Dataset2(root_path + "data_val.npy", label_path=root_path + "label_val.npy")
    # val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)

    for idx, (data, label) in enumerate(train_dataloader):
        print(idx, data.shape, label.shape)





