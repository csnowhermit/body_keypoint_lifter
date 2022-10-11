import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_mp33_human36m import Keypoint_Dataset2 as Keypoint_Dataset


'''
    单帧模型
'''

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return x + self.block(x)

'''
    最后的三个head
'''
class Last(nn.Module):
    def __init__(self, out_dim):
        super(Last, self).__init__()
        self.block = nn.Sequential(
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.block(x)



class SimpleBaseline(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(SimpleBaseline, self).__init__()

        self.pre_block = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.res_block1 = ResBlock()
        self.res_block2 = ResBlock()

        # 这里分成3head的
        self.last = Last(out_dim=out_dim)


    def forward(self, x):
        output = self.pre_block(x)
        output = self.res_block1(output)
        output = self.res_block2(output)    # [batch_size, 1024]

        last = self.last(output)

        return last

if __name__ == '__main__':
    model = SimpleBaseline(input_dim=33*2, out_dim=23*3)
    model.eval()

    root_path = "D:/workspace/workspace_python/verify_body/data/"

    train_dataset = Keypoint_Dataset(root_path + "data_train.npy", label_path=root_path + "label_train.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    for i, (data, label) in enumerate(train_dataloader):
        pred = model(data)
        print(i, data.shape, pred.shape, label.shape)



