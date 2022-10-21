from __future__ import print_function
import os
import torch.nn.parallel
import numpy as np

from model_body import SimpleBaseline
from vis_label import plot_line_chart_part

'''
    可视化推理结果及标签
'''

root_path = "./data/"
keypoint2d = np.load("./data/data_val.npy")
keypoint3d = np.load("./data/label_val.npy")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleBaseline(input_dim=33*2, out_dim=33*3)
model.to(device)

checkpoint_body = torch.load(os.path.join(root_path, "checkpoint/body_199_loss_897.556960.pt"), map_location=device)
model.load_state_dict(checkpoint_body['model_state'])
model.eval()

for i in range(keypoint2d.shape[0]):
    input_data = keypoint2d[i, :, :]

    # 归一化
    lefthand_min_width = np.min(input_data[:, 0])
    lefthand_min_height = np.min(input_data[:, 1])

    lefthand_width = np.max(input_data[:, 0]) - np.min(input_data[:, 0])
    lefthand_height = np.max(input_data[:, 1]) - np.min(input_data[:, 1])

    if float(lefthand_width) == 0.0:
        lefthand_width = 1.0
    if float(lefthand_height) == 0.0:
        lefthand_height = 1.0

    input_data[:, 0] = (input_data[:, 0] - lefthand_min_width) / lefthand_width
    input_data[:, 1] = (input_data[:, 1] - lefthand_min_height) / lefthand_height

    input_data = input_data.flatten()

    input_data = torch.tensor(input_data).unsqueeze(0)
    input_data = input_data.to(device)

    pred_body = model(input_data)

    pred_body = pred_body.view(-1, 3)
    pred_body = pred_body.detach().cpu().numpy()

    # 5.输出，可视化
    xp = pred_body.T[0].T
    yp = pred_body.T[1].T
    zp = pred_body.T[2].T
    plot_line_chart_part(xp, yp, zp, i, mode='pred')

    # 6.可视化label
    label = keypoint3d[i]
    xp = label.T[0].T
    yp = label.T[1].T
    zp = label.T[2].T
    plot_line_chart_part(xp, yp, zp, i, mode='label')

