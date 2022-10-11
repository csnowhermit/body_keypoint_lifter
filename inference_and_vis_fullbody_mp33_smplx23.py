from __future__ import print_function
import os
from tqdm import tqdm
import torch.nn.parallel
import numpy as np

from model_single_frame import SimpleBaseline as BodyModel
from model_single_frame import SimpleBaseline as LefthandModel
from model_single_frame import SimpleBaseline as RighthandModel
from common.vis_label_utils import plot_line_chart_part

'''
    直接使用mediapipe的33个关键点作为输入
'''

root_path = "./data/"
keypoint2d_hand = np.load("/path/to/hand_mp.npy")    # 这里2d关键点只做左右手
keypoint2d_body = np.load("/path/to/body_mp.npy")    # 这里2d关键点只做左右手

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 人体
body_model = BodyModel(input_dim=33 * 2, out_dim=23 * 3)
body_model.to(device)

checkpoint_body = torch.load(os.path.join(root_path, "checkpoint/body.pt"), map_location=device)
body_model.load_state_dict(checkpoint_body['model_state'])
body_model.eval()
body_joint_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 62, 65]  # 60、61左大小脚趾；63、64右大小脚趾；10、11为left_foot right_foot

# 左手
lefthand_model = LefthandModel(input_dim=21*2, out_dim=21*3)
lefthand_model.to(device)

checkpoint_lefthand = torch.load(os.path.join(root_path, "checkpoint/lefthand.pt"), map_location=device)
lefthand_model.load_state_dict(checkpoint_lefthand['model_state'])
lefthand_model.eval()
lefthand_joint_index = [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 66, 67, 68, 69, 70]

# 右手
righthand_model = RighthandModel(input_dim=21*2, out_dim=21*3)
righthand_model.to(device)

checkpoint_righthand = torch.load(os.path.join(root_path, "checkpoint/righthand.pt"), map_location=device)
righthand_model.load_state_dict(checkpoint_righthand['model_state'])
righthand_model.eval()
righthand_joint_index = [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 71, 72, 73, 74, 75]  # 手指指尖也要

# 人脸的下标顺序
face_joint_index = [55, 86, 87, 88, 89, 92, 93, 94, 92, 91, 90, 85, 84, 83, 82, 81, 76, 77, 78, 79, 80, 57, 104, 103, 102, 101, 106, 105, 23, 56, 95, 96, 97, 98, 99, 100, 24, 110, 116, 121, 125, 113, 112, 111, 115, 114, 107, 108, 109, 117, 118, 123, 122, 124, 119, 120, 126, 135, 143, 142, 141, 140, 139, 138, 137, 136, 127, 128, 129, 130, 131, 132, 133, 134, 59, 58]


final_output_list = []
for i in tqdm(range(keypoint2d_hand.shape[0])):
    input_data = keypoint2d_hand[i, :, :]
    body_data = keypoint2d_body[i, :, :]

    # 1.分别整理三个模型所需的输入
    body_input = []
    lefthand_input = []
    righthand_input = []

    body_input = body_data

    for j in range(144):
        if j in lefthand_joint_index:
            lefthand_input.append(input_data[j])
        if j in righthand_joint_index:
            righthand_input.append(input_data[j])
    body_input = np.array(body_input)
    lefthand_input = np.array(lefthand_input)
    righthand_input = np.array(righthand_input)

    # 2.推理人体部分
    # 归一化
    body_min_width = np.min(body_input[:, 0])
    body_min_height = np.min(body_input[:, 1])

    body_width = np.max(body_input[:, 0]) - np.min(body_input[:, 0])
    body_height = np.max(body_input[:, 1]) - np.min(body_input[:, 1])

    if float(body_width) == 0.0:
        body_width = 1.0
    if float(body_height) == 0.0:
        body_height = 1.0

    body_input[:, 0] = (body_input[:, 0] - body_min_width) / body_width
    body_input[:, 1] = (body_input[:, 1] - body_min_height) / body_height

    body_input = body_input.flatten()

    body_input = torch.tensor(body_input).unsqueeze(0)
    body_input = body_input.to(device)

    pred_body = body_model(body_input)

    pred_body = pred_body.view(-1, 3)
    pred_body = pred_body.detach().cpu().numpy()

    # 3.左手
    lefthand_min_width = np.min(lefthand_input[:, 0])
    lefthand_min_height = np.min(lefthand_input[:, 1])

    lefthand_width = np.max(lefthand_input[:, 0]) - np.min(lefthand_input[:, 0])
    lefthand_height = np.max(lefthand_input[:, 1]) - np.min(lefthand_input[:, 1])

    if float(lefthand_width) == 0.0:
        lefthand_width = 1.0
    if float(lefthand_height) == 0.0:
        lefthand_height = 1.0

    lefthand_input[:, 0] = (lefthand_input[:, 0] - lefthand_min_width) / lefthand_width
    lefthand_input[:, 1] = (lefthand_input[:, 1] - lefthand_min_height) / lefthand_height

    lefthand_input = lefthand_input.flatten()

    lefthand_input = torch.tensor(lefthand_input).unsqueeze(0)
    lefthand_input = lefthand_input.to(device)

    pred_lefthand = lefthand_model(lefthand_input)

    pred_lefthand = pred_lefthand.view(-1, 3)
    pred_lefthand = pred_lefthand.detach().cpu().numpy()

    # 4.右手
    righthand_min_width = np.min(righthand_input[:, 0])
    righthand_min_height = np.min(righthand_input[:, 1])

    righthand_width = np.max(righthand_input[:, 0]) - np.min(righthand_input[:, 0])
    righthand_height = np.max(righthand_input[:, 1]) - np.min(righthand_input[:, 1])

    if float(righthand_width) == 0.0:
        righthand_width = 1.0
    if float(righthand_height) == 0.0:
        righthand_height = 1.0

    righthand_input[:, 0] = (righthand_input[:, 0] - righthand_min_width) / righthand_width
    righthand_input[:, 1] = (righthand_input[:, 1] - righthand_min_height) / righthand_height

    righthand_input = righthand_input.flatten()

    righthand_input = torch.tensor(righthand_input).unsqueeze(0)
    righthand_input = righthand_input.to(device)

    pred_righthand = righthand_model(righthand_input)

    pred_righthand = pred_righthand.view(-1, 3)
    pred_righthand = pred_righthand.detach().cpu().numpy()

    # 拼接结果
    final_output = np.zeros([144, 3])

    # # 5.将左手手腕移到人体的左手手腕处
    left_body_wrist = pred_body[18]    # body 23个点用这个
    left_wrist = pred_lefthand[0]

    move_vector = left_body_wrist - left_wrist  # 表示从手部手腕，移动到身体手腕的向量
    lefthand_after_move_vector = pred_lefthand + move_vector  # 将左手手腕移到人体的手腕处

    # 6.将右手手腕移到人体的右手手腕处
    right_body_wrist = pred_body[19]    # body 23个点用这个
    right_wrist = pred_righthand[0]

    move_vector = right_body_wrist - right_wrist  # 表示从手部手腕，移动到身体手腕的向量
    righthand_after_move_vector = pred_righthand + move_vector  # 将右手手腕移到人体的手腕处

    # 7.拼接结果
    # 先拼body的
    body_cnt = 0
    for j in range(144):
        if j in body_joint_index:    # 各自按照各自的顺序拼
            final_output[j] = pred_body[body_cnt]
            body_cnt += 1

    # 再拼左手的
    lefthand_cnt = 0
    for j in range(144):
        if j in lefthand_joint_index:
            final_output[j] = lefthand_after_move_vector[lefthand_cnt]
            lefthand_cnt += 1

    # 再拼右手的
    righthand_cnt = 0
    for j in range(144):
        if j in righthand_joint_index:
            final_output[j] = righthand_after_move_vector[righthand_cnt]
            righthand_cnt += 1

    final_output_list.append(final_output)

    # 8.可视化
    xp = final_output.T[0].T
    yp = final_output.T[1].T
    zp = final_output.T[2].T
    plot_line_chart_part(xp, yp, zp, "%s_%d" % ("pred", i), mode='pred')


# 保存最终推理结果
final_output_list = np.array(final_output_list, dtype=np.float32)
print(final_output_list.shape)
np.save("./pred.npy", final_output_list)