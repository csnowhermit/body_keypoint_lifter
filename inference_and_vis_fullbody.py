from __future__ import print_function
import os
from tqdm import tqdm
import torch.nn.parallel
import numpy as np

from model_body import SimpleBaseline as BodyModel
from model_lefthand import SimpleBaseline as LefthandModel
from model_righthand import SimpleBaseline as RighthandModel
from vis_label import plot_full_body, plot_body, plot_hand

'''
    推理并可视化人全身
    1.人体：mediapipe 33个点；
    2.左右手：按smplx的序列输入
'''

root_path = "./data/"

username = "fanzhaoxin"

keypoint2d_hand = np.load("D:/dataset/lifter_dataset/inference_data/%s_hand_smplx.npy" % username)    # 左右手
keypoint2d_body = np.load("D:/dataset/lifter_dataset/inference_data/%s_body_mp.npy" % username)    # 人体

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 人体
body_model = BodyModel(input_dim=33 * 2, out_dim=33 * 3)
body_model.to(device)

checkpoint_body = torch.load(os.path.join(root_path, "checkpoint20221021/body_197_loss_148.711475_78.878653.pt"), map_location=device)
body_model.load_state_dict(checkpoint_body['model_state'])
body_model.eval()

# 左手
lefthand_model = LefthandModel(input_dim=21*2, out_dim=21*3)
lefthand_model.to(device)

checkpoint_lefthand = torch.load(os.path.join(root_path, "D:/workspace/workspace_python/lifter_hand_left/data/checkpoint20221021/lefthand_181_loss_0.037327_26.498532.pt"), map_location=device)
lefthand_model.load_state_dict(checkpoint_lefthand['model_state'])
lefthand_model.eval()
lefthand_joint_index = [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 66, 67, 68, 69, 70]

# 右手
righthand_model = RighthandModel(input_dim=21*2, out_dim=21*3)
righthand_model.to(device)

checkpoint_righthand = torch.load(os.path.join(root_path, "D:/workspace/workspace_python/lifter_hand_right/data/checkpoint20221021/righthand_181_loss_0.037298_26.458579.pt"), map_location=device)
righthand_model.load_state_dict(checkpoint_righthand['model_state'])
righthand_model.eval()
righthand_joint_index = [21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 71, 72, 73, 74, 75]  # 手指指尖也要


# 人脸的下标顺序
face_joint_index = [55, 86, 87, 88, 89, 92, 93, 94, 92, 91, 90, 85, 84, 83, 82, 81, 76, 77, 78, 79, 80, 57, 104, 103, 102, 101, 106, 105, 23, 56, 95, 96, 97, 98, 99, 100, 24, 110, 116, 121, 125, 113, 112, 111, 115, 114, 107, 108, 109, 117, 118, 123, 122, 124, 119, 120, 126, 135, 143, 142, 141, 140, 139, 138, 137, 136, 127, 128, 129, 130, 131, 132, 133, 134, 59, 58]


final_hand_output_list = []    # 最终保存的左右手的list，按smplx顺序
final_body_output_list = []    # 最终保存的人体的list，按mediapipe顺序
for i in tqdm(range(keypoint2d_hand.shape[0])):
    hand_data = keypoint2d_hand[i, :, :]
    body_data = keypoint2d_body[i, :, :]

    # 1.分别整理三个模型所需的输入
    body_input = []
    lefthand_input = []
    righthand_input = []

    body_input = body_data

    for j in range(144):
        if j in lefthand_joint_index:
            lefthand_input.append(hand_data[j])
        if j in righthand_joint_index:
            righthand_input.append(hand_data[j])
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

    # 拼接左右手之前先做配准（即让人体的某个点坐标始终为0，暂定为左髋点，下标为23）
    left_hip = pred_body[23]

    pred_body[:, 0] = pred_body[:, 0] - left_hip[0]
    pred_body[:, 1] = pred_body[:, 1] - left_hip[1]
    pred_body[:, 2] = pred_body[:, 2] - left_hip[2]

    # 左手的配准
    left_wrist = pred_lefthand[0]

    pred_lefthand[:, 0] = pred_lefthand[:, 0] - left_wrist[0]
    pred_lefthand[:, 1] = pred_lefthand[:, 1] - left_wrist[1]
    pred_lefthand[:, 2] = pred_lefthand[:, 2] - left_wrist[2]

    # 右手的配准
    right_wrist = pred_righthand[0]

    pred_righthand[:, 0] = pred_righthand[:, 0] - right_wrist[0]
    pred_righthand[:, 1] = pred_righthand[:, 1] - right_wrist[1]
    pred_righthand[:, 2] = pred_righthand[:, 2] - right_wrist[2]

    ### 单独可视化的画不用这一步
    # # # 5.将左手手腕移到人体的左手手腕处
    # left_body_wrist = pred_body[15]    # mediapipe中左手手腕的下标
    # left_wrist = pred_lefthand[0]
    #
    # move_vector = left_body_wrist - left_wrist  # 表示从手部手腕，移动到身体手腕的向量
    # lefthand_after_move_vector = pred_lefthand + move_vector  # 将左手手腕移到人体的手腕处
    #
    # # # 6.将右手手腕移到人体的右手手腕处
    # right_body_wrist = pred_body[16]    # mediapipe中右手手腕的下标
    # right_wrist = pred_righthand[0]
    #
    # move_vector = right_body_wrist - right_wrist  # 表示从手部手腕，移动到身体手腕的向量
    # righthand_after_move_vector = pred_righthand + move_vector  # 将右手手腕移到人体的手腕处

    # 7.拼接结果
    # 再拼左手的
    lefthand_cnt = 0
    for j in range(144):
        if j in lefthand_joint_index:
            final_output[j] = pred_lefthand[lefthand_cnt]
            lefthand_cnt += 1

    # 再拼右手的
    righthand_cnt = 0
    for j in range(144):
        if j in righthand_joint_index:
            final_output[j] = pred_righthand[righthand_cnt]
            righthand_cnt += 1

    final_hand_output_list.append(final_output)
    final_body_output_list.append(pred_body)


    # 8.可视化
    # plot_full_body(pred_body, final_output, "./output/%s_%d_3d.png" % (username, i), mode='pred')
    plot_body(pred_body, "./output/%s_%d_body_3d.png" % (username, i), mode='body')
    plot_hand(final_output, "./output/%s_%d_lefthand_3d.png" % (username, i), mode='left')
    plot_hand(final_output, "./output/%s_%d_righthand_3d.png" % (username, i), mode='right')


# 保存最终推理结果
final_hand_output_list = np.array(final_hand_output_list, dtype=np.float32)
final_body_output_list = np.array(final_body_output_list, dtype=np.float32)
print(final_hand_output_list.shape)
print(final_body_output_list.shape)

np.save("./pred_hand_%s.npy" % username, final_hand_output_list)
np.save("./pred_body_%s.npy" % username, final_body_output_list)

