import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mp_dict import MP_BODY_JOINT_NAMES, mp_body_dict, mp_halfbody_dict
from mp_dict_hand import SMPLX_JOINT_NAMES, smplx_hand_dict


# 随机颜色
colorList = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)] for i in range(len(mp_body_dict) + len(smplx_hand_dict))]

'''
    可视化全身的结果
    :param pred_body 人体 [33, 3]
    :param pred_hand 左右手 [144, 3]
    :param save_name 保存的文件名
    :param mode 结果类别：pred、label
'''
def plot_full_body(pred_body, pred_hand, save_name, mode='pred'):
    fig = plt.figure(figsize=(16, 16))  # figsize=(16, 16)
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')

    # 1.人体部分
    body_xp = pred_body.T[0].T
    body_yp = pred_body.T[1].T
    body_zp = pred_body.T[2].T

    idx = 0
    for key in mp_body_dict.keys():
        curr_x = []
        curr_y = []
        curr_z = []

        for item in mp_body_dict[key]:    # mediapipe中每个点都不为0，除了配准用的基准点
            curr_x.append(body_xp[MP_BODY_JOINT_NAMES.index(item)])
            curr_y.append(body_yp[MP_BODY_JOINT_NAMES.index(item)])
            curr_z.append(body_zp[MP_BODY_JOINT_NAMES.index(item)])


        if len(curr_x) > 0:  # 只有有点时才画
            ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
        idx += 1

    # 2.左右手部分
    hand_xp = pred_hand.T[0].T
    hand_yp = pred_hand.T[1].T
    hand_zp = pred_hand.T[2].T

    for key in smplx_hand_dict.keys():
        curr_x = []
        curr_y = []
        curr_z = []

        for item in smplx_hand_dict[key]:
            if hand_xp[SMPLX_JOINT_NAMES.index(item)] == 0.0 and hand_yp[SMPLX_JOINT_NAMES.index(item)] == 0.0 and hand_zp[SMPLX_JOINT_NAMES.index(item)] == 0.0:
                continue
            else:
                curr_x.append(hand_xp[SMPLX_JOINT_NAMES.index(item)])
                curr_y.append(hand_yp[SMPLX_JOINT_NAMES.index(item)])
                curr_z.append(hand_zp[SMPLX_JOINT_NAMES.index(item)])

        if len(curr_x) > 0:  # 只有有点时才画
            ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
        idx += 1

    ax.legend()
    plt.savefig(save_name)


'''
    可视化全身的结果
    :param pred_body 人体 [33, 3]
    :param save_name 保存的文件名
    :param mode 结果类别：pred、label
'''
def plot_body(pred_body, save_name, mode='pred'):
    fig = plt.figure(figsize=(16, 16))  # figsize=(16, 16)
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_zlim(-1, 1)

    # 1.人体部分
    body_xp = pred_body.T[0].T
    body_yp = pred_body.T[1].T
    body_zp = pred_body.T[2].T

    idx = 0
    for key in mp_body_dict.keys():
        curr_x = []
        curr_y = []
        curr_z = []

        for item in mp_body_dict[key]:    # mediapipe中每个点都不为0，除了配准用的基准点
            curr_x.append(body_xp[MP_BODY_JOINT_NAMES.index(item)])
            curr_y.append(body_yp[MP_BODY_JOINT_NAMES.index(item)])
            curr_z.append(body_zp[MP_BODY_JOINT_NAMES.index(item)])

        if len(curr_x) > 0:  # 只有有点时才画
            ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
        idx += 1

    ax.legend()
    plt.savefig(save_name)
    plt.cla()    # 清空ax对象


'''
    可视化全身的结果
    :param pred_hand 左右手 [144, 3]
    :param save_name 保存的文件名
    :param mode 结果类别：pred、label
'''
def plot_hand(pred_hand, save_name, mode='left'):
    fig = plt.figure(figsize=(16, 16))  # figsize=(16, 16)
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    ax.set_zlim(-0.5, 0.5)

    # 2.左右手部分
    hand_xp = pred_hand.T[0].T
    hand_yp = pred_hand.T[1].T
    hand_zp = pred_hand.T[2].T

    idx = 0
    for key in smplx_hand_dict.keys():
        if str(key).startswith(mode):
            curr_x = []
            curr_y = []
            curr_z = []

            for item in smplx_hand_dict[key]:
                curr_x.append(hand_xp[SMPLX_JOINT_NAMES.index(item)])
                curr_y.append(hand_yp[SMPLX_JOINT_NAMES.index(item)])
                curr_z.append(hand_zp[SMPLX_JOINT_NAMES.index(item)])

            if len(curr_x) > 0:  # 只有有点时才画
                ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
            idx += 1

    ax.legend()
    plt.savefig(save_name)
    plt.cla()    # 清空ax对象


'''
    半身可视化的结果
    :param pred_body 人体 [23, 3]
    :param save_name 保存的文件名
    :param mode 结果类别：pred、label
'''
def plot_halfbody(pred_body, save_name, mode='pred'):
    fig = plt.figure(figsize=(16, 16))  # figsize=(16, 16)
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_zlim(-1, 1)

    # 1.人体部分
    body_xp = pred_body.T[0].T
    body_yp = pred_body.T[1].T
    body_zp = pred_body.T[2].T

    idx = 0
    for key in mp_halfbody_dict.keys():
        curr_x = []
        curr_y = []
        curr_z = []

        for item in mp_halfbody_dict[key]:    # mediapipe中每个点都不为0，除了配准用的基准点
            curr_x.append(body_xp[MP_BODY_JOINT_NAMES.index(item)])
            curr_y.append(body_yp[MP_BODY_JOINT_NAMES.index(item)])
            curr_z.append(body_zp[MP_BODY_JOINT_NAMES.index(item)])

        if len(curr_x) > 0:  # 只有有点时才画
            ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
        idx += 1

    ax.legend()
    plt.savefig(save_name)
    plt.cla()    # 清空ax对象

