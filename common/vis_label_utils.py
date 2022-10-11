import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common.joint_names import JOINT_NAMES
from common.my_smplx_dict import smplx_dict


colorList = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)] for item in smplx_dict.keys()]

'''
    画折线图
'''
def plot_line_chart(xp, yp, zp, i, mode='target'):
    # 画折线图
    fig = plt.figure(figsize=(16, 16))
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')

    idx = 0
    for key in smplx_dict.keys():
        curr_x = [xp[JOINT_NAMES.index(item)] for item in smplx_dict[key]]
        curr_y = [yp[JOINT_NAMES.index(item)] for item in smplx_dict[key]]
        curr_z = [zp[JOINT_NAMES.index(item)] for item in smplx_dict[key]]

        ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
        idx += 1
    ax.legend()
    # plt.show()
    plt.savefig("./output/%d_%s.png" % (i, mode))

'''
    部分点画折线图
'''
def plot_line_chart_part(xp, yp, zp, i, mode='target'):
    # 画折线图
    fig = plt.figure(figsize=(16, 16))    # figsize=(16, 16)
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')

    idx = 0
    for key in smplx_dict.keys():
        curr_x = []
        curr_y = []
        curr_z = []

        for item in smplx_dict[key]:
            if xp[JOINT_NAMES.index(item)] == 0.0 and yp[JOINT_NAMES.index(item)] == 0.0 and zp[JOINT_NAMES.index(item)] == 0.0:
                continue
            else:
                curr_x.append(xp[JOINT_NAMES.index(item)])
                curr_y.append(yp[JOINT_NAMES.index(item)])
                curr_z.append(zp[JOINT_NAMES.index(item)])

        if len(curr_x) > 0:    # 只有有点时才画
            ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
            # # plt.show()
            # ax.legend()
            # plt.savefig("./output/%d_%s_%s.png" % (i, mode, key))

        idx += 1
    ax.legend()
    # plt.show()
    plt.savefig("./output/%s_%s.png" % (str(i), mode))

'''
    画2D的折线图
'''
def plot_line_chart_part_2d(xp, yp, i, mode='target'):
    # 画折线图
    fig = plt.figure(figsize=(16, 16))    # figsize=(16, 16)

    idx = 0
    for key in smplx_dict.keys():
        curr_x = []
        curr_y = []

        for item in smplx_dict[key]:
            if xp[JOINT_NAMES.index(item)] == 0.0 and yp[JOINT_NAMES.index(item)] == 0.0:
                continue
            else:
                curr_x.append(xp[JOINT_NAMES.index(item)])
                curr_y.append(yp[JOINT_NAMES.index(item)])

        if len(curr_x) > 0:    # 只有有点时才画
            # ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
            plt.plot(curr_x, curr_y, 'go-', color=colorList[idx])
            # # plt.show()
            # ax.legend()
            # plt.savefig("./output/%d_%s_%s.png" % (i, mode, key))

        idx += 1
    # plt.show()
    plt.savefig("./output2d/%s_%s.png" % (str(i), mode))



'''
    画半身的折线图
'''
def plot_line_chart_halfbody(xp, yp, zp, i, mode='target'):
    keypoint3d_delete_joints_list = ['pelvis', 'spine1', 'spine2', 'spine3',
                                     'left_hip', 'left_knee', 'left_ankle', 'left_heel', 'left_foot', 'left_big_toe',
                                     'left_small_toe',
                                     'right_hip', 'right_knee', 'right_ankle', 'right_heel', 'right_foot',
                                     'right_big_toe', 'right_small_toe']

    # 先整理实际存在的joint list
    final_joint_names = []
    for joint in JOINT_NAMES:
        if joint not in keypoint3d_delete_joints_list:
            final_joint_names.append(joint)

    # 画折线图
    fig = plt.figure(figsize=(16, 16))
    ax = fig.gca(projection='3d')
    ax.set_title(mode, fontsize=30)
    plt.xlabel('ai')
    plt.ylabel('bi')
    plt.title = mode

    idx = 0
    for key in smplx_dict.keys():
        # 一组一组的画（这里要过滤掉下半身的joint）
        curr_x = []
        curr_y = []
        curr_z = []

        for item in smplx_dict[key]:
            if item in final_joint_names:    # 进行预测的joint才有
                curr_x.append(xp[final_joint_names.index(item)])
                curr_y.append(yp[final_joint_names.index(item)])
                curr_z.append(zp[final_joint_names.index(item)])

        if len(curr_x) > 0:    # 只有这条线上有点才画
            ax.plot(curr_x, curr_y, curr_z, 'go-', color=colorList[idx])
        idx += 1
    ax.legend()
    # plt.show()
    plt.savefig("./output/%d_%s.png" % (i, mode))

'''
    画三维的散点图
'''
def plot_point_chart(xp, yp, zp, i, mode='target'):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xp, yp, zp)

    # 添加坐标轴（顺序是Z、Y、X）
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('X', fontdict={'size': 15, 'color': 'red'})

    plt.savefig("./output/%s_%s.png" % (i, mode))


