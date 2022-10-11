import os
import cv2
import json
import numpy as np
import yaml
import smplx
import torch
from tqdm import tqdm
from pycocotools.coco import COCO

import pyrender
import trimesh

from common.vis_label_utils import plot_line_chart_part, plot_line_chart_part_2d, plot_point_chart
from common.pix2world_util import pixel_to_world, convert_2d_to_3d
from common.joint_names import JOINT_NAMES


'''
    human3.6m，图像坐标转世界坐标
'''


def main(target_subject):
    # target_subject = 11
    # target_action = 2
    # target_subaction = 1
    # target_frame = 0
    # target_cam = 1

    keypoint3d = {}    # <img_path, joint3d>

    # 1、读取数据
    db = COCO('D:/dataset/NeuralAnnot_Release/Human3.6M/data/annotations/Human36M_subject' + str(target_subject) + '_data.json')
    # camera load
    with open('D:/dataset/NeuralAnnot_Release/Human3.6M/data/annotations/Human36M_subject' + str(target_subject) + '_camera.json', 'r') as f:
        cameras = json.load(f)
    # joint coordinate load，这里是[17, 3]的
    with open('D:/dataset/NeuralAnnot_Release/Human3.6M/data/annotations/Human36M_subject' + str(target_subject) + '_joint_3d.json', 'r') as f:
        joints = json.load(f)
    # smplx parameter load
    with open('D:/dataset/NeuralAnnot_Release/Human3.6M/data/annotations/Human36M_subject' + str(target_subject) + '_SMPLX_NeuralAnnot.json', 'r') as f:
        smplx_params = json.load(f)

    with open("D:/dataset/NeuralAnnot_Release/Human3.6M/mp_result/s_%02d_mp.json" % target_subject, 'r', encoding='utf-8') as f:
        mp_data = json.load(f)

    # cam_list = []
    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]    # 通过image_id读取到图片 dict
        # cv2.imshow("img", img)
        # cv2.waitKey()

        subject = img['subject']
        action_idx = img['action_idx']
        subaction_idx = img['subaction_idx']
        frame_idx = img['frame_idx']
        cam_idx = img['cam_idx']

        file_name = img['file_name']    # 's_11_act_02_subact_01_ca_01/s_11_act_02_subact_01_ca_01_000001.jpg'

        # if file_name.split("/")[0] == "s_11_act_02_subact_02_ca_01":
        #     print("缺失文件：", file_name)

        save_file_name_prefix = file_name.split("/")[1].split(".")[0]
        img_filename = file_name.split("/")[1]    # 图片名称

        point2d = mp_data[img_filename]    # 图像坐标
        point2d = np.array(point2d, dtype=np.float32)[:, 0:2]

        # camera parameter
        cam_param = cameras[str(cam_idx)]
        # R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
        R, t, f, c = cam_param['R'], cam_param['t'], cam_param['f'], cam_param['c']

        cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

        f = cam_param["focal"]
        c = cam_param["princpt"]
        camera_intrinsic = np.mat(np.zeros((3, 3), dtype=np.float32))  # 内参矩阵
        camera_intrinsic[0, 0] = f[0]
        camera_intrinsic[1, 1] = f[1]
        camera_intrinsic[0, 2] = c[0]
        camera_intrinsic[1, 2] = c[1]
        camera_intrinsic[2, 2] = np.float32(1)

        # point3d = convert_2d_to_3d(point2d, cam_param["R"], cam_param["t"], camera_intrinsic, height=0)    # point3d [33, 3]
        point3d = pixel_to_world(camera_intrinsic, cam_param["R"], np.asmatrix(cam_param["t"]).T, point2d)    # 用common包里的
        print(point3d)

        # # 将mediapipe的结果整理成smplx格式，并可视化
        # smplx_data = np.zeros([144, 3], dtype=np.float32)
        # smplx_data[JOINT_NAMES.index('left_shoulder')] = point3d[11]
        # smplx_data[JOINT_NAMES.index('right_shoulder')] = point3d[12]
        #
        # smplx_data[JOINT_NAMES.index('left_elbow')] = point3d[13]
        # smplx_data[JOINT_NAMES.index('right_elbow')] = point3d[14]
        #
        # smplx_data[JOINT_NAMES.index('left_wrist')] = point3d[15]
        # smplx_data[JOINT_NAMES.index('right_wrist')] = point3d[16]
        #
        # smplx_data[JOINT_NAMES.index('left_hip')] = point3d[23]
        # smplx_data[JOINT_NAMES.index('right_hip')] = point3d[24]
        #
        # smplx_data[JOINT_NAMES.index('left_knee')] = point3d[25]
        # smplx_data[JOINT_NAMES.index('right_knee')] = point3d[26]
        #
        # smplx_data[JOINT_NAMES.index('left_ankle')] = point3d[27]
        # smplx_data[JOINT_NAMES.index('right_ankle')] = point3d[28]
        #
        # smplx_data[JOINT_NAMES.index('left_heel')] = point3d[29]
        # smplx_data[JOINT_NAMES.index('right_heel')] = point3d[30]
        #
        # xp = smplx_data.T[0].T
        # yp = smplx_data.T[1].T
        # zp = smplx_data.T[2].T
        # plot_line_chart_part(xp, yp, zp, "%s_3d" % (save_file_name_prefix), mode='label')

        # 直接可视化散点图
        xp = point3d.T[0].T
        yp = point3d.T[1].T
        zp = point3d.T[2].T
        plot_point_chart(xp, yp, zp, "%s_3d" % (save_file_name_prefix), mode='label')

        # 保存
        keypoint3d[file_name.split("/")[1]] = point3d.tolist()

    with open("D:/dataset/NeuralAnnot_Release/Human3.6M/world_coord/S%d_world.json" % target_subject, 'w', encoding='utf-8') as f:
        json.dump(keypoint3d, f)


if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    for target_subject in subject_list:
        print("Processing...", target_subject)
        main(target_subject)