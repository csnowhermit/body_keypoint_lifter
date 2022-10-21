import os
import cv2
import json
import numpy as np
import yaml
import smplx
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import pyrender
import trimesh

from vis_label import plot_line_chart_part, plot_line_chart_part_2d
from data_utils import DataGenerator2to3


'''
    S11中缺失如下序列的标签：s_11_act_02_subact_02_ca_01
    smplx.create()输出的joints为127个，build_layer()为144个
    smplx.create()中参数use_face_contour，为True，返回144个参数（多出的17个点是脸盘的17点）；为False返回127个参数；默认为False
'''

def main(target_subject):
    # target_subject = 11
    # target_action = 2
    # target_subaction = 1
    # target_frame = 0
    # target_cam = 1

    keypoint3d = {}    # <img_path, joint3d>
    keypoint2d = {}    # <img_path, joint3d>

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

    # 2、初始化smplx
    smplx_layer = smplx.create('./checkpoint/', 'smplx')

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
        # print("aid:", aid, subject, action_idx, subaction_idx, frame_idx)

        # 相机参数
        cam_param = cameras[str(cam_idx)]
        # R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)

        # smplx parameter
        smplx_param = smplx_params[str(action_idx)][str(subaction_idx)][str(frame_idx)]

        generated2D, generated3D = DataGenerator2to3(smplx_layer, smplx_param, cam_param)

        # print(generated2D.shape)
        # print(generated3D.shape)

        # # 3d关键点可视化：直接画火柴人，可视化看
        # vis_joints = generated3D
        # xp = vis_joints.T[0].T
        # yp = vis_joints.T[1].T
        # zp = vis_joints.T[2].T
        # plot_line_chart_part(xp, yp, zp, "%s_3d" % (save_file_name_prefix), mode='label')
        #
        # # 2d关键点可视化
        # xp = generated2D.T[0].T
        # yp = generated2D.T[1].T
        # plot_line_chart_part_2d(xp, yp, "%s_2d" % (save_file_name_prefix), mode='label')

        keypoint2d[save_file_name_prefix] = generated2D.tolist()
        keypoint3d[save_file_name_prefix] = generated3D.tolist()

    with open("./%s_mp_2d.json" % target_subject, 'w', encoding='utf-8') as f:
        json.dump(keypoint2d, f)
    with open("./%s_mp_3d.json" % target_subject, 'w', encoding='utf-8') as f:
        json.dump(keypoint3d, f)



if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    for target_subject in subject_list:
        print("Processing...", target_subject)
        main(target_subject)