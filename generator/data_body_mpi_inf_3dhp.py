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

def main():
    keypoint3d = {}    # <img_path, joint3d>
    keypoint2d = {}    # <img_path, joint2d>

    # 1、读取数据
    db = COCO('D:/dataset/NeuralAnnot_Release/MPI-INF-3DHP/data/MPI-INF-3DHP_1k.json')
    with open('D:/dataset/NeuralAnnot_Release/MPI-INF-3DHP/data/MPI-INF-3DHP_joint_3d.json') as f:
        joints = json.load(f)
    with open('D:/dataset/NeuralAnnot_Release/MPI-INF-3DHP/data/MPI-INF-3DHP_camera_1k.json') as f:
        cameras = json.load(f)
    with open('D:/dataset/NeuralAnnot_Release/MPI-INF-3DHP/data/MPI-INF-3DHP_SMPLX_NeuralAnnot.json', 'r') as f:
        smplx_params = json.load(f)

    # 2、初始化smplx
    smplx_layer = smplx.create('./checkpoint/', 'smplx')

    cnt = 0
    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        subject_idx = img['subject_idx']
        seq_idx = img['seq_idx']
        frame_idx = img['frame_idx']
        cam_idx = img['cam_idx']

        # 这里只用指定摄像头的
        if cam_idx in [2, 4, 5, 8]:
            cnt += 1
            img_file_name = img['file_name']
            save_file_prefix = "S%s_Seq%s_cam%s_%s" % (str(subject_idx), str(seq_idx), str(cam_idx), str(img_file_name))

            # camera parameter
            cam_param = cameras[str(subject_idx)][str(seq_idx)][str(cam_idx)]
            R, t, focal, princpt = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'f': focal, 'c': princpt}

            # smplx parameter
            smplx_param = smplx_params[str(subject_idx)][str(seq_idx)][str(frame_idx)]

            generated2D, generated3D = DataGenerator2to3(smplx_layer, smplx_param, cam_param)

            # # 3d关键点可视化：直接画火柴人，可视化看
            # vis_joints = generated3D
            # xp = vis_joints.T[0].T
            # yp = vis_joints.T[1].T
            # zp = vis_joints.T[2].T
            # plot_line_chart_part(xp, yp, zp, "%s_3d" % (img_file_name), mode='label')
            #
            # # 2d关键点可视化
            # xp = generated2D.T[0].T
            # yp = generated2D.T[1].T
            # plot_line_chart_part_2d(xp, yp, "%s_2d" % (img_file_name), mode='label')

            keypoint2d[save_file_prefix] = generated2D.tolist()
            keypoint3d[save_file_prefix] = generated3D.tolist()

    with open("./mpi_inf_3dhp_mp_2d.json", 'w', encoding='utf-8') as f:
        json.dump(keypoint2d, f)
    with open("./mpi_inf_3dhp_mp_3d.json", 'w', encoding='utf-8') as f:
        json.dump(keypoint3d, f)


    # final
    print("统计图片个数：", cnt)
    print("2D关键点个数：", len(list(keypoint2d.keys())))
    print("3D关键点个数：", len(list(keypoint3d.keys())))


if __name__ == '__main__':
    main()