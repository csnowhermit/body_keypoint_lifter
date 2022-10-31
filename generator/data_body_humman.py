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
from data_utils import TianLun_index, Projectonto2D

standup_list = ['p000210_a001355',
'p000210_a005341',
'p000210_a005857',
'p000211_a000206',
'p000211_a000227',
'p000211_a001427',
'p000212_a001072',
'p000212_a005829',
'p000213_a000036',
'p000213_a000992',
'p000213_a000993',
'p000213_a005135',
'p000214_a000194',
'p000214_a001402',
'p000214_a003001',
'p000214_a005191',
'p000214_a005882',
'p000215_a000032',
'p000215_a000148',
'p000215_a000279',
'p000215_a005057',
'p000216_a000040',
'p000216_a001023',
'p000216_a001246',
'p000216_a005727',
'p000217_a001243',
'p000219_a001036',
'p000221_a000988',
'p000230_a000150',
'p000230_a001388',
'p000230_a001437',
'p000230_a005371',
'p000231_a000652',
'p000231_a005710',
'p000232_a000701',
'p000232_a001426',
'p000232_a005008',
'p000233_a001229',
'p000233_a001230',
'p000234_a000388',
'p000234_a001231',
'p000234_a003012',
'p000235_a000169',
'p000235_a001428',
'p000236_a000223',
'p000236_a000379',
'p000236_a001410',
'p000237_a000216',
'p000237_a000226',
'p000238_a000075',
'p000238_a000225',
'p000238_a001244',
'p000239_a001425',
'p000239_a005611',
'p000240_a000061',
'p000240_a005833',
'p000241_a000091',
'p000241_a000793',
'p000243_a000213',
'p000243_a000968',
'p000245_a000048',
'p000245_a000204',
'p000245_a006003',
'p000246_a000221',
'p000246_a001082',
'p000247_a000146',
'p000247_a000158',
'p000247_a000224',
'p000248_a000019',
'p000248_a000285',
'p000248_a000898',
'p000249_a001026',
'p000249_a001351',
'p000250_a000170',
'p000250_a000212',
'p000250_a000222',
'p000250_a001362',
'p000251_a000065',
'p000251_a000182',
'p000251_a000551',
'p000251_a005728',
'p000252_a006000',
'p000253_a000077',
'p000253_a000893',
'p000253_a006004']
kinect_list = [0, 1, 4, 5, 6, 9]    # 只处理这几个相机的图片

'''
    预处理humman数据集
'''

def main():
    base_path = "D:/dataset/lifter_dataset/body/humman/pose/"

    keypoint3d = {}  # <img_path, joint3d>
    keypoint2d = {}  # <img_path, joint3d>

    # 初始化smplx
    smplx_layer = smplx.create('./checkpoint/', 'smplx')

    for file in tqdm(os.listdir(base_path)):
        with open(os.path.join(base_path, file), 'r', encoding='utf-8') as f:
            params = json.load(f)

        save_file_name_prefix = file[0: -5]

        # 只保留站着的人
        if save_file_name_prefix in standup_list:
            continue

        # 解析参数
        global_orient = params['global_orient']
        body_pose = params['body_pose']
        trans1 = params['trans1']
        betas = params['betas']

        frame_list = params['frame_list']
        cam_param_list = params['cam_param_list']
        kinect_num = params['kinect_num']

        ## 都用第一张图的pose，分别选10个计算（即10张图对应一个3d火柴人）
        ## 每张图都得用：只用一张图的pose的话，数据集太少了
        for i in range(frame_list):
            curr_global_orient = global_orient[i]

            # 这里指定固定的global参数
            # curr_global_orient = [1.5, 0.0, 0.0]    # 使用这组global参数，3d可视化踩在xy面上，但2d关键点倒了

            curr_body_pose = body_pose[i]
            curr_trans1 = trans1[i]

            curr_global_orient = torch.FloatTensor(curr_global_orient).view(1, 3)
            curr_body_pose = torch.FloatTensor(curr_body_pose).view(-1, 3)[0:21, :]    # [23, 3]，取前21个
            betas = torch.FloatTensor(betas).view(1, -1)    # smplx shape parameter
            curr_trans1 = torch.FloatTensor(curr_trans1).view(1, -1)    # translator

            with torch.no_grad():
                output = smplx_layer(betas=betas,
                                     body_pose=curr_body_pose.view(1, -1),
                                     global_orient=curr_global_orient,
                                     transl=curr_trans1)
            ## smplx output models with m as the scaler
            ## cam_param always with mm as the scaler
            mesh_cam = output.vertices[0].numpy() * 1000
            joint_3d = TianLun_index @ mesh_cam

            # 将3d关键点旋转：之前是踩在墙上的，现在旋转回xy面上
            rotation_angle = [2.0, 0.0, 0.0]
            rotation_angle = torch.FloatTensor(rotation_angle).view(1, 3)
            rotation_angle = rotation_angle.numpy()
            rotation_mat, _ = cv2.Rodrigues(rotation_angle)
            rotation_joint_3d = joint_3d @ rotation_mat

            # # 可视化3d的
            # vis_joints = rotation_joint_3d
            # xp = vis_joints.T[0].T
            # yp = vis_joints.T[1].T
            # zp = vis_joints.T[2].T
            # plot_line_chart_part(xp, yp, zp, "%s_%d_3d" % (save_file_name_prefix, i), mode='label')
            #
            # # pyrender可视化
            # vertex_colors = np.ones([mesh_cam.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]  # [10475, 4]
            # tri_mesh = trimesh.Trimesh(mesh_cam, smplx_layer.faces,
            #                            vertex_colors=vertex_colors)
            #
            # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            #
            # scene = pyrender.Scene()
            # scene.add(mesh)
            #
            # sm = trimesh.creation.uv_sphere(radius=0.005)
            # sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            # tfs = np.tile(np.eye(4), (len(joint_3d), 1, 1))
            # tfs[:, :3, 3] = joint_3d
            # joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            # scene.add(joints_pcl)
            #
            # pyrender.Viewer(scene, use_raymond_lighting=True)

            # 使用不同的相机分别映射到2d，只用指定的几个相机
            for j in kinect_list:
                cam_param = cam_param_list[j]
                joint_2d = Projectonto2D(joint_3d, cam_param)    # 用旋转之前的映射回2d

                # # 可视化
                # xp = joint_2d.T[0].T
                # yp = joint_2d.T[1].T
                # plot_line_chart_part_2d(xp, yp, "%s_%d_%d_2d" % (save_file_name_prefix, i, j), mode='label')

                keypoint2d["%s_%d_%d" % (save_file_name_prefix, i, j)] = joint_2d.tolist()
                keypoint3d["%s_%d_%d" % (save_file_name_prefix, i, j)] = rotation_joint_3d.tolist()    # 保存旋转之后的joint3d
        with open("./humman_mp_%s_2d.json" % save_file_name_prefix, 'w', encoding='utf-8') as f:
            json.dump(keypoint2d, f)
        with open("./humman_mp_%s_3d.json" % save_file_name_prefix, 'w', encoding='utf-8') as f:
            json.dump(keypoint3d, f)






if __name__ == '__main__':
    main()