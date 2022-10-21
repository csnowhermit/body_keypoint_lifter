import os
import json
import numpy as np
import cv2
import torch
import smplx
from tqdm import tqdm
from pycocotools.coco import COCO

import pyrender
import trimesh

from vis_label import plot_line_chart_part, plot_line_chart_part_2d
from data_utils import cam2pixel


def main():
    mode = 'train'
    db = COCO('D:/dataset/NeuralAnnot_Release/MSCOCO/annotations/coco_wholebody_%s_v1.0.json' % mode)
    # smplx parameter load
    with open('D:/dataset/NeuralAnnot_Release/MSCOCO/annotations/MSCOCO_train_SMPLX_all_NeuralAnnot.json', 'r') as f:
        smplx_params = json.load(f)

    keypoint2d = {}
    keypoint3d = {}

    smplx_layer = smplx.create('./checkpoint/', 'smplx', use_pca=False)
    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]

        # image path and bbox
        img_path = img['file_name']    # '000000537548.jpg'
        save_file_name_prefix = img_path[0: -4]

        # smplx parameter
        if str(aid) not in smplx_params.keys():
            continue
        smplx_param = smplx_params[str(aid)]
        root_pose, body_pose, shape, trans = smplx_param['smplx_param']['root_pose'], smplx_param['smplx_param']['body_pose'], smplx_param['smplx_param']['shape'], smplx_param['smplx_param']['trans']
        lhand_pose, rhand_pose, jaw_pose, expr = smplx_param['smplx_param']['lhand_pose'], smplx_param['smplx_param']['rhand_pose'], smplx_param['smplx_param']['jaw_pose'], smplx_param['smplx_param']['expr']
        root_pose = torch.FloatTensor(root_pose).view(1, -1)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPL shape parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector
        lhand_pose = torch.FloatTensor(lhand_pose).view(1, -1)
        rhand_pose = torch.FloatTensor(rhand_pose).view(1, -1)
        jaw_pose = torch.FloatTensor(jaw_pose).view(1, -1)
        expr = torch.FloatTensor(expr).view(1, -1)

        # get mesh and joint coordinates
        with torch.no_grad():
            output = smplx_layer(betas=shape,
                                 body_pose=body_pose.view(1, -1),
                                 global_orient=root_pose,
                                 transl=trans,
                                 left_hand_pose=lhand_pose,
                                 right_hand_pose=rhand_pose,
                                 jaw_pose=jaw_pose,
                                 expression=expr)
        mesh_cam = output.vertices[0].numpy()
        joint3d = output.joints[0].numpy()    # [127, 3]

        # pyrender 可视化
        # pyrender可视化
        vertex_colors = np.ones([mesh_cam.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]  # [10475, 4]
        tri_mesh = trimesh.Trimesh(mesh_cam, smplx_layer.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joint3d), 1, 1))
        tfs[:, :3, 3] = joint3d
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)


        # 映射回2d关键点

        # 有内参没外参
        f = np.array(smplx_param['cam_param']['focal'], dtype=np.float32)
        c = np.array(smplx_param['cam_param']['princpt'], dtype=np.float32)

        joint2d = cam2pixel(joint3d, f, c)[:, :2]

        # 这里可视化按照smplx的格式画
        # 3d关键点可视化：直接画火柴人，可视化看
        # vis_joints = joint3d
        # xp = vis_joints.T[0].T
        # yp = vis_joints.T[1].T
        # zp = vis_joints.T[2].T
        # plot_line_chart_part(xp, yp, zp, "%s_3d" % (save_file_name_prefix), mode='label')
        #
        # # 2d关键点可视化
        # xp = joint2d.T[0].T
        # yp = joint2d.T[1].T
        # plot_line_chart_part_2d(xp, yp, "%s_2d" % (save_file_name_prefix), mode='label')

        keypoint2d[save_file_name_prefix] = joint2d.tolist()
        keypoint3d[save_file_name_prefix] = joint3d.tolist()

    with open("./hand_coco_%s_smplx_2d.json" % mode, 'w', encoding='utf-8') as f:
        json.dump(keypoint2d, f)
    with open("./hand_coco_%s_smplx_3d.json" % mode, 'w', encoding='utf-8') as f:
        json.dump(keypoint3d, f)


if __name__ == '__main__':
    main()