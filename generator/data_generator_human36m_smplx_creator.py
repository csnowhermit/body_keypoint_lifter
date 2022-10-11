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

from common.vis_label_utils import plot_line_chart_part, plot_line_chart_part_2d


'''
    S11中缺失如下序列的标签：s_11_act_02_subact_02_ca_01
    smplx.create()输出的joints为127个，build_layer()为144个
    smplx.create()中参数use_face_contour，为True，返回144个参数（多出的17个点是脸盘的17点）；为False返回127个参数；默认为False
'''

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


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

    # 2、初始化smplx
    body_model = smplx.create("../checkpoint/", 'smplx')    # use_face_contour为True，返回144个点；否则返回127个点

    print(body_model)

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
        R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)

        # smplx parameter
        smplx_param = smplx_params[str(action_idx)][str(subaction_idx)][str(frame_idx)]
        root_pose, body_pose, shape, trans = smplx_param['root_pose'], smplx_param['body_pose'], smplx_param['shape'], smplx_param['trans']

        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation
        root_pose = root_pose.numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        root_pose = torch.from_numpy(root_pose).view(1, 3)

        # get mesh and joint coordinates
        with torch.no_grad():
            final_body_model_output = body_model(betas=shape, body_pose=body_pose.view(1,-1), global_orient=root_pose, transl=trans)

        vertices = final_body_model_output.vertices.detach().cpu().numpy().squeeze()    # [10475, 3]
        # joints = final_body_model_output.joints.detach().cpu().numpy().squeeze()    # [127, 3]
        joints = final_body_model_output.get('joints')  # [1, 127, 3]

        # print("vertices.shape:", vertices.shape)
        # print("joints.shape:", joints.shape)

        # # pyrender可视化
        # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]  # [10475, 4]
        # tri_mesh = trimesh.Trimesh(vertices, body_model.faces,
        #                            vertex_colors=vertex_colors)
        #
        # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        #
        # scene = pyrender.Scene()
        # scene.add(mesh)
        #
        # sm = trimesh.creation.uv_sphere(radius=0.005)
        # sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        # tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        # tfs[:, :3, 3] = joints
        # joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        # scene.add(joints_pcl)
        #
        # pyrender.Viewer(scene, use_raymond_lighting=True)

        # # 3d关键点可视化：直接画火柴人，可视化看
        # vis_joints = joints.detach().cpu().numpy()[0]
        # xp = vis_joints.T[0].T
        # yp = vis_joints.T[1].T
        # zp = vis_joints.T[2].T
        # plot_line_chart_part(xp, yp, zp, "%s_3d" % (save_file_name_prefix), mode='label')
        #
        # # 映射到2d关键点（第一种方法）
        # # 世界坐标-->相机坐标-->图像坐标
        # joint_cam = world2cam(vis_joints, R, t)  # [144, 3]
        # proj_joints = cam2pixel(joint_cam, f, c)[:, :2]  # [144, 2]
        #
        # # 2d关键点可视化
        # xp = proj_joints.T[0].T
        # yp = proj_joints.T[1].T
        # plot_line_chart_part_2d(xp, yp, "%s_2d" % (save_file_name_prefix), mode='label')

        # # 映射到2d关键点（第二种方法）
        # camera_data = build_cam_proj(camera_cfg, dtype=torch.float32)  # 配置文件直接从yml里读
        # projection = camera_data['camera']
        # camera_param_dim = camera_data['dim']
        # camera_mean = camera_data['mean']
        # camera_scale_func = camera_data['scale_func']
        #
        # # Extract the camera parameters estimated by the body only image, to compute the projected 2d joints
        # # camera_param = [0.0, 1.0, 1.0]
        # # camera_param = [camera_param[2], camera_param[0], camera_param[1]]
        # camera_params = torch.tensor(trans).unsqueeze(0)  # 模型预测的相机参数，从标签中读取
        # scale = camera_params[:, 0].view(-1, 1)
        # translation = camera_params[:, 1:3]
        # # Pass the predicted scale through exp() to make sure that the
        # # scale values are always positive
        # scale = camera_scale_func(scale)
        #
        # # Project the joints on the image plane (projected joints tensor value are very small, may be normalized according to images' shape)
        # proj_joints = projection(
        #     joints,  # 3d joints
        #     scale=scale, translation=translation)
        #
        # # # 2d关键点可视化
        # proj_joints = proj_joints.detach().cpu().numpy().squeeze()  # [144, 2]
        # # xp = proj_joints.T[0].T
        # # yp = proj_joints.T[1].T
        # # plot_line_chart_part_2d(xp, yp, "%s_%s" % (str(seq_id).replace("|", "_"), str(img_id)), mode='label')


        # 保存
        joints = joints.detach().cpu().numpy()[0].tolist()
        # keypoint3d.append(joints)
        # print(type(joints), joints)
        keypoint3d[file_name.split("/")[1]] = joints

    with open("D:/dataset/NeuralAnnot_Release/Human3.6M/S%d_keypoint3d_smplx_creator127.json" % target_subject, 'w', encoding='utf-8') as f:
        json.dump(keypoint3d, f)


if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    for target_subject in subject_list:
        print("Processing...", target_subject)
        main(target_subject)
