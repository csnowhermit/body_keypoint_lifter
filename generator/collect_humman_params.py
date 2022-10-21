import os
import cv2
import json
import numpy as np
from tqdm import tqdm

from smc_reader import SMCReader

'''
    收集humman数据集的参数
'''

if __name__ == '__main__':
    base_path = "D:/dataset/humman/"
    save_path = "./"
    for file in tqdm(os.listdir(base_path)):
        if str(file).endswith(".smc") is False:
            continue
        reader = SMCReader(os.path.join(base_path, file))

        # 获取pose参数
        smpl_dict = reader.smc['SMPL']
        global_orient = smpl_dict['global_orient'][...].tolist()
        body_pose = smpl_dict['body_pose'][...].tolist()
        transl = smpl_dict['transl'][...].tolist()
        betas = smpl_dict['betas'][...].tolist()

        frame_list = reader.get_smpl_num_frames()  # 帧数
        kinect_num = int(reader.get_num_kinect())  # 相机个数

        cam_param_list = []

        for i in range(kinect_num):
            # 相机外参
            T_cam2world = reader.get_kinect_color_extrinsics(kinect_id=i, homogeneous=True)  # homogeneous=True，返回[4, 4]矩阵；否则返回R、t的dict
            T_world2cam = np.linalg.inv(T_cam2world)    # world-->cam的外参

            R = T_world2cam[0:3, 0:3].tolist()
            T = T_world2cam[0:3, -1].T.tolist()

            # 相机内参
            T_intrinsics = reader.get_kinect_color_intrinsics(kinect_id=i).tolist()
            # print("T_intrinsics:", T_intrinsics.shape, T_intrinsics)

            f = [float(T_intrinsics[0][0]), float(T_intrinsics[1][1])]
            c = [float(T_intrinsics[0][2]), float(T_intrinsics[1][2])]

            cam_param = {'f': f, 'c': c, 'R': R, 'T': T}
            cam_param_list.append(cam_param)

        # print("kinect_num:", kinect_num)

        save_dict = {}
        save_dict['global_orient'] = global_orient
        save_dict['body_pose'] = body_pose
        save_dict['trans1'] = transl
        save_dict['betas'] = betas

        save_dict['frame_list'] = int(frame_list)
        save_dict['cam_param_list'] = cam_param_list
        save_dict['kinect_num'] = int(kinect_num)

        with open(os.path.join(save_path, "%s.json" % file[0: -4]), 'w', encoding='utf-8') as f:
            json.dump(save_dict, f)


