import numpy as np
import torch
import smplx
import os.path as osp
import os
from scipy import sparse

# os.environ["PYOPENGL_PLATFORM"] = "egl"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# indexes for mediapipe joints, made by TianLun from PsyAI
MP_index = sparse.lil_matrix((33, 10475))    # len(mesh_cam)
MP_index[0, 8970] = 1
MP_index[1, 9285] = 1
MP_index[2, [9503, 9798]] = 1 / 2
MP_index[3, 1175] = 1
MP_index[4, 9084] = 1
MP_index[5, [10049, 10344]] = 1 / 2
MP_index[6, 2437] = 1
MP_index[7, 5] = 1
MP_index[8, 617] = 1
MP_index[9, 1730] = 1
MP_index[10, 2845] = 1
MP_index[11, [4482, 4504, 4515]] = 1 / 3
MP_index[12, [7215, 7240, 7251]] = 1 / 3
index13 = [4249, 4250, 4275, 4276, 4281, 4282, 4285, 4286, 4319, 4334, 4341, 4342, 4369, 4370, 4377, 4383, 4389, 5628]
MP_index[13, index13] = 1 / len(index13)
index14 = [6993, 6994, 7019, 7020, 7023, 7024, 7027, 7028, 7057, 7070, 7077, 7078, 7105, 7106, 7113, 7119, 7125, 8322]
MP_index[14, index14] = 1 / len(index14)
index15 = [4632, 4672, 4674, 4686, 4703, 4712, 4713, 4715, 4720, 4723, 4820, 4822, 4842, 4849, 4893]
MP_index[15, index15] = 1 / len(index15)
index16 = [7452, 7453, 7454, 7455, 7457, 7458, 7460, 7461, 7462, 7497, 7498, 7557, 7559, 7580, 7584, 7591]
MP_index[16, index16] = 1 / len(index16)
MP_index[17, [4807, 5387]] = 1 / 2
MP_index[18, [7543, 8121]] = 1 / 2
MP_index[19, [4659, 4747]] = 1 / 2
MP_index[20, [7395, 7483]] = 1 / 2
MP_index[21, [4728, 4902]] = 1 / 2
MP_index[22, [7586, 7638]] = 1 / 2
MP_index[23, [4144, 5685]] = 1 / 2
MP_index[24, [6888, 8379]] = 1 / 2
index25 = [3627, 3628, 3631, 3632, 3633, 3634, 3638, 3640, 3641, 3645, 3646, 3647, 3648, 3673, 3674, 3780, 4153]
MP_index[25, index25] = 1 / len(index25)
index26 = [6388, 6389, 6392, 6394, 6399, 6401, 6402, 6406, 6407, 6408, 6409, 6434, 6435, 6538, 6897]
MP_index[26, index26] = 1 / len(index26)
index27 = [5752, 5753, 5754, 5755, 5756, 5757, 5758, 5760, 5761, 5762, 5763, 5764, 5878, 5879, 5880]
MP_index[27, index27] = 1 / len(index27)
index28 = [8446, 8447, 8448, 8449, 8450, 8452, 8454, 8455, 8456, 8457, 8458, 8572, 8573, 8574]
MP_index[28, index28] = 1 / len(index28)
MP_index[29, 8847] = 1
MP_index[30, 8635] = 1
MP_index[31, [5895, 5912]] = 1 / 2
MP_index[32, [8589, 8606]] = 1 / 2


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    z = pixel_coord[:, 2]
    print(z)
    x = (pixel_coord[:, 0] - c[0]) * z / f[0]
    y = (pixel_coord[:, 1] - c[1]) * z / f[1]

    return np.stack((x, y, z), 1)


def world2cam(world_coord, R, t):
    cam_coord = world_coord @ R.T + t.reshape(1, 3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = (cam_coord - t.reshape(1, 3)) @ R
    return world_coord


def Projectonto2D(world_coord, cam_param):
    f = np.array(cam_param['f'], dtype=np.float32)
    c = np.array(cam_param['c'], dtype=np.float32)

    if 'R' in cam_param.keys() and 't' in cam_param.keys():
        R = np.array(cam_param['R'], dtype=np.float32)
        t = np.array(cam_param['t'], dtype=np.float32)
        return cam2pixel(world2cam(world_coord, R, t), f, c)[:, :2]
    else:
        return cam2pixel(world_coord, f, c)[:, :2]    # 没有外参的话，直接当世界坐标为相机坐标


def DataGenerator2to3(smplx_layer, smplx_param, cam_param, KeyIndexs='MP'):
    root_pose, body_pose, shape, trans = smplx_param['root_pose'], smplx_param['body_pose'], smplx_param['shape'], smplx_param['trans']
    root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
    body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
    shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
    trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

    with torch.no_grad():
        output = smplx_layer(betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose, transl=trans)
    ## smplx output models with m as the scaler
    ## cam_param always with mm as the scaler
    mesh_cam = output.vertices[0].numpy() * 1000
    if KeyIndexs == 'MP':
        joint_cam = MP_index @ mesh_cam
    elif KeyIndexs == 'Smplx':
        joint_cam = output.joints[0].numpy() * 1000

    return Projectonto2D(joint_cam, cam_param), joint_cam


# please use the generator
# generated2D, generated3D = DataGenerator2to3(smplx_param, cam_param)