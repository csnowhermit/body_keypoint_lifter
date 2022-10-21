import os
import json
import numpy as np
from tqdm import tqdm

from common_utils import split_sorted_list

'''
    从S1_keypoint3d_smplx_creator127.json文件整理到[n, 127, 3]的npy文件
'''
def collect_json_to_npy_human36m():
    base_path = "D:/dataset/lifter_dataset/body/human3.6m/json/"
    save_path = "D:/dataset/lifter_dataset/body/human3.6m/mp_3d/"

    for file in os.listdir(base_path):
        if file.endswith("_mp_3d.json") is True:
            with open(os.path.join(base_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(file)
            seq_img_dict = {}  # <视频名称, 图片帧号>

            # 1.先整理<seq_name, img_list>
            for k in tqdm(data.keys()):
                seq_name = str(k)[0: -7]  # 拿出seq_name（这里k没有后缀）
                if seq_name in seq_img_dict.keys():
                    tmp = seq_img_dict[seq_name]
                    tmp.append(int(k.split(".")[0].split("_")[-1]))
                    seq_img_dict[seq_name] = tmp
                else:
                    tmp = []
                    tmp.append(int(k.split(".")[0].split("_")[-1]))
                    seq_img_dict[seq_name] = tmp

            # 2.对于每个视频序列，校验其连续性
            result = []
            for seq in tqdm(seq_img_dict.keys()):
                img_list = seq_img_dict[seq]
                img_list = sorted(img_list)
                part_list = split_sorted_list(img_list)
                if len(part_list) > 1:
                    result.append(seq)

                # 拼个img_name，在data中拿数据
                if len(part_list) == 1:
                    curr_seq = []
                    for j in img_list:
                        img_name = "%s_%06d" % (seq, j)    # 这里key没有后缀
                        res = data[img_name]
                        res = np.array(res, dtype=np.float32)
                        curr_seq.append(res)

                    curr_seq = np.array(curr_seq, dtype=np.float32)
                    # print("curr_seq.shape:", curr_seq.shape)
                    np.save(os.path.join(save_path, "%s.npy" % (str(seq))), curr_seq)
            print("有间断的视频序列：", result)

'''
    处理mpi_inf_3dhp数据集
'''
def collect_json_to_npy_mpi_inf_3dhp():
    base_path = "D:/dataset/lifter_dataset/body/mpi_inf_3dhp/json/"
    save_path = "D:/dataset/lifter_dataset/body/mpi_inf_3dhp/mp_2d/"

    for file in os.listdir(base_path):
        if file.endswith("_mp_2d.json") is True:
            with open(os.path.join(base_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(file)
            seq_img_dict = {}  # <视频名称, 图片帧号>

            # 1.先整理<seq_name, img_list>
            for k in tqdm(data.keys()):
                seq_name = str(k)[0: -12]  # 拿出seq_name
                if seq_name in seq_img_dict.keys():
                    tmp = seq_img_dict[seq_name]
                    tmp.append(int(k.split(".")[0].split("_")[-1]))
                    seq_img_dict[seq_name] = tmp
                else:
                    tmp = []
                    tmp.append(int(k.split(".")[0].split("_")[-1]))
                    seq_img_dict[seq_name] = tmp

            # 2.对于每个视频序列，校验其连续性
            result = []
            for seq in tqdm(seq_img_dict.keys()):
                cam_id = str(seq).split("_")[2][-1]    # 相机id

                img_list = seq_img_dict[seq]
                img_list = sorted(img_list)
                part_list = split_sorted_list(img_list)
                if len(part_list) > 1:
                    result.append(seq)

                # 拼个img_name，在data中拿数据
                if len(part_list) == 1:
                    curr_seq = []
                    for j in img_list:
                        img_name = "%s%s_%06d.jpg" % (seq, cam_id, j)    # 这里key没有后缀
                        res = data[img_name]
                        res = np.array(res, dtype=np.float32)
                        curr_seq.append(res)

                    curr_seq = np.array(curr_seq, dtype=np.float32)
                    # print("curr_seq.shape:", curr_seq.shape)
                    np.save(os.path.join(save_path, "%s.npy" % (str(seq))), curr_seq)
            print("有间断的视频序列：", result)


'''
    做单帧的训练数据集
'''
def data_pre_process_single_frame():
    mode = "train"
    keypoint_2d_path = "D:/dataset/lifter_dataset/body/workdata/data_%s/" % mode
    keypoint_3d_path = "D:/dataset/lifter_dataset/body/workdata/label/"

    data_list = []
    label_list = []

    for file in tqdm(os.listdir(keypoint_2d_path)):
        data = np.load(os.path.join(keypoint_2d_path, file))
        label = np.load(os.path.join(keypoint_3d_path, file))
        # print(data.shape, label.shape)

        for i in range(data.shape[0]):
            curr_data = data[i]
            curr_label = label[i]

            data_list.append(curr_data)
            label_list.append(curr_label)

    data_list = np.array(data_list, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.float32)

    print(data_list.shape)
    print(label_list.shape)
    np.save("./data/data_%s.npy" % mode, data_list)
    np.save("./data/label_%s.npy" % mode, label_list)


if __name__ == '__main__':
    # collect_json_to_npy_human36m()
    # collect_json_to_npy_mpi_inf_3dhp()
    data_pre_process_single_frame()


