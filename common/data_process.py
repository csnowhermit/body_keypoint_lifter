import os
import json
import shutil
import random
import numpy as np
from tqdm import tqdm

from common.common_utils import split_sorted_list

'''
    提前处理好数据集：前面缺的补0，中间的0值直接用
'''
def pre_process_no_padding():
    mode = "val"
    keypoint_2d_path = "D:/dataset/NeuralAnnot_Release/Human3.6M/data_%s/" % mode
    keypoint_3d_path = "D:/dataset/NeuralAnnot_Release/Human3.6M/keypoint3d_127/"

    num_frame = 30

    data_list = []
    label_list = []

    for file in tqdm(os.listdir(keypoint_2d_path)):
        prefix = file[0: -8]
        cam_idx = str(file).split("_")[-2]

        if cam_idx == "01" or cam_idx == "03":    # 跳过左后方和右后方的片段
            continue

        with open(os.path.join(keypoint_2d_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        label = np.load(os.path.join(keypoint_3d_path, file[0:-8] + ".npy"))
        # print("curr_file:", file)

        img_list = [int(str(k).split(".")[0].split("_")[-1]) for k in data.keys()]
        img_list = sorted(img_list)

        np_data = []
        for img_file in img_list:
            # if len(data["%08d.jpeg" % img_file]) > 0:
            if len(data["%s_%06d.jpg" % (prefix, img_file)]) > 0:
                curr_frame_data = np.array(data["%s_%06d.jpg" % (prefix, img_file)], dtype=np.float32)[:, 0:2]  # 只取xy
                # 之后还原到原图，再进行归一化

                # # 转为绝对坐标
                # curr_frame_data[:, 0] = curr_frame_data[:, 0] * w
                # curr_frame_data[:, 1] = curr_frame_data[:, 1] * h

                # 归一化
                min_width = np.min(curr_frame_data[:, 0])
                min_height = np.min(curr_frame_data[:, 1])

                curr_width = np.max(curr_frame_data[:, 0]) - np.min(curr_frame_data[:, 0])
                curr_height = np.max(curr_frame_data[:, 1]) - np.min(curr_frame_data[:, 1])

                if float(curr_width) == 0.0:
                    curr_width = 1.0
                if float(curr_height) == 0.0:
                    curr_height = 1.0

                curr_frame_data[:, 0] = (curr_frame_data[:, 0] - min_width) / curr_width
                curr_frame_data[:, 1] = (curr_frame_data[:, 1] - min_height) / curr_height
            else:
                curr_frame_data = np.zeros([33, 2], dtype=np.float32)
            np_data.append(curr_frame_data)

        np_data = np.array(np_data, dtype=np.float32)

        # 如果序列值小于帧数，则全部用；否则从[self.num_frame, len(img_list)-1]随机，拿取30帧
        if len(img_list) < num_frame:
            end_index = len(img_list) - 1  # 数组下标是从0开始的
            # 这时只有一个end_index
            # curr_data = collect_input_data(np_data, num_frame, end_index)
            curr_data = []
            detfect = np_data.shape[0] - len(data)
            for j in range(detfect):
                curr_data.append(np.zeros([33, 2], dtype=np.float32))
            for j in range(np_data.shape[0]):
                curr_data.append(np_data[j])

            curr_data = np.array(curr_data, dtype=np.float32)

            curr_label = label[end_index]

            data_list.append(curr_data)
            label_list.append(curr_label)
        else:
            # 从第30帧开始，也包括30帧，对应的坐标是29
            # 从30帧开始，一直到序列末尾，使用
            # end_index = random.randint(self.num_frame - 1, len(img_list) - 1)  # random.randint()两边都是闭区间
            for end_index in range(num_frame, np_data.shape[0]):
                start_index = end_index - num_frame
                # result = data[start_index: end_index]
                curr_data = np_data[start_index: end_index, :, :]

                curr_label = label[end_index-1]

                data_list.append(curr_data)
                label_list.append(curr_label)
        # print("第一个序列处理完毕：", len(data_list), len(label_list))    # 校验数据处理对不对
        # break
    data_list = np.array(data_list, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.float32)

    print(data_list.shape)
    print(label_list.shape)
    np.save("./data/data_%s.npy" % mode, data_list)
    np.save("./data/label_%s.npy" % mode, label_list)


'''
    从S1_keypoint3d_smplx_creator127.json文件整理到[n, 127, 3]的npy文件
'''
def collect_json_to_npy():
    base_path = "D:/dataset/NeuralAnnot_Release/Human3.6M/"
    save_path = "D:/dataset/NeuralAnnot_Release/Human3.6M/keypoint3d_127b/"

    for file in os.listdir(base_path):
        if file.endswith("_build_layer127.json") is True:
            with open(os.path.join(base_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(file)
            seq_img_dict = {}  # <视频名称, 图片帧号>

            # 1.先整理<seq_name, img_list>
            for k in tqdm(data.keys()):
                seq_name = str(k)[0: -11]  # 拿出seq_name
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
                        img_name = "%s_%06d.jpg" % (seq, j)
                        res = data[img_name]
                        res = np.array(res, dtype=np.float32)
                        curr_seq.append(res)

                    curr_seq = np.array(curr_seq, dtype=np.float32)
                    # print("curr_seq.shape:", curr_seq.shape)
                    np.save(os.path.join(save_path, "%s.npy" % (str(seq))), curr_seq)
            print("有间断的视频序列：", result)


'''
    human3.6m数据集制作单帧训练数据
'''
def data_pre_process_single_frame():
    mode = "train"
    keypoint_2d_path = "D:/dataset/NeuralAnnot_Release/Human3.6M/data_%s/" % mode
    keypoint_3d_path = "D:/dataset/NeuralAnnot_Release/Human3.6M/keypoint3d_127b/"

    data_list = []
    label_list = []

    for file in tqdm(os.listdir(keypoint_2d_path)):
        prefix = file[0: -8]
        cam_idx = str(file).split("_")[-2]

        # if cam_idx == "01" or cam_idx == "03":    # 跳过左后方和右后方的片段
        #     continue

        with open(os.path.join(keypoint_2d_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        label = np.load(os.path.join(keypoint_3d_path, file[0:-8] + ".npy"))
        # print("curr_file:", file)

        img_list = [int(str(k).split(".")[0].split("_")[-1]) for k in data.keys()]
        img_list = sorted(img_list)

        np_data = []
        for img_file in img_list:
            # if len(data["%08d.jpeg" % img_file]) > 0:
            if len(data["%s_%06d.jpg" % (prefix, img_file)]) > 0:
                curr_frame_data = np.array(data["%s_%06d.jpg" % (prefix, img_file)], dtype=np.float32)[:, 0:2]  # 只取xy
                # 之后还原到原图，再进行归一化

                # # 转为绝对坐标
                # curr_frame_data[:, 0] = curr_frame_data[:, 0] * w
                # curr_frame_data[:, 1] = curr_frame_data[:, 1] * h

                # 归一化
                min_width = np.min(curr_frame_data[:, 0])
                min_height = np.min(curr_frame_data[:, 1])

                curr_width = np.max(curr_frame_data[:, 0]) - np.min(curr_frame_data[:, 0])
                curr_height = np.max(curr_frame_data[:, 1]) - np.min(curr_frame_data[:, 1])

                if float(curr_width) == 0.0:
                    curr_width = 1.0
                if float(curr_height) == 0.0:
                    curr_height = 1.0

                curr_frame_data[:, 0] = (curr_frame_data[:, 0] - min_width) / curr_width
                curr_frame_data[:, 1] = (curr_frame_data[:, 1] - min_height) / curr_height
            else:
                curr_frame_data = np.zeros([33, 2], dtype=np.float32)
            np_data.append(curr_frame_data)

        np_data = np.array(np_data, dtype=np.float32)

        for j in range(np_data.shape[0]):
            curr_data = np_data[j]
            curr_label = label[j]
            # print(curr_data.shape, curr_label.shape)

            data_list.append(curr_data)
            label_list.append(curr_label)
    data_list = np.array(data_list, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.float32)

    print(data_list.shape)
    print(label_list.shape)
    np.save("./data/data_%s.npy" % mode, data_list)
    np.save("./data/label_%s.npy" % mode, label_list)

if __name__ == '__main__':
    # pre_process_no_padding()
    collect_json_to_npy()
    data_pre_process_single_frame()
