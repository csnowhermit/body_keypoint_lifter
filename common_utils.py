import numpy as np

def split_sorted_list(frame_list):
    frame_list = sorted(frame_list)

    part_list = []
    tmp = []
    for i in range(len(frame_list) - 1):
        curr_frame_id = frame_list[i]
        if i == 0 or len(tmp) == 0:
            tmp.append(curr_frame_id)

        next_frame_id = frame_list[i + 1]
        if next_frame_id - curr_frame_id == 1:
            tmp.append(next_frame_id)

            if i + 1 == len(frame_list) - 1:
                part_list.append(tmp)
        else:
            part_list.append(tmp)
            tmp = []  # 新开一个序列，

    return part_list