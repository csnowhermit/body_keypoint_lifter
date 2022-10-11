import os

'''
    smplx中144个joint的层级关系
'''

smplx_dict = {}
smplx_dict['body'] = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'jaw', 'head']

# 左胳膊+左手
smplx_dict['left_arm'] = ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']
smplx_dict['left_thumb'] = ['left_wrist', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb']
smplx_dict['left_index'] = ['left_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_index']
smplx_dict['left_middle'] = ['left_wrist', 'left_middle1', 'left_middle2', 'left_middle3', 'left_middle']
smplx_dict['left_ring'] = ['left_wrist', 'left_ring1', 'left_ring2', 'left_ring3', 'left_ring']
smplx_dict['left_pinky'] = ['left_wrist', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky']

# 右胳膊+右手
smplx_dict['right_arm'] = ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
smplx_dict['right_thumb'] = ['right_wrist', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb']
smplx_dict['right_index'] = ['right_wrist', 'right_index1', 'right_index2', 'right_index3', 'right_index']
smplx_dict['right_middle'] = ['right_wrist', 'right_middle1', 'right_middle2', 'right_middle3', 'right_middle']
smplx_dict['right_ring'] = ['right_wrist', 'right_ring1', 'right_ring2', 'right_ring3', 'right_ring']
smplx_dict['right_pinky'] = ['right_wrist', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky']

# 左腿
smplx_dict['left_leg'] = ['pelvis', 'left_hip', 'left_knee', 'left_ankle', 'left_heel', 'left_foot']
smplx_dict['left_big_toe'] = ['left_foot', 'left_big_toe']    # 左大脚趾
smplx_dict['left_small_toe'] = ['left_foot', 'left_small_toe']    # 左小脚趾

# 右腿
smplx_dict['right_leg'] = ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'right_heel', 'right_foot']
smplx_dict['right_big_toe'] = ['right_foot', 'right_big_toe']    # 右大脚趾
smplx_dict['right_small_toe'] = ['right_foot', 'right_small_toe']    # 右小脚趾

# # 头部
# # 鼻子
# smplx_dict['nose'] = ['nose', 'nose1', 'nose2', 'nose3', 'nose4']
# smplx_dict['left_nose'] = ['nose_middle', 'left_nose_1', 'left_nose_2']
# smplx_dict['right_nose'] = ['nose_middle', 'right_nose_1', 'right_nose_2']
#
# # 眉毛
# smplx_dict['left_eye_brow'] = ['left_eye_brow1', 'left_eye_brow2', 'left_eye_brow3', 'left_eye_brow4', 'left_eye_brow5']
# smplx_dict['right_eye_brow'] = ['right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5']
#
# # 眼睛
# smplx_dict['left_eye'] = ['left_eye', 'left_eye1', 'left_eye2', 'left_eye3', 'left_eye4', 'left_eye5', 'left_eye6']
# smplx_dict['left_eye_smplhf'] = ['left_eye_smplhf']
# smplx_dict['right_eye'] = ['right_eye', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6']
# smplx_dict['right_eye_smplhf'] = ['right_eye_smplhf']
#
#
# # 嘴巴
# smplx_dict['mouth_top'] = ['mouth_top']
# smplx_dict['mouth_bottom'] = ['mouth_bottom']
# smplx_dict['lip_top'] = ['lip_top']
# smplx_dict['lip_bottom'] = ['lip_bottom']
#
# smplx_dict['left_mouth'] = ['left_mouth_1', 'left_mouth_2', 'left_mouth_3', 'left_mouth_4', 'left_mouth_5']
# smplx_dict['right_mouth'] = ['right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'right_mouth_4', 'right_mouth_5']
# smplx_dict['left_lip'] = ['left_lip_1', 'left_lip_2', 'left_lip_3']
# smplx_dict['right_lip'] = ['right_lip_1', 'right_lip_2', 'right_lip_3']
#
# # 脸
# smplx_dict['contour_middle'] = ['contour_middle']
# smplx_dict['left_contour'] = ['left_contour_1', 'left_contour_2', 'left_contour_3', 'left_contour_4', 'left_contour_5', 'left_contour_6', 'left_contour_7', 'left_contour_8']
# smplx_dict['right_contour'] = ['right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8']
#
# # 耳朵
# smplx_dict['left_ear'] = ['left_ear']
# smplx_dict['right_ear'] = ['right_ear']

if __name__ == '__main__':
    body = ['body', 'left_arm', 'right_arm', 'left_leg', 'left_big_toe', 'left_small_toe', 'right_leg', 'right_big_toe',
            'right_small_toe']
    # hand = ['left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky']
    hand = ['right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']

    body_joint_list = []
    hand_joint_list = []
    face_joint_list = []

    for key in smplx_dict.keys():
        if key in body:
            for joint in smplx_dict[key]:
                body_joint_list.append(joint)
        elif key in hand:
            for joint in smplx_dict[key]:
                hand_joint_list.append(joint)
        else:
            for joint in smplx_dict[key]:
                face_joint_list.append(joint)

    print("body_joint_list:", len(body_joint_list), body_joint_list)
    print("hand_joint_list:", len(hand_joint_list), hand_joint_list)
    print("face_joint_list:", len(face_joint_list), face_joint_list)

    for b in body:
        for key in smplx_dict[b]:
            body_joint_list.append(key)

    for h in hand:
        for key in smplx_dict[h]:
            hand_joint_list.append(key)

    print("body:", body_joint_list)
    print("hand:", hand_joint_list)

    body_index_list = []
    hand_index_list = []
    face_index_list = []

    from joint_names import JOINT_NAMES
    for item in JOINT_NAMES:
        if item in body_joint_list:
            body_index_list.append(JOINT_NAMES.index(item))
        elif item in hand_joint_list:
            hand_index_list.append(JOINT_NAMES.index(item))
        else:
            face_index_list.append(JOINT_NAMES.index(item))

    print("joint index:")
    print("body_index_list = ", len(list(set(body_index_list))), body_index_list)
    print("hand_index_list = ", len(list(set(hand_index_list))), hand_index_list)
    print("face_index_list = ", len(list(set(face_index_list))), face_index_list)

    # 各个根节点的index
    print("各个根节点的index：")
    print("身体根节点：", JOINT_NAMES.index('pelvis'))
    print("左手手腕：", JOINT_NAMES.index('left_wrist'))
    print("右手手腕：", JOINT_NAMES.index('right_wrist'))

