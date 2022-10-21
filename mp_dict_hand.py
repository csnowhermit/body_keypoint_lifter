
########################################################
## mediapipe中人手的顺序
MP_HAND_JOINT_NAMES = [
    'wrist',
    'thumb_cmc',
    'thumb_mcp',
    'thumb_ip',
    'thumb_tip',
    'index_finger_mcp',
    'index_finger_pip',
    'index_finger_dip',
    'index_finger_tip',
    'middle_finger_mcp',
    'middle_finger_pip',
    'middle_finger_dip',
    'middle_finger_tip',
    'ring_finger_mcp',
    'ring_finger_pip',
    'ring_finger_dip',
    'ring_finger_tip',
    'pinky_finger_mcp',
    'pinky_finger_pip',
    'pinky_finger_dip',
    'pinky_finger_tip'
]

# 人手的连线
mp_hand_dict = {}
mp_hand_dict['thumb'] = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip']
mp_hand_dict['index'] = ['wrist', 'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip']
mp_hand_dict['middle'] = ['wrist', 'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip']
mp_hand_dict['ring'] = ['wrist', 'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip']
mp_hand_dict['pinky'] = ['wrist', 'pinky_finger_mcp', 'pinky_finger_pip', 'pinky_finger_dip', 'pinky_finger_tip']
mp_hand_dict['finger_root'] = ['index_finger_mcp', 'middle_finger_mcp', 'ring_finger_mcp', 'pinky_finger_mcp']


########################################################
## smplx中人手的顺序
SMPLX_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]


# 按照smplx的格式可视化
smplx_hand_dict = {}
# 左手
smplx_hand_dict['left_thumb'] = ['left_wrist', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb']
smplx_hand_dict['left_index'] = ['left_wrist', 'left_index1', 'left_index2', 'left_index3', 'left_index']
smplx_hand_dict['left_middle'] = ['left_wrist', 'left_middle1', 'left_middle2', 'left_middle3', 'left_middle']
smplx_hand_dict['left_ring'] = ['left_wrist', 'left_ring1', 'left_ring2', 'left_ring3', 'left_ring']
smplx_hand_dict['left_pinky'] = ['left_wrist', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky']

smplx_hand_dict['left_hand'] = ['left_index1', 'left_middle1', 'left_ring1', 'left_pinky1']

# 右胳膊+右手
smplx_hand_dict['right_thumb'] = ['right_wrist', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb']
smplx_hand_dict['right_index'] = ['right_wrist', 'right_index1', 'right_index2', 'right_index3', 'right_index']
smplx_hand_dict['right_middle'] = ['right_wrist', 'right_middle1', 'right_middle2', 'right_middle3', 'right_middle']
smplx_hand_dict['right_ring'] = ['right_wrist', 'right_ring1', 'right_ring2', 'right_ring3', 'right_ring']
smplx_hand_dict['right_pinky'] = ['right_wrist', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky']

smplx_hand_dict['right_hand'] = ['right_index1', 'right_middle1', 'right_ring1', 'right_pinky1']