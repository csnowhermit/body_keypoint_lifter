

MP_BODY_JOINT_NAMES = ['nose',
                  'left_eye_inner',
                  'left_eye',
                  'left_eye_outer',
                  'right_eye_inner',
                  'right_eye',
                  'right_eye_outer',
                  'left_ear',
                  'right_ear',
                  'mouth_left',
                  'mouth_right',
                  'left_shoulder',
                  'right_shoulder',
                  'left_elbow',
                  'right_elbow',
                  'left_wrist',
                       'right_wrist',
                       'left_pinky',
                       'right_pinky',
                       'left_index',
                       'right_index',
                       'left_thumb',
                       'right_thumb',
                       'left_hip',
                       'right_hip',
                       'left_knee',
                       'right_knee',
                       'left_ankle',
                       'right_ankle',
                       'left_heel',
                       'right_heel',
                       'left_foot_index',
                       'right_foot_index'
                       ]

mp_body_dict = {}
# 横着连线
mp_body_dict['left_arm'] = ['left_shoulder', 'left_elbow', 'left_wrist']
mp_body_dict['right_arm'] = ['right_shoulder', 'right_elbow', 'right_wrist']

# mp_smplx_dict['left_pinky'] = ['left_wrist', 'left_pinky']
# mp_smplx_dict['left_index'] = ['left_wrist', 'left_index']
# mp_smplx_dict['left_thumb'] = ['left_wrist', 'left_thumb']
#
# mp_smplx_dict['right_pinky'] = ['right_wrist', 'right_pinky']
# mp_smplx_dict['right_index'] = ['right_wrist', 'right_index']
# mp_smplx_dict['right_thumb'] = ['right_wrist', 'right_thumb']

mp_body_dict['left_leg'] = ['left_hip', 'left_knee', 'left_ankle', 'left_heel']
mp_body_dict['right_leg'] = ['right_hip', 'right_knee', 'right_ankle', 'right_heel']

mp_body_dict['left_foot_index'] = ['left_heel', 'left_foot_index']
mp_body_dict['right_foot_index'] = ['right_heel', 'right_foot_index']

mp_body_dict['left_ankle_index'] = ['left_ankle', 'left_foot_index']
mp_body_dict['right_ankle_index'] = ['right_ankle', 'right_foot_index']


# 上半身连线
mp_body_dict['left'] = ['left_shoulder', 'left_hip']
mp_body_dict['right'] = ['right_shoulder', 'right_hip']

# 横着连线
mp_body_dict['shoulder'] = ['left_shoulder', 'right_shoulder']
mp_body_dict['hip'] = ['left_hip', 'right_hip']

# # 头部
# mp_smplx_dict['mouth'] = ['mouth_left', 'mouth_right']
# mp_smplx_dict['left_ear'] = ['left_ear']
# mp_smplx_dict['right_ear'] = ['right_ear']
# mp_smplx_dict['nose'] = ['nose']
# mp_smplx_dict['left_eye'] = ['left_eye_inner', 'left_eye', 'left_eye_outer']
# mp_smplx_dict['right_eye'] = ['right_eye_inner', 'right_eye', 'right_eye_outer']


## 半身可视化
mp_halfbody_dict = {}
# 横着连线
mp_halfbody_dict['left_arm'] = ['left_shoulder', 'left_elbow', 'left_wrist']
mp_halfbody_dict['right_arm'] = ['right_shoulder', 'right_elbow', 'right_wrist']

# mp_smplx_dict['left_pinky'] = ['left_wrist', 'left_pinky']
# mp_smplx_dict['left_index'] = ['left_wrist', 'left_index']
# mp_smplx_dict['left_thumb'] = ['left_wrist', 'left_thumb']
#
# mp_smplx_dict['right_pinky'] = ['right_wrist', 'right_pinky']
# mp_smplx_dict['right_index'] = ['right_wrist', 'right_index']
# mp_smplx_dict['right_thumb'] = ['right_wrist', 'right_thumb']


# 横着连线
mp_halfbody_dict['shoulder'] = ['left_shoulder', 'right_shoulder']

# # 头部
# mp_smplx_dict['mouth'] = ['mouth_left', 'mouth_right']
# mp_smplx_dict['left_ear'] = ['left_ear']
# mp_smplx_dict['right_ear'] = ['right_ear']
# mp_smplx_dict['nose'] = ['nose']
# mp_smplx_dict['left_eye'] = ['left_eye_inner', 'left_eye', 'left_eye_outer']
# mp_smplx_dict['right_eye'] = ['right_eye_inner', 'right_eye', 'right_eye_outer']
