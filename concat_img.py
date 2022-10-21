
import os
import cv2
import numpy as np
from tqdm import tqdm


'''
    拼接图片成视频
'''
def concat_img_to_video(username, ddate):
    origin_img_path = "D:/dataset/lifter_dataset/inference_data/img_anno/%s" % username
    total_frame = len(os.listdir(origin_img_path))

    writer = cv2.VideoWriter("D:/dataset/lifter_dataset/inference_data/concat_video/pred_%s_%s.mp4" % (username, ddate), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (1600 * 4, 1600), True)

    for i in tqdm(range(total_frame)):
        body = cv2.imread(os.path.join("./output/", "%s_%d_body_3d.png" % (username, i)))
        lefthand = cv2.imread(os.path.join("./output/", "%s_%d_lefthand_3d.png" % (username, i)))
        righthand = cv2.imread(os.path.join("./output/", "%s_%d_righthand_3d.png" % (username, i)))
        origin_img = cv2.imread(os.path.join(origin_img_path, "%s_%d.jpg" % (username, i)))
        origin_img = cv2.resize(origin_img, (1600, 1600))

        img = np.concatenate([origin_img, body, lefthand, righthand], axis=1)

        writer.write(img)
    writer.release()

if __name__ == '__main__':
    concat_img_to_video('fanzhaoxin', '20221021')
