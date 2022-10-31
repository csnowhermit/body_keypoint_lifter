# body_keypoint_lifter

​	人体2D关键点-->3D关键点。



## 1、简介

​	该项目将人体图像坐标2D关键点lift成世界坐标3D关键点，供后续IK（Inverse Kinematics）算法使用。

## 2、模型

​	目前已实现了两个模型：

（1）model_single_frame.py，单帧模型，输入维度[33, 2]，输出维度[23, 3]；

​	可能存在的抖动问题从信号处理的角度解决；

（2）model_temporal.py，基于Transformer的时序模型，从模型上直接解决抖动问题；

## 3、数据集

​	数据集标签参考来源：https://github.com/mks0601/NeuralAnnot_RELEASE

​	采用human3.6m或gta_human等数据集，数据处理流程：

标签中的pose参数+smplx --> joint3d --> 结合相机参数，投影得到2D关键点；

也可直接采用mediapipe或openpose检测图片，得到2D关键点；

数据集处理的脚本：generator/data_generator_human36m.py

``` python
generator/data_generator_human36m.py
generator/data_generator_human36m_smplx_creator.py
```

## 4、开源算法在不同数据集上的指标

| 编号 | 开源算法  |  开源数据集  |  MPJPE   | PA-MPJPE | 备注 |
| :--: | :-------: | :----------: | :------: | :------: | :--: |
|  1   | mediapipe |  human3.6m   | 1434.12  |  83.14   |      |
|  2   | mediapipe |     3DPW     | 4110.20  |  89.53   |      |
|  3   | mediapipe |    humman    | 2213.53  |  103.09  |      |
|  4   | mediapipe | MPI_INF_3DHP | 1708.89  |  104.48  |      |
|  5   | mediapipe |    MSCOCO    | 29521.84 |  122.44  |      |
|  6   | mediapipe |     MPII     | 32846.84 |  112.21  |      |

## 5、我们算法的指标

| 编号 |     部位     | MPJPE | PA-MPJPE | 备注 |
| :--: | :----------: | :---: | :------: | :--: |
|  1   |     左手     | 18.78 |   7.72   |      |
|  2   |     右手     | 18.59 |   7.67   |      |
|  3   | 人体（全身） | 9.97  |   8.02   |      |
|  4   | 人体（半身） | 11.50 |   7.98   |      |

