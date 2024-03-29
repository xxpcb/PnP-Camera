**Perspective-n-Point (PnP)** 问题：

- 给定n个3D参考点，以及对应的摄像机图像上的n个2D投影点
- 已知3D点在世界坐标系下的坐标，以及2D点在图像坐标系下的坐标
- 已知摄像机畸变标定参数

**求解**：世界坐标系与摄像机坐标系之间的位姿变换

**用途**：摄像机位姿跟踪（本项目）



**硬件设备**：

- USB摄像头
- 二维码标记图

**软件工具**：

- OpenGL：摄像机位置3D显示，位姿变换
- OpenCV：二维码标记识别，PnP计算
- Eigen：矩阵运算

**效果展示**：

![](https://github.com/xxpcb/PnP-Camera/blob/master/pnp-camera.png)

