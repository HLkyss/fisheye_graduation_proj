# fisheye_graduatipn_proj
本科毕设部分代码备份

具体细节太久有的不记得了，以后需要时再整理。
***
<img src="https://github.com/HLkyss/fisheye_graduatipn_proj/assets/69629475/3fb7a41f-04a4-4f22-a1c6-fc586207a2a7" width="1000">
<img src="https://github.com/HLkyss/fisheye_graduatipn_proj/assets/69629475/808ef662-21ec-4908-8c1d-6dda0db4e774" width="1000">
<img src="https://github.com/HLkyss/fisheye_graduatipn_proj/assets/69629475/076a458b-6a19-4860-a8e9-b47fcf712c4d" width="1000">
<img src="https://github.com/HLkyss/fisheye_graduatipn_proj/assets/69629475/ee491bf7-55f0-4525-9404-19ae4cc009e7" width="1000">
<img src="https://github.com/HLkyss/fisheye_graduatipn_proj/assets/69629475/f5f3128b-cd08-452b-bb08-22ae45aa053d" width="1000">

***
useless


欧拉角顺序：
欧拉角来自于本质矩阵，本质矩阵通过opencv的fundfundamental函数获得，因此得到的旋转矩阵应该是opencv相机坐标系下的旋转矩阵，
最后决定顺序如下：
先绕y轴，得到yaw
再绕x轴，得到pitch
最后绕z轴，得到roll
-----------------------------------------
虚拟相机：
要能保证求得鱼眼图像下的鱼眼相机位姿后，在yaw方向减120度就能获得虚拟相机的位姿，
需要：
1.欧拉角顺序，第一个角就是yaw，
2.虚拟相机位置是在鱼眼相机本位上，各水平旋转60度
-----------------------------------------
all_ue统计:
100:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
160:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
260:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
80:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
180:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
280:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
350:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
320:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
330:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
380:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
397:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
10:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
20:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=pitch_ba*M_PI/180;
    euler_angles_ba[2]=-roll_ba*M_PI/180;
32:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
40:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
70:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
75:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
86:
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
