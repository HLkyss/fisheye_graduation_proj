//使用2D-2D的特征匹配估计相机运动
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <string>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace cv;

// 声明一些全局变量
cv::Ptr<cv::StereoSGBM> SGBM = cv::StereoSGBM::create();

void virtual_pix_to_virtual_cam(const Eigen::Vector2d &pt_virtual_2d,const cv::Mat &K_virtual,Eigen::Vector3d &pt_3d);
void fish_cam_to_fish_pix(const Eigen::Vector3d &pt_3d_fish,const cv::Mat &K,Eigen::Vector2d &fish_pix, const double &alpha, const double &ksi);
void SGBMUpdate(int pos, void* data);
void SGBMStart();
void euler_angle(Mat &R);



int main(int argc, char **argv)
{
    //Mat image_fish = imread("/home/hl18/Desktop/share/img/R_test/0.png", CV_LOAD_IMAGE_COLOR);
    //Mat image_fish = imread("/home/hl18/Desktop/share/img/cell/cam0/test.0000.png", CV_LOAD_IMAGE_COLOR);
    //Mat image_fish = imread("/home/hl18/Desktop/share/img/stereo-img/no_shake/cam0/path3.0213.png", CV_LOAD_IMAGE_COLOR);
//    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/stereo-img/no_shake/cam0/path3.0166.png", CV_LOAD_IMAGE_COLOR);//13,123
//    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/stereo-img/no_shake/cam1/path3.0166.png", CV_LOAD_IMAGE_COLOR);

//    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/stereo-high/left_noshake_higher/path3.0280.png", CV_LOAD_IMAGE_COLOR);//high 3,253,260,280
//    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/stereo-high/right_noshake_higher/path3.0280.png", CV_LOAD_IMAGE_COLOR);
//    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/real_fisheye/left4.jpg", CV_LOAD_IMAGE_COLOR);
//    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/real_fisheye/right4.jpg", CV_LOAD_IMAGE_COLOR);//图4

    Mat image_fish_left = imread("/media/hl/Stuff/ubuntu_share_2/img/real_fisheye/out_left1.jpg", CV_LOAD_IMAGE_COLOR);//high 3,253,260,280
    Mat image_fish_right = imread("/media/hl/Stuff/ubuntu_share_2/img/real_fisheye/out_right1.jpg", CV_LOAD_IMAGE_COLOR);



//    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/indoor2/higher_cam0/indoor2.0019.png", CV_LOAD_IMAGE_COLOR);//high 3,
//    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/indoor2/higher_cam1/indoor2.0019.png", CV_LOAD_IMAGE_COLOR);



//    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/cell/cam0/indoor2.0014.png", CV_LOAD_IMAGE_COLOR);
//    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/indoor2/cam1/indoor2.0014.png", CV_LOAD_IMAGE_COLOR);

    //Mat K = (Mat_<double>(3, 3) << 240.81199402, 0, 539.66673938, 0, 240.78335949, 539.1616528, 0, 0, 1);//DS Model
    //Mat K = (Mat_<double>(3, 3) << 239.79678799, 0, 539.76995704, 0, 239.72181445, 539.19549573, 0, 0, 1);//DS Model new
    //Mat K = (Mat_<double>(3, 3) << 248.89280656, 0, 539.43845479, 0, 248.5884288, 539.10069305, 0, 0, 1);//DS Model new1
    //Mat K = (Mat_<double>(3, 3) << 252.28724715, 0, 539.19192922, 0, 251.95114552, 539.10008295, 0, 0, 1);//DS Model new2
    //Mat K = (Mat_<double>(3, 3) << 244.31339106, 0, 539.66602239, 0, 243.94952245, 538.98743521, 0, 0, 1);//DS Model new3
    //Mat K = (Mat_<double>(3, 3) << 257.3320673, 0, 544.41885058, 0, 257.06231132, 539.2503583, 0, 0, 1);//DS Model new4
    //Mat K = (Mat_<double>(3, 3) << 251.64176002, 0, 542.95940486, 0, 251.30969595, 538.29781881, 0, 0, 1);//DS Model new5
    //Mat K = (Mat_<double>(3, 3) << 249.1472835, 0, 539.46565243, 0, 248.8003115, 538.95775648, 0, 0, 1);//DS Model new6
    //Mat K = (Mat_<double>(3, 3) << 974.90718116, 0, 2163.25698355, 0, 977.54993613, 2154.12876773, 0, 0, 1);//DS Model high
    //Mat K = (Mat_<double>(3, 3) << 990.37794771, 0, 2159.39059877, 0, 989.91367694, 2159.59669391, 0, 0, 1);//DS Model high,april1
    //Mat K = (Mat_<double>(3, 3) << 981.84829708, 0, 2159.77745084, 0, 980.41502662, 2158.81587262, 0, 0, 1);//DS Model high,april2
    //Mat K = (Mat_<double>(3, 3) << 411.30961729, 0, 587.34620693, 0, 410.99461855, 484.32560589, 0, 0, 1);//real
    Mat K1 = (Mat_<double>(3, 3) << 396.01307103, 0, 708.54459392, 0, 395.81900138, 464.01861501, 0, 0, 1);//real
    Mat K2 = (Mat_<double>(3, 3) << 416.68602176, 0, 617.57644155, 0, 415.75414779, 449.3526344, 0, 0, 1);//real
    //Mat K = (Mat_<double>(3, 3) << 411.66660775, 0, 587.34649248, 0, 411.35121595, 484.3256435, 0, 0, 1);//DS Model high,april2

    //-0.03234126 0.61896438
    //-0.00438942 0.64304057


    //Mat K = (Mat_<double>(3, 3) << 220, 0, 540, 0, 220, 540, 0, 0, 1);//DS Model
    //Mat K = (Mat_<double>(3, 3) << 219.39021038, 0, 543.42173266, 0, 219.51814631, 538.73714057, 0, 0, 1);//DS Model

    //Mat K_virtual = (Mat_<double>(3, 3) << 10.0, 0, 900, 0, 10.0, 900, 0, 0, 1);//想看包含边像素缘的，用这个
    //Mat K_virtual = (Mat_<double>(3, 3) << 50.0, 0, 900, 0, 50.0, 900, 0, 0, 1);//想看好看的圆形图像，用这个
    //Mat K_virtual = (Mat_<double>(3, 3) << 200, 0, 400, 0, 200, 400, 0, 0, 1);//想看好看的圆形图像，用这个
    //Mat K_virtual = (Mat_<double>(3, 3) << 800, 0, 475, 0, 800, 600, 0, 0, 1);//想看好看的圆形图像，用这个
    Mat K_virtual1 = (Mat_<double>(3, 3) << 800, 0, 700, 0, 800, 800, 0, 0, 1);//想看好看的圆形图像，用这个
    Mat K_virtual2 = (Mat_<double>(3, 3) << 800, 0, 350, 0, 800, 800, 0, 0, 1);//想看好看的圆形图像，用这个
    //Mat K_virtual = (Mat_<double>(3, 3) << 200.0, 0, 900, 0, 200.0, 900, 0, 0, 1);//想看好看的圆形图像，用这个
    //Mat K_virtual = (Mat_<double>(3, 3) << 24.81199402, 0, 539.66673938, 0, 24.78335949, 539.1616528, 0, 0, 1);//DS Model

    cv::Mat image_re_left = cv::Mat(cv::Size(960,1500), image_fish_left.type(), cv::Scalar(0, 0, 0));// 创建三通道黑色图像
    cv::Mat image_re_right = cv::Mat(cv::Size(960,1500), image_fish_right.type(), cv::Scalar(0, 0, 0));// 创建三通道黑色图像
    //cv::Mat image_re = cv::Mat(cv::Size(2000,2000), image_fish.type());// 创建三通道黑色图像
    //cv::imshow("image",image_re);
    //cv::waitKey(0);

    Eigen::Matrix3d R_virtual = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_virtual2 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_virtual3 = Eigen::Matrix3d::Identity();
    //绕y轴转60度的旋转矩阵
    R_virtual << 0.5, 0, -0.866, 0, 1, 0, 0.866, 0, 0.5;
    //绕y轴转-60度的旋转矩阵
    R_virtual2 << 0.5, 0, 0.866, 0, 1, 0, -0.866, 0, 0.5;
    R_virtual3 << 1,0,0,0,1,0,0,0,1;

    //遍历每个像素点
//    int u=0;
//    int v=0;
    //for (int v = 200; v < image_re.rows-200 ; v++)//有黑边
    for (int v = 0; v < image_re_left.rows-0 ; v++)
    {

        for (int u = 0; u < image_re_left.cols-0; u++)
        {
            Eigen::Vector2d pt_virtual_2d;  //虚拟相机像素坐标
            pt_virtual_2d << u, v;

            //虚拟相机像素坐标————虚拟相机下归一化坐标
            Eigen::Vector3d pt_3d;  //虚拟相机下归一化坐标
            virtual_pix_to_virtual_cam(pt_virtual_2d,K_virtual1,pt_3d);


            //虚拟相机下归一化坐标————鱼眼相机下归一化坐标
            Eigen::Vector3d pt_2d_fish=R_virtual2*pt_3d;    //鱼眼相机下归一化坐标(2d)  2
            double x_virtual = pt_2d_fish.x()/pt_2d_fish.z();//虚拟相机坐标系下的归一化坐标
            double y_virtual = pt_2d_fish.y()/pt_2d_fish.z();//虚拟相机坐标系下的归一化坐标
            Eigen::Vector3d pt_3d_fish;
            pt_3d_fish<<x_virtual,y_virtual,1;

            //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
            Eigen::Vector2d fish_pix;//鱼眼相机像素坐标
            //fish_cam_to_fish_pix(pt_3d_fish,K,fish_pix);

            double alpha1=0.61896438;
            double ksi1=-0.03234126;
            fish_cam_to_fish_pix(pt_3d_fish,K1,fish_pix,alpha1,ksi1);

            //传递颜色
            if(fish_pix.x()>image_fish_left.cols*1/3 && fish_pix.y()>image_fish_left.rows*0/3 && fish_pix.x()<(image_fish_left.cols+0) && fish_pix.y()<image_fish_left.rows*3/3)
            {
                image_re_left.at<Vec3b>(v, u)[0] = image_fish_left.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[0];
                image_re_left.at<Vec3b>(v, u)[1] = image_fish_left.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[1];
                image_re_left.at<Vec3b>(v, u)[2] = image_fish_left.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[2];
            }
            else
            {
                image_re_left.at<Vec3b>(v, u)[0] = 0;
                image_re_left.at<Vec3b>(v, u)[1] = 0;
                image_re_left.at<Vec3b>(v, u)[2] = 0;
            }
        }
    }

    for (int v_ = 0; v_ < image_re_right.rows-0 ; v_++)
    {

        for (int u_ = 0; u_ < image_re_right.cols-0; u_++)
        {
            Eigen::Vector2d pt_virtual_2d_;  //虚拟相机像素坐标
            pt_virtual_2d_ << u_, v_;

            //虚拟相机像素坐标————虚拟相机下归一化坐标
            Eigen::Vector3d pt_3d_;  //虚拟相机下归一化坐标
            virtual_pix_to_virtual_cam(pt_virtual_2d_,K_virtual2,pt_3d_);


            //虚拟相机下归一化坐标————鱼眼相机下归一化坐标
            Eigen::Vector3d pt_2d_fish_=R_virtual*pt_3d_;    //鱼眼相机下归一化坐标(2d)
            double x_virtual_ = pt_2d_fish_.x()/pt_2d_fish_.z();//虚拟相机坐标系下的归一化坐标
            double y_virtual_ = pt_2d_fish_.y()/pt_2d_fish_.z();//虚拟相机坐标系下的归一化坐标
            Eigen::Vector3d pt_3d_fish_;
            pt_3d_fish_<<x_virtual_,y_virtual_,1;

            //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
            Eigen::Vector2d fish_pix_;//鱼眼相机像素坐标
            //fish_cam_to_fish_pix(pt_3d_fish_,K,fish_pix_);

            double alpha2=0.64304057;
            double ksi2=-0.00438942;
            fish_cam_to_fish_pix(pt_3d_fish_,K2,fish_pix_,alpha2,ksi2);

            //传递颜色
            if(fish_pix_.x()>0 && fish_pix_.y()>image_fish_right.rows*0/3 && fish_pix_.x()<(image_fish_right.cols*2/3) && fish_pix_.y()<image_fish_right.rows*3/3)
            {
                image_re_right.at<Vec3b>(v_, u_)[0] = image_fish_right.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[0];
                image_re_right.at<Vec3b>(v_, u_)[1] = image_fish_right.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[1];
                image_re_right.at<Vec3b>(v_, u_)[2] = image_fish_right.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[2];
            }
            else
            {
                image_re_right.at<Vec3b>(v_, u_)[0] = 0;
                image_re_right.at<Vec3b>(v_, u_)[1] = 0;
                image_re_right.at<Vec3b>(v_, u_)[2] = 0;
            }
        }
    }
//    imshow("Left Image Reshaped", image_re_left);
//    imshow("Right Image Reshaped", image_re_right);

    ////from disparity_only_1.cpp

    cv::Mat left_image;
    cv::Mat right_image;

    // RGB转换成GRAY
    cv::cvtColor(image_re_left,left_image,cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_re_right,right_image,cv::COLOR_BGR2GRAY);



    // Load camera matrices
    Mat camera_matrix1 = K_virtual1; // Load camera matrix for left camera
    Mat camera_matrix2 = K_virtual2; // Load camera matrix for right camera
    Mat distortion_coeffs1 = (Mat_<double>(4, 1) << 0,0,0,0); // Load distortion coefficients for left camera
    Mat distortion_coeffs2 = (Mat_<double>(4, 1) << 0,0,0,0); // Load distortion coefficients for right camera

    // Compute relative essential matrix and fundamental matrix
    // Mat relative_rotation, relative_translation;
    // Mat essential_matrix = findEssentialMat(..., ...);
    // Mat fundamental_matrix = findFundamentalMat(..., ...);
    Mat R,T;
    R = (Mat_<double>(3, 3) << 1,0,0,0,1,0,0,0,1);
    //平移向量
    T = (Mat_<double>(3, 1) << 0, 1, 0);
    //T = (Mat_<double>(3, 1) << -38.78508484229597, 4.63812755472254, 8.614230468360322);//图4
//    T = (Mat_<double>(3, 1) << -38.60773059812425, 4.692911948156371, 9.349851095579611);//图6


    ////从欧拉角中减去两个yaw方向的60度后，再恢复成旋转矩阵，效果应更好

    // 图out1,epipolar_constraint_limit=0.2,修正图位姿：BA roll yaw pitch = -5.30282, 161.998, 0.88596
    Eigen::Vector3d euler_angles_ba; // 021:XZY顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    euler_angles_ba[0]=1.25*M_PI/180;//pitch 1.2(试出来的，用来得到效果最好的修正)
    euler_angles_ba[1]=-5*M_PI/180;//roll -5
    euler_angles_ba[2]=(180-161.998)*M_PI/180;//yaw，虚拟相机已经分别转了60度，故要减去120
    //T = (Mat_<double>(3, 1) << -38.78508484229597, 4.63812755472254, 8.614230468360322);//图4

/*    // 图out2,epipolar_constraint_limit=0.1,修正图位姿：BA roll yaw pitch = -17.8806, 143.587, 19.1694
    Eigen::Vector3d euler_angles_ba; // 021:XZY顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    euler_angles_ba[0]=-10.1694*M_PI/180;//pitch 1.2(试出来的，用来得到效果最好的修正)
    euler_angles_ba[1]=-10.8806*M_PI/180;//roll -5
    euler_angles_ba[2]=(180-150.587)*M_PI/180;//yaw，虚拟相机已经分别转了60度，故要减去120
    //T = (Mat_<double>(3, 1) << -38.78508484229597, 4.63812755472254, 8.614230468360322);//图4*/




    Mat RR;
    cout<<"from euler to R ="<<endl;//验证和R estimated by g2o结果是否一致
    Eigen::Matrix3d rotation_matrix_set;
    rotation_matrix_set = Eigen::AngleAxisd(euler_angles_ba[0], Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(euler_angles_ba[1], Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(euler_angles_ba[2], Eigen::Vector3d::UnitY());
    cout << "虚拟相机之间的 rotation matrix =\n" << rotation_matrix_set << endl;
    //将eigen转换成mat
    cv::eigen2cv(rotation_matrix_set, RR);

    euler_angle(RR);//输出：虚拟相机之间的 roll yaw pitch = 0.424221, 1.086, 4.7553

    // Compute rectification transform matrices
    Mat R1, R2, P1, P2, Q;
    stereoRectify(camera_matrix1, distortion_coeffs1, camera_matrix2, distortion_coeffs2,
                  left_image.size(), RR, T, R1, R2, P1, P2, Q);//用R：优化前；用RR：优化后
    /*https://blog.csdn.net/qq_25458977/article/details/114829674
    K1	第一个相机的内参,Size为3x3, 数据类型为CV_32F 或者 CV_64F
    D1	第一个相机的畸变参数, Size必须为4x1, 数据类型为CV_32F 或者 CV_64F
    K2	第二个相机的内参,Size为3x3, 数据类型为CV_32F 或者 CV_64F
    D2	第二个相机的畸变参数, Size必须为4x1, 数据类型为CV_32F 或者 CV_64F
    imageSize	做双目标定StereoCalibration() 时用的图片的size, 如ImageSize = cv::Size(640,480)
    R	两个相机之间的旋转矩阵, Rrl, 如果内参采用Kalibr标定, 那么这里的R就是Kalibr标定出的T的前3x3
    tvec	两个相机之间的平移向量,trl, 即为左目相机在右目相机坐标系中的坐标, 所以,如果两个相机左右摆放, 该向量中x值一般为负数;
    R1	第一个相机的修正矩阵, 即从实际去畸变后的左目摆放位姿到经过极线矫正后的左目位姿之间, 有一个旋转量,为R1
    R2	第二个相机的修正矩阵, 即从实际去畸变后的右目摆放位姿到经过极线矫正后的右目位姿之间, 有一个旋转量,为R2
    P1	修正后第一个相机的投影矩阵; P1包含了R1和K1, 可直接将左目相机坐标系的三维点,投影到像素坐标系中; 要注意会投影到修正后的图像中
    P2	修正后第二个相机的投影矩阵; P2包含了R2和K2, 可直接将左目相机坐标系的三维点,投影到像素坐标系中; 要注意会投影到修正后的图像中
    Q	视差图转换成深度图的矩阵; 用于将视差图转换成深度图, 也就是将视差图中的每个像素点的视差值,转换成深度值

    flags	Operation flags that may be zero or fisheye::CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
    newImageSize	修正后图像的新Size.  该参数应该与下一步使用initUndistortRectifyMap()时所使用的iMAGE SIZE一致. 默认为 (0,0), 表示和 imageSize 一致. 当图像的径向畸变较严重时, 这个值设置的大一点,可以更好地保留一个细节;  (see the stereo_calib.cpp sample in OpenCV samples directory)
    balance	值在[0,1]之间, 设置这个值可以改变新图像的focal length, 从最小值到最大值之间变动;
    fov_scale	新的focal length = original focal length/ fov_scale
    */

    // Apply rectification to images
    Mat left_image_rectified, right_image_rectified;
    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(camera_matrix1, distortion_coeffs1, R1, P1, left_image.size(),
                            CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(camera_matrix2, distortion_coeffs2, R2, P2, right_image.size(),
                            CV_32FC1, map2x, map2y);
    remap(left_image, left_image_rectified, map1x, map1y, INTER_LINEAR);
    remap(right_image, right_image_rectified, map2x, map2y, INTER_LINEAR);

    // 对图像进行高斯滤波
    GaussianBlur(left_image_rectified, left_image_rectified, Size(7, 7), 0);
    /* 高斯滤波核的大小是根据图像的特征和噪声程度来决定的。一般来说，核的大小越大，滤波效果越平滑，但也会导致图像细节的丢失。因此，在选择核的大小时需要权衡平滑程度和细节保留程度。对于一般的图像，常见的核大小为3x3、5x5、7x7等。*/
    GaussianBlur(right_image_rectified, right_image_rectified, Size(7, 7), 0);

    // Display rectified images
//    imshow("Left Image Rectified", left_image_rectified);
//    imshow("Right Image Rectified", right_image_rectified);
    //waitKey();

//https://blog.csdn.net/one_cup_of_pepsi/article/details/121156675

    /* https://blog.csdn.net/wwp2016/article/details/86080722
     * minDisparity：最小视差，默认为0。此参数决定左图中的像素点在右图匹配搜索的起点，int 类型；
        minDisparity必须大于或等于0，小于numDisparity。如果minDisparity为0，那么搜索起点就是0，如果minDisparity为16，那么搜索起点就是16，以此类推。如果minDisparity为负数，那么会报错。
     * numDisparity:视差搜索范围长度，其值必须为16的整数倍。最大视差 maxDisparity = minDisparity + numDisparities -1；
        确定立体/深度图的分辨率。可以定义的“深度”级别由您的 numDisparity 的值驱动。 .如果您的 numDisparity值越高，意味着分辨率越高，这意味着将定义更多的深度级别。如果它较低，则意味着分辨率会较低，这意味着您可能无法看到许多“级别”的深度。增加 numDisparity使算法变慢，但给出更好的结果。https://www.coder.work/article/7036853
     * blockSize:SAD代价计算窗口大小。窗口大小为奇数，一般在3*3 到21*21之间。
     * P1、P2：视差连续性惩罚系数，默认为0。P1和P2是SGBM算法的两个重要参数，它们的值越大，视差越平滑，但是会增加计算量。
         P1、P2：能量函数参数，P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。需要指出，在动态规划时，P1和P2都是常数。
         一般建议：P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize；P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize；其中cn=1或3，取决于图像的颜色格式。
     * disp12MaxDiff：左右一致性检测最大容许误差阈值。int 类型
        左右视差图最大容许差异，默认为-1，即不执行左右视差检查。如果大于0，那么将会被执行。
     * preFilterCap：预过滤器的截断值。预过滤器是为了去除图像噪声。它的值越大，图像越清晰，但是会增加计算量。
     * uniquenessRatio：uniquenessRatio主要可以防止误匹配。视差唯一性百分比，视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
     * speckleWindowSize：视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。
        视差检查窗口大小。平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
     * speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。
     */

    int minDisparity=0;//minDisparity：最小视差，默认为0。此参数决定左图中的像素点在右图匹配搜索的起点
    int numDisparity=16*7;//numDisparity:视差搜索范围长度，其值必须为16的整数倍。确定立体/深度图的分辨率。最大视差 maxDisparity = minDisparity + numDisparities -1；
    int blockSize=3*3;//SAD代价计算窗口大小,默认大小为5。窗口大小为奇数，一般在3*3 到21*21之间。
    int disp12MaxDiff=20;//disp12MaxDiff：左右一致性检测最大容许误差阈值,默认为-1，即不执行左右视差检查。如果大于0，那么将会被执行。
    int preFilterCap=0;//preFilterCap：预过滤器的截断值。预过滤器是为了去除图像噪声。它的值越大，图像越清晰，但是会增加计算量。
    int uniquenessRatio=5;//uniquenessRatio：uniquenessRatio主要可以防止误匹配。视差唯一性百分比，通常为5~15.
    int speckleWindowSize=27;//视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
    int speckleRange=5;//speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            //0, 96, 9, 4 * 3 * 9, 16 * 3 * 9, 20, 0, 0, 27, 3);    // 神奇的参数 8*10* blockSize
            minDisparity, numDisparity, blockSize, 8 * 18 * blockSize, 32 * 18 * blockSize, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange);    // 神奇的参数
    //0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 20, 0, 0, 27, 3);    // 神奇的参数
    //0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数


    cv::Mat disparity_sgbm, disparity;
//    sgbm->compute(image_re_left, image_re_right, disparity_sgbm);
    sgbm->compute(left_image_rectified, right_image_rectified, disparity_sgbm);
    //sgbm->compute(image_re_left, image_re_right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

//    // 绘制极线
//    cv::Scalar lineColor(0, 0, 255);   // 红色
//    cv::Mat imgLeftRectLine = left_image_rectified.clone();
//    cv::Mat imgRightRectLine = right_image_rectified.clone();
//    for (int j = 0; j < imgLeftRectLine.rows; j += 16) {
//        cv::line(imgLeftRectLine, cv::Point(0, j), cv::Point(imgLeftRectLine.cols, j), lineColor, 1, cv::LINE_AA);
//        cv::line(imgRightRectLine, cv::Point(0, j), cv::Point(imgRightRectLine.cols, j), lineColor, 1, cv::LINE_AA);
//    }
//
//    cv::imshow("Left rectified image", imgLeftRectLine);
//    cv::imshow("Right rectified image", imgRightRectLine);

    namedWindow("左目原图",CV_WINDOW_NORMAL);
    namedWindow("右目原图",CV_WINDOW_NORMAL);
    namedWindow("左目修正",CV_WINDOW_NORMAL);
    namedWindow("右目修正",CV_WINDOW_NORMAL);
    namedWindow("左目对齐",CV_WINDOW_NORMAL);
    namedWindow("右目对齐",CV_WINDOW_NORMAL);
    namedWindow("视差图",CV_WINDOW_NORMAL);
    cv::imshow("左目原图",image_fish_left);
    cv::imshow("右目原图",image_fish_right);
    cv::imshow("左目修正",image_re_left);
    cv::imshow("右目修正",image_re_right);
    imshow("左目对齐", left_image_rectified);
    imshow("右目对齐", right_image_rectified);
    cv::imshow("视差图", disparity / 96.0);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/unfish_left_out2.png", image_re_left);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/unfish_right_out2.png", image_re_right);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/epip_left_out1-2_ba.png", left_image_rectified);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/epip_right_out1-2_ba.png", right_image_rectified);


//保存视差图（https://blog.csdn.net/nzise_se/article/details/78489554）
    cv::Mat disp8U = Mat(disparity.rows, disparity.cols, CV_8UC1);       //显示
    normalize(disparity, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    //cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/disp_noba.png", disp8U);

    cv::waitKey(0);


//    // SGBM算法
//    cv::Mat disp;
//    cv::namedWindow("SGBM_disparity");
//    SGBMStart();
//
//    while (true) {
//        SGBM->compute(left, right, disp);
//        cv::Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
//        normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
//        cv::imshow("SGBM_disparity", disp8U);
//        cv::waitKey(4);
//    }

    return 0;
}

void euler_angle(Mat &R)
{
    // 欧拉角 四元数
    Eigen::Matrix<double, 3, 3> R_matrix;
    cv::cv2eigen(R, R_matrix); // cv::Mat 转换成 Eigen::Matrix
    Eigen::Vector3d euler_angles = R_matrix.eulerAngles(0, 2, 1); // 021:XZY顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    double pitch=euler_angles[0]*180/M_PI;
    double roll=euler_angles[1]*180/M_PI;
    double yaw=euler_angles[2]*180/M_PI;
    //cout << "better yaw pitch roll = " << euler_angles.transpose() << endl;

//    if(roll<-90)//-170
//    {
//        roll=180+roll;
//    }
//    if(roll>90)//170
//    {
//        roll=180-roll;
//    }
//    if(pitch>90)//150
//    {
//        pitch=180-pitch;
//    }
//    if(pitch<-90)//-150
//    {
//        pitch=180+pitch;
//    }
//    if(yaw>0&&yaw<90)//60
//    {
//        yaw=180-yaw;
//    }
//    if(yaw>-90&&yaw<0)//-60
//    {
//        yaw=180+yaw;
//    }

    cout << "虚拟相机之间的 roll yaw pitch = " << roll <<", "<< yaw <<", "<< pitch << endl;//转化成角度制显示
    Eigen::Quaterniond qq=Eigen::Quaterniond(R_matrix);
    //cout << "better quaternion(x,y,z,w) = " << qq.coeffs().transpose() << endl;//
    //cout<<"better t="<<t<<endl;

}

//回调函数
void SGBMUpdate(int pos, void* data) {
    //cv::Mat disp;
    int SGBMNum = 2;
    int blockSize = cv::getTrackbarPos("blockSize", "SGBM_disparity");
    if (blockSize % 2 == 0){
        blockSize += 1;
    }
    if (blockSize < 5){
        blockSize = 5;
    }
    SGBM->setBlockSize(blockSize);
    SGBM->setNumDisparities(cv::getTrackbarPos("numDisparities", "SGBM_disparity"));
    SGBM->setSpeckleWindowSize(cv::getTrackbarPos("speckleWindowSize", "SGBM_disparity"));
    SGBM->setSpeckleRange(cv::getTrackbarPos("speckleRange", "SGBM_disparity"));
    SGBM->setUniquenessRatio(cv::getTrackbarPos("uniquenessRatio", "SGBM_disparity"));
    SGBM->setDisp12MaxDiff(cv::getTrackbarPos("disp12MaxDiff", "SGBM_disparity"));
    /*int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
    int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;*/
    // 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
    SGBM->setP1(600);
    // p1控制视差平滑度，p2值越大，差异越平滑
    SGBM->setP2(2400);
    SGBM->setMode(cv::StereoSGBM::MODE_SGBM);

}

/*
	SGBM函数初始化函数
*/
void SGBMStart() {
    // 最小视差值
    int minDisparity = 0;
    int SGBMNum = 2;
    // 视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
    int numDisparities = SGBMNum * 16;
    // 匹配块大小，大于1的奇数
    int blockSize = 5;
    // P1, P2控制视差图的光滑度
    // 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，P2 = 4 * P1
    int P1 = 600;
    // p1控制视差平滑度，p2值越大，差异越平滑
    int P2 = 2400;
    // 左右视差图的最大容许差异（超过将被清零），默认为 - 1，即不执行左右视差检查。
    int disp12MaxDiff = 200;
    int preFilterCap = 0;
    // 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
    int uniquenessRatio = 6;
    // 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
    int speckleWindowSize = 60;
    // 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
    int speckleRange = 2;

    //cv::namedWindow("SGBM_disparity");
    cv::createTrackbar("blockSize", "SGBM_disparity", &blockSize, 21, SGBMUpdate);
    cv::createTrackbar("numDisparities", "SGBM_disparity", &numDisparities, 20, SGBMUpdate);
    cv::createTrackbar("speckleWindowSize", "SGBM_disparity", &speckleWindowSize, 200, SGBMUpdate);
    cv::createTrackbar("speckleRange", "SGBM_disparity", &speckleRange, 50, SGBMUpdate);
    cv::createTrackbar("uniquenessRatio", "SGBM_disparity", &uniquenessRatio, 50, SGBMUpdate);
    cv::createTrackbar("disp12MaxDiff", "SGBM_disparity", &disp12MaxDiff, 21, SGBMUpdate);

    // 创建SGBM算法对象

}


//void virtual_pix_to_virtual_cam(Eigen::Vector2d &pt_virtual_2d,const cv::Mat &K_virtual,const Eigen::Vector3d &pt_3d);
void virtual_pix_to_virtual_cam(const Eigen::Vector2d &pt_virtual_2d, const cv::Mat &K_virtual,Eigen::Vector3d &pt_3d)
{
    int uu=int(pt_virtual_2d.x());
    int vv=int(pt_virtual_2d.y());
    double cx=K_virtual.at<double>(0,2);
    double cy=K_virtual.at<double>(1,2);
    double fx=K_virtual.at<double>(0,0);
    double fy=K_virtual.at<double>(1,1);
    double mx = (uu - cx) / fx;
    double my = (vv - cy) / fy;
//    double x=mx/ sqrt(mx*mx+my*my+1);
//    double y=my/ sqrt(mx*mx+my*my+1);
    pt_3d << mx,my,1;
}

void fish_cam_to_fish_pix(const Eigen::Vector3d &pt_3d_fish,const cv::Mat &K,Eigen::Vector2d &fish_pix, const double &alpha, const double &ksi)
{
    //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
    double cx=K.at<double>(0,2);
    double cy=K.at<double>(1,2);
    double fx=K.at<double>(0,0);
    double fy=K.at<double>(1,1);

//    double alpha=0.63222804;
//    double ksi=-0.00441318;//real
//    double alpha=0.63246005;
//    double ksi=-0.00355231 ;//april2

    //-0.03234126 0.61896438
    //-0.00438942 0.64304057

    double xx=pt_3d_fish.x();
    double yy=pt_3d_fish.y();
    double zz=pt_3d_fish.z();
    double d1= sqrt(xx*xx+yy*yy+zz*zz);
    double d2= sqrt(xx*xx+yy*yy+(ksi*d1+zz)*(ksi*d1+zz));
    double u_fish=fx*xx/(alpha*d2+(1-alpha)*(ksi*d1+zz))+cx;
    double v_fish=fy*yy/(alpha*d2+(1-alpha)*(ksi*d1+zz))+cy;
    fish_pix<<u_fish,v_fish;
}



//下面应该没用、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//#include <cmath>
//#include <Eigen/Core>
//#include <Eigen/Geometry>
//#include <opencv2/core/eigen.hpp>
//#include <vector>
//#include <string>
//#include <pangolin/pangolin.h>
//#include <unistd.h>
//
//using namespace std;
//using namespace cv;
//
//// 声明一些全局变量
//cv::Ptr<cv::StereoSGBM> SGBM = cv::StereoSGBM::create();
//
//void virtual_pix_to_virtual_cam(const Eigen::Vector2d &pt_virtual_2d,const cv::Mat &K_virtual,Eigen::Vector3d &pt_3d);
//void fish_cam_to_fish_pix(const Eigen::Vector3d &pt_3d_fish,const cv::Mat &K,Eigen::Vector2d &fish_pix);
//void SGBMUpdate(int pos, void* data);
//void SGBMStart();
//
//
//
//int main(int argc, char **argv)
//{
//    //Mat image_fish = imread("/home/hl18/Desktop/share/img/R_test/0.png", CV_LOAD_IMAGE_COLOR);
//    //Mat image_fish = imread("/home/hl18/Desktop/share/img/cell/cam0/test.0000.png", CV_LOAD_IMAGE_COLOR);
//    //Mat image_fish = imread("/home/hl18/Desktop/share/img/stereo-img/no_shake/cam0/path3.0213.png", CV_LOAD_IMAGE_COLOR);
////    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/stereo-img/no_shake/cam0/path3.0166.png", CV_LOAD_IMAGE_COLOR);//13,123
////    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/stereo-img/no_shake/cam1/path3.0166.png", CV_LOAD_IMAGE_COLOR);
//
////    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/stereo-high/left_noshake_higher/path3.0280.png", CV_LOAD_IMAGE_COLOR);//high 3,253,260,280
////    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/stereo-high/right_noshake_higher/path3.0280.png", CV_LOAD_IMAGE_COLOR);
//    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/real_fisheye/left4.jpg", CV_LOAD_IMAGE_COLOR);//high 3,253,260,280
//    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/real_fisheye/right4.jpg", CV_LOAD_IMAGE_COLOR);
//
//
//
////    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/indoor2/higher_cam0/indoor2.0019.png", CV_LOAD_IMAGE_COLOR);//high 3,
////    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/indoor2/higher_cam1/indoor2.0019.png", CV_LOAD_IMAGE_COLOR);
//
//
//
////    Mat image_fish_left = imread("/home/hl18/Desktop/share/img/cell/cam0/indoor2.0014.png", CV_LOAD_IMAGE_COLOR);
////    Mat image_fish_right = imread("/home/hl18/Desktop/share/img/indoor2/cam1/indoor2.0014.png", CV_LOAD_IMAGE_COLOR);
//
//    //Mat K = (Mat_<double>(3, 3) << 240.81199402, 0, 539.66673938, 0, 240.78335949, 539.1616528, 0, 0, 1);//DS Model
//    //Mat K = (Mat_<double>(3, 3) << 239.79678799, 0, 539.76995704, 0, 239.72181445, 539.19549573, 0, 0, 1);//DS Model new
//    //Mat K = (Mat_<double>(3, 3) << 248.89280656, 0, 539.43845479, 0, 248.5884288, 539.10069305, 0, 0, 1);//DS Model new1
//    //Mat K = (Mat_<double>(3, 3) << 252.28724715, 0, 539.19192922, 0, 251.95114552, 539.10008295, 0, 0, 1);//DS Model new2
//    //Mat K = (Mat_<double>(3, 3) << 244.31339106, 0, 539.66602239, 0, 243.94952245, 538.98743521, 0, 0, 1);//DS Model new3
//    //Mat K = (Mat_<double>(3, 3) << 257.3320673, 0, 544.41885058, 0, 257.06231132, 539.2503583, 0, 0, 1);//DS Model new4
//    //Mat K = (Mat_<double>(3, 3) << 251.64176002, 0, 542.95940486, 0, 251.30969595, 538.29781881, 0, 0, 1);//DS Model new5
//    //Mat K = (Mat_<double>(3, 3) << 249.1472835, 0, 539.46565243, 0, 248.8003115, 538.95775648, 0, 0, 1);//DS Model new6
//    //Mat K = (Mat_<double>(3, 3) << 974.90718116, 0, 2163.25698355, 0, 977.54993613, 2154.12876773, 0, 0, 1);//DS Model high
//    //Mat K = (Mat_<double>(3, 3) << 990.37794771, 0, 2159.39059877, 0, 989.91367694, 2159.59669391, 0, 0, 1);//DS Model high,april1
//    //Mat K = (Mat_<double>(3, 3) << 981.84829708, 0, 2159.77745084, 0, 980.41502662, 2158.81587262, 0, 0, 1);//DS Model high,april2
//    Mat K = (Mat_<double>(3, 3) << 411.30961729, 0, 587.34620693, 0, 410.99461855, 484.32560589, 0, 0, 1);//real
//    //Mat K = (Mat_<double>(3, 3) << 411.66660775, 0, 587.34649248, 0, 411.35121595, 484.3256435, 0, 0, 1);//DS Model high,april2
//
//
//    //Mat K = (Mat_<double>(3, 3) << 220, 0, 540, 0, 220, 540, 0, 0, 1);//DS Model
//    //Mat K = (Mat_<double>(3, 3) << 219.39021038, 0, 543.42173266, 0, 219.51814631, 538.73714057, 0, 0, 1);//DS Model
//
//    //Mat K_virtual = (Mat_<double>(3, 3) << 10.0, 0, 900, 0, 10.0, 900, 0, 0, 1);//想看包含边像素缘的，用这个
//    //Mat K_virtual = (Mat_<double>(3, 3) << 50.0, 0, 900, 0, 50.0, 900, 0, 0, 1);//想看好看的圆形图像，用这个
//    //Mat K_virtual = (Mat_<double>(3, 3) << 200, 0, 400, 0, 200, 400, 0, 0, 1);//想看好看的圆形图像，用这个
//    Mat K_virtual = (Mat_<double>(3, 3) << 800, 0, 300, 0, 800, 600, 0, 0, 1);//想看好看的圆形图像，用这个
//    //Mat K_virtual = (Mat_<double>(3, 3) << 200.0, 0, 900, 0, 200.0, 900, 0, 0, 1);//想看好看的圆形图像，用这个
//    //Mat K_virtual = (Mat_<double>(3, 3) << 24.81199402, 0, 539.66673938, 0, 24.78335949, 539.1616528, 0, 0, 1);//DS Model
//
//    cv::Mat image_re_left = cv::Mat(cv::Size(740,960), image_fish_left.type(), cv::Scalar(0, 0, 0));// 创建三通道黑色图像
//    cv::Mat image_re_right = cv::Mat(cv::Size(740,960), image_fish_right.type(), cv::Scalar(0, 0, 0));// 创建三通道黑色图像
//    //cv::Mat image_re = cv::Mat(cv::Size(2000,2000), image_fish.type());// 创建三通道黑色图像
//    //cv::imshow("image",image_re);
//    //cv::waitKey(0);
//
//    Eigen::Matrix3d R_virtual = Eigen::Matrix3d::Identity();
//    Eigen::Matrix3d R_virtual2 = Eigen::Matrix3d::Identity();
//    Eigen::Matrix3d R_virtual3 = Eigen::Matrix3d::Identity();
//    //绕y轴转60度的旋转矩阵
//    R_virtual << 0.5, 0, -0.866, 0, 1, 0, 0.866, 0, 0.5;
//    //绕y轴转-60度的旋转矩阵
//    R_virtual2 << 0.5, 0, 0.866, 0, 1, 0, -0.866, 0, 0.5;
//    R_virtual3 << 1,0,0,0,1,0,0,0,1;
//
//    //遍历每个像素点
////    int u=0;
////    int v=0;
//    //for (int v = 200; v < image_re.rows-200 ; v++)//有黑边
//    for (int v = 0; v < image_re_left.rows-0 ; v++)
//    {
//
//        for (int u = 0; u < image_re_left.cols-0; u++)
//        {
//            Eigen::Vector2d pt_virtual_2d;  //虚拟相机像素坐标
//            pt_virtual_2d << u, v;
//
//            //虚拟相机像素坐标————虚拟相机下归一化坐标
//            Eigen::Vector3d pt_3d;  //虚拟相机下归一化坐标
//            virtual_pix_to_virtual_cam(pt_virtual_2d,K_virtual,pt_3d);
//
//
//            //虚拟相机下归一化坐标————鱼眼相机下归一化坐标
//            Eigen::Vector3d pt_2d_fish=R_virtual2*pt_3d;    //鱼眼相机下归一化坐标(2d)  2
//            double x_virtual = pt_2d_fish.x()/pt_2d_fish.z();//虚拟相机坐标系下的归一化坐标
//            double y_virtual = pt_2d_fish.y()/pt_2d_fish.z();//虚拟相机坐标系下的归一化坐标
//            Eigen::Vector3d pt_3d_fish;
//            pt_3d_fish<<x_virtual,y_virtual,1;
//
//            //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
//            Eigen::Vector2d fish_pix;//鱼眼相机像素坐标
//            fish_cam_to_fish_pix(pt_3d_fish,K,fish_pix);
//
//            //传递颜色
//            if(fish_pix.x()>image_fish_left.cols*1/3 && fish_pix.y()>image_fish_left.rows*0/3 && fish_pix.x()<(image_fish_left.cols+0) && fish_pix.y()<image_fish_left.rows*3/3)
//            {
//                image_re_left.at<Vec3b>(v, u)[0] = image_fish_left.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[0];
//                image_re_left.at<Vec3b>(v, u)[1] = image_fish_left.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[1];
//                image_re_left.at<Vec3b>(v, u)[2] = image_fish_left.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[2];
//            }
//            else
//            {
//                image_re_left.at<Vec3b>(v, u)[0] = 0;
//                image_re_left.at<Vec3b>(v, u)[1] = 0;
//                image_re_left.at<Vec3b>(v, u)[2] = 0;
//            }
//        }
//    }
//
//    for (int v_ = 0; v_ < image_re_right.rows-0 ; v_++)
//    {
//
//        for (int u_ = 0; u_ < image_re_right.cols-0; u_++)
//        {
//            Eigen::Vector2d pt_virtual_2d_;  //虚拟相机像素坐标
//            pt_virtual_2d_ << u_, v_;
//
//            //虚拟相机像素坐标————虚拟相机下归一化坐标
//            Eigen::Vector3d pt_3d_;  //虚拟相机下归一化坐标
//            virtual_pix_to_virtual_cam(pt_virtual_2d_,K_virtual,pt_3d_);
//
//
//            //虚拟相机下归一化坐标————鱼眼相机下归一化坐标
//            Eigen::Vector3d pt_2d_fish_=R_virtual*pt_3d_;    //鱼眼相机下归一化坐标(2d)
//            double x_virtual_ = pt_2d_fish_.x()/pt_2d_fish_.z();//虚拟相机坐标系下的归一化坐标
//            double y_virtual_ = pt_2d_fish_.y()/pt_2d_fish_.z();//虚拟相机坐标系下的归一化坐标
//            Eigen::Vector3d pt_3d_fish_;
//            pt_3d_fish_<<x_virtual_,y_virtual_,1;
//
//            //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
//            Eigen::Vector2d fish_pix_;//鱼眼相机像素坐标
//            fish_cam_to_fish_pix(pt_3d_fish_,K,fish_pix_);
//
//            //传递颜色
//            if(fish_pix_.x()>0 && fish_pix_.y()>image_fish_right.rows*0/3 && fish_pix_.x()<(image_fish_right.cols*2/3) && fish_pix_.y()<image_fish_right.rows*3/3)
//            {
//                image_re_right.at<Vec3b>(v_, u_)[0] = image_fish_right.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[0];
//                image_re_right.at<Vec3b>(v_, u_)[1] = image_fish_right.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[1];
//                image_re_right.at<Vec3b>(v_, u_)[2] = image_fish_right.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[2];
//            }
//            else
//            {
//                image_re_right.at<Vec3b>(v_, u_)[0] = 0;
//                image_re_right.at<Vec3b>(v_, u_)[1] = 0;
//                image_re_right.at<Vec3b>(v_, u_)[2] = 0;
//            }
//        }
//    }
//    imshow("Left Image Reshaped", image_re_left);
//    imshow("Right Image Reshaped", image_re_right);
//
//    ////from disparity_only_1.cpp
//
//    cv::Mat left_image;
//    cv::Mat right_image;
//
//    // RGB转换成GRAY
//    cv::cvtColor(image_re_left,left_image,cv::COLOR_BGR2GRAY);
//    cv::cvtColor(image_re_right,right_image,cv::COLOR_BGR2GRAY);
//
//    //二层金字塔极线对齐
//    //Mat leftPyramid[2], rightPyramid[2];
//    pyrDown(left_image, left_image, Size(left_image.cols / 2, left_image.rows / 2));
//    pyrDown(right_image, right_image, Size(right_image.cols / 2, right_image.rows / 2));
//    pyrDown(left_image, left_image, Size(left_image.cols / 2, left_image.rows / 2));
//    pyrDown(right_image, right_image, Size(right_image.cols / 2, right_image.rows / 2));
//
//
////    // Load camera matrices
////    Mat camera_matrix1 = K_virtual; // Load camera matrix for left camera
////    Mat camera_matrix2 = K_virtual; // Load camera matrix for right camera
////    Mat distortion_coeffs1 = (Mat_<double>(4, 1) << 0,0,0,0); // Load distortion coefficients for left camera
////    Mat distortion_coeffs2 = (Mat_<double>(4, 1) << 0,0,0,0); // Load distortion coefficients for right camera
////
////    // Compute relative essential matrix and fundamental matrix
////    // Mat relative_rotation, relative_translation;
////    // Mat essential_matrix = findEssentialMat(..., ...);
////    // Mat fundamental_matrix = findFundamentalMat(..., ...);
////    Mat R,T;
////    R = (Mat_<double>(3, 3) << 1,0,0,0,1,0,0,0,1);
////    //平移向量
////    //T = (Mat_<double>(3, 1) << 0, 1, 0);
////    T = (Mat_<double>(3, 1) << -34.50948002970356, 11.92104569273593, -16.33965904390889);
////
////    //从欧拉角中减去两个yaw方向的60度后，再恢复成旋转矩阵，效果应更好
////    //BA roll yaw pitch = 0.424221, -118.914, 4.7553
////    Eigen::Vector3d euler_angles_ba; // 021:XZY顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
////    euler_angles_ba[0]=4.7553*M_PI/180;//pitch
////    euler_angles_ba[1]=0.424221*M_PI/180;//roll
////    euler_angles_ba[2]=(-118.914+120)*M_PI/180;//yaw
////    Mat RR;
////    cout<<"from euler to R ="<<endl;//验证和R estimated by g2o结果是否一致
////    Eigen::Matrix3d rotation_matrix_set;
////    rotation_matrix_set = Eigen::AngleAxisd(euler_angles_ba[0], Eigen::Vector3d::UnitX()) *
////                        Eigen::AngleAxisd(euler_angles_ba[1], Eigen::Vector3d::UnitZ()) *
////                        Eigen::AngleAxisd(euler_angles_ba[2], Eigen::Vector3d::UnitY());
////    cout << "rotation matrix_4 =\n" << rotation_matrix_set << endl;
////    //将eigen转换成mat
////    cv::eigen2cv(rotation_matrix_set, RR);
////
////    // Compute rectification transform matrices
////    Mat R1, R2, P1, P2, Q;
////    stereoRectify(camera_matrix1, distortion_coeffs1, camera_matrix2, distortion_coeffs2,
////                  left_image.size(), RR, T, R1, R2, P1, P2, Q);
////    /*https://blog.csdn.net/qq_25458977/article/details/114829674
////    K1	第一个相机的内参,Size为3x3, 数据类型为CV_32F 或者 CV_64F
////    D1	第一个相机的畸变参数, Size必须为4x1, 数据类型为CV_32F 或者 CV_64F
////    K2	第二个相机的内参,Size为3x3, 数据类型为CV_32F 或者 CV_64F
////    D2	第二个相机的畸变参数, Size必须为4x1, 数据类型为CV_32F 或者 CV_64F
////    imageSize	做双目标定StereoCalibration() 时用的图片的size, 如ImageSize = cv::Size(640,480)
////    R	两个相机之间的旋转矩阵, Rrl, 如果内参采用Kalibr标定, 那么这里的R就是Kalibr标定出的T的前3x3
////    tvec	两个相机之间的平移向量,trl, 即为左目相机在右目相机坐标系中的坐标, 所以,如果两个相机左右摆放, 该向量中x值一般为负数;
////    R1	第一个相机的修正矩阵, 即从实际去畸变后的左目摆放位姿到经过极线矫正后的左目位姿之间, 有一个旋转量,为R1
////    R2	第二个相机的修正矩阵, 即从实际去畸变后的右目摆放位姿到经过极线矫正后的右目位姿之间, 有一个旋转量,为R2
////    P1	修正后第一个相机的投影矩阵; P1包含了R1和K1, 可直接将左目相机坐标系的三维点,投影到像素坐标系中; 要注意会投影到修正后的图像中
////    P2	修正后第二个相机的投影矩阵; P2包含了R2和K2, 可直接将左目相机坐标系的三维点,投影到像素坐标系中; 要注意会投影到修正后的图像中
////    Q	视差图转换成深度图的矩阵; 用于将视差图转换成深度图, 也就是将视差图中的每个像素点的视差值,转换成深度值
////
////    flags	Operation flags that may be zero or fisheye::CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
////    newImageSize	修正后图像的新Size.  该参数应该与下一步使用initUndistortRectifyMap()时所使用的iMAGE SIZE一致. 默认为 (0,0), 表示和 imageSize 一致. 当图像的径向畸变较严重时, 这个值设置的大一点,可以更好地保留一个细节;  (see the stereo_calib.cpp sample in OpenCV samples directory)
////    balance	值在[0,1]之间, 设置这个值可以改变新图像的focal length, 从最小值到最大值之间变动;
////    fov_scale	新的focal length = original focal length/ fov_scale
////    */
////
////    // Apply rectification to images
////    Mat left_image_rectified, right_image_rectified;
////    Mat map1x, map1y, map2x, map2y;
////    initUndistortRectifyMap(camera_matrix1, distortion_coeffs1, R1, P1, left_image.size(),
////                            CV_32FC1, map1x, map1y);
////    initUndistortRectifyMap(camera_matrix2, distortion_coeffs2, R2, P2, right_image.size(),
////                            CV_32FC1, map2x, map2y);
////    remap(left_image, left_image_rectified, map1x, map1y, INTER_LINEAR);
////    remap(right_image, right_image_rectified, map2x, map2y, INTER_LINEAR);
//
//    // 对图像进行高斯滤波
////    GaussianBlur(left_image, left_image, Size(3, 3), 0);
////    /* 高斯滤波核的大小是根据图像的特征和噪声程度来决定的。一般来说，核的大小越大，滤波效果越平滑，但也会导致图像细节的丢失。因此，在选择核的大小时需要权衡平滑程度和细节保留程度。对于一般的图像，常见的核大小为3x3、5x5、7x7等。*/
////    GaussianBlur(right_image, right_image, Size(3, 3), 0);
//
//    // Display rectified images
////    imshow("Left Image Rectified", left_image);
////    imshow("Right Image Rectified", right_image);
//    //waitKey();
//
////https://blog.csdn.net/one_cup_of_pepsi/article/details/121156675
//
//    /* https://blog.csdn.net/wwp2016/article/details/86080722
//     * minDisparity：最小视差，默认为0。此参数决定左图中的像素点在右图匹配搜索的起点，int 类型；
//        minDisparity必须大于或等于0，小于numDisparity。如果minDisparity为0，那么搜索起点就是0，如果minDisparity为16，那么搜索起点就是16，以此类推。如果minDisparity为负数，那么会报错。
//     * numDisparity:视差搜索范围长度，其值必须为16的整数倍。最大视差 maxDisparity = minDisparity + numDisparities -1；
//        确定立体/深度图的分辨率。可以定义的“深度”级别由您的 numDisparity 的值驱动。 .如果您的 numDisparity值越高，意味着分辨率越高，这意味着将定义更多的深度级别。如果它较低，则意味着分辨率会较低，这意味着您可能无法看到许多“级别”的深度。增加 numDisparity使算法变慢，但给出更好的结果。https://www.coder.work/article/7036853
//     * blockSize:SAD代价计算窗口大小。窗口大小为奇数，一般在3*3 到21*21之间。
//     * P1、P2：视差连续性惩罚系数，默认为0。P1和P2是SGBM算法的两个重要参数，它们的值越大，视差越平滑，但是会增加计算量。
//         P1、P2：能量函数参数，P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。需要指出，在动态规划时，P1和P2都是常数。
//         一般建议：P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize；P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize；其中cn=1或3，取决于图像的颜色格式。
//     * disp12MaxDiff：左右一致性检测最大容许误差阈值。int 类型
//        左右视差图最大容许差异，默认为-1，即不执行左右视差检查。如果大于0，那么将会被执行。
//     * preFilterCap：预过滤器的截断值。预过滤器是为了去除图像噪声。它的值越大，图像越清晰，但是会增加计算量。
//     * uniquenessRatio：uniquenessRatio主要可以防止误匹配。视差唯一性百分比，视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
//     * speckleWindowSize：视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。
//        视差检查窗口大小。平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
//     * speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。
//     */
//
//    int minDisparity=0;//minDisparity：最小视差，默认为0。此参数决定左图中的像素点在右图匹配搜索的起点
//    int numDisparity=180;//numDisparity:视差搜索范围长度，其值必须为16的整数倍。确定立体/深度图的分辨率。最大视差 maxDisparity = minDisparity + numDisparities -1；
//    int blockSize=3*3;//SAD代价计算窗口大小,默认大小为5。窗口大小为奇数，一般在3*3 到21*21之间。
//    int disp12MaxDiff=50;//disp12MaxDiff：左右一致性检测最大容许误差阈值,默认为-1，即不执行左右视差检查。如果大于0，那么将会被执行。
//    int preFilterCap=50;//preFilterCap：预过滤器的截断值。预过滤器是为了去除图像噪声。它的值越大，图像越清晰，但是会增加计算量。
//    int uniquenessRatio=0;//uniquenessRatio：uniquenessRatio主要可以防止误匹配。视差唯一性百分比，通常为5~15.
//    int speckleWindowSize=140;//视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
//    int speckleRange=5;//speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。
//    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
//            //0, 96, 9, 4 * 3 * 9, 16 * 3 * 9, 20, 0, 0, 27, 3);    // 神奇的参数 8*10* blockSize
//            minDisparity, numDisparity, blockSize, 4 * 8 * blockSize, 16 * 8 * blockSize, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange);    // 神奇的参数
//            //0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 20, 0, 0, 27, 3);    // 神奇的参数
//            //0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
//
//
//    cv::Mat disparity_sgbm, disparity;
////    sgbm->compute(image_re_left, image_re_right, disparity_sgbm);
//    sgbm->compute(left_image, right_image, disparity_sgbm);
//    //sgbm->compute(image_re_left, image_re_right, disparity_sgbm);
//    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
//
//    namedWindow("左目原图",CV_WINDOW_NORMAL);
//    namedWindow("右目原图",CV_WINDOW_NORMAL);
//    namedWindow("左目修正",CV_WINDOW_NORMAL);
//    namedWindow("右目修正",CV_WINDOW_NORMAL);
//    namedWindow("左目对齐",CV_WINDOW_NORMAL);
//    namedWindow("右目对齐",CV_WINDOW_NORMAL);
//    namedWindow("视差图",CV_WINDOW_NORMAL);
//    cv::imshow("左目原图",image_fish_left);
//    cv::imshow("右目原图",image_fish_right);
//    cv::imshow("左目修正",image_re_left);
//    cv::imshow("右目修正",image_re_right);
//    cv::imshow("左目对齐",left_image);
//    cv::imshow("右目对齐",right_image);
//    cv::imshow("disparity", disparity / 96.0);
////    cv::imwrite("/home/hl18/Desktop/share/img/stereo-high/reshape/unfish_left.png", image_re_left);
////    cv::imwrite("/home/hl18/Desktop/share/img/stereo-high/reshape/unfish_right.png", image_re_right);
//    cv::waitKey(0);
//
//
////    // SGBM算法
////    cv::Mat disp;
////    cv::namedWindow("SGBM_disparity");
////    SGBMStart();
////
////    while (true) {
////        SGBM->compute(left, right, disp);
////        cv::Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
////        normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
////        cv::imshow("SGBM_disparity", disp8U);
////        cv::waitKey(4);
////    }
//
//    return 0;
//}
//
////回调函数
//void SGBMUpdate(int pos, void* data) {
//    //cv::Mat disp;
//    int SGBMNum = 2;
//    int blockSize = cv::getTrackbarPos("blockSize", "SGBM_disparity");
//    if (blockSize % 2 == 0){
//        blockSize += 1;
//    }
//    if (blockSize < 5){
//        blockSize = 5;
//    }
//    SGBM->setBlockSize(blockSize);
//    SGBM->setNumDisparities(cv::getTrackbarPos("numDisparities", "SGBM_disparity"));
//    SGBM->setSpeckleWindowSize(cv::getTrackbarPos("speckleWindowSize", "SGBM_disparity"));
//    SGBM->setSpeckleRange(cv::getTrackbarPos("speckleRange", "SGBM_disparity"));
//    SGBM->setUniquenessRatio(cv::getTrackbarPos("uniquenessRatio", "SGBM_disparity"));
//    SGBM->setDisp12MaxDiff(cv::getTrackbarPos("disp12MaxDiff", "SGBM_disparity"));
//    /*int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
//    int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;*/
//    // 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
//    SGBM->setP1(600);
//    // p1控制视差平滑度，p2值越大，差异越平滑
//    SGBM->setP2(2400);
//    SGBM->setMode(cv::StereoSGBM::MODE_SGBM);
//
//}
//
///*
//	SGBM函数初始化函数
//*/
//void SGBMStart() {
//    // 最小视差值
//    int minDisparity = 0;
//    int SGBMNum = 2;
//    // 视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
//    int numDisparities = SGBMNum * 16;
//    // 匹配块大小，大于1的奇数
//    int blockSize = 5;
//    // P1, P2控制视差图的光滑度
//    // 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，P2 = 4 * P1
//    int P1 = 600;
//    // p1控制视差平滑度，p2值越大，差异越平滑
//    int P2 = 2400;
//    // 左右视差图的最大容许差异（超过将被清零），默认为 - 1，即不执行左右视差检查。
//    int disp12MaxDiff = 200;
//    int preFilterCap = 0;
//    // 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
//    int uniquenessRatio = 6;
//    // 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
//    int speckleWindowSize = 60;
//    // 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
//    int speckleRange = 2;
//
//    //cv::namedWindow("SGBM_disparity");
//    cv::createTrackbar("blockSize", "SGBM_disparity", &blockSize, 21, SGBMUpdate);
//    cv::createTrackbar("numDisparities", "SGBM_disparity", &numDisparities, 20, SGBMUpdate);
//    cv::createTrackbar("speckleWindowSize", "SGBM_disparity", &speckleWindowSize, 200, SGBMUpdate);
//    cv::createTrackbar("speckleRange", "SGBM_disparity", &speckleRange, 50, SGBMUpdate);
//    cv::createTrackbar("uniquenessRatio", "SGBM_disparity", &uniquenessRatio, 50, SGBMUpdate);
//    cv::createTrackbar("disp12MaxDiff", "SGBM_disparity", &disp12MaxDiff, 21, SGBMUpdate);
//
//    // 创建SGBM算法对象
//
//}
//
//
////void virtual_pix_to_virtual_cam(Eigen::Vector2d &pt_virtual_2d,const cv::Mat &K_virtual,const Eigen::Vector3d &pt_3d);
//void virtual_pix_to_virtual_cam(const Eigen::Vector2d &pt_virtual_2d, const cv::Mat &K_virtual,Eigen::Vector3d &pt_3d)
//{
//    int uu=int(pt_virtual_2d.x());
//    int vv=int(pt_virtual_2d.y());
//    double cx=K_virtual.at<double>(0,2);
//    double cy=K_virtual.at<double>(1,2);
//    double fx=K_virtual.at<double>(0,0);
//    double fy=K_virtual.at<double>(1,1);
//    double mx = (uu - cx) / fx;
//    double my = (vv - cy) / fy;
////    double x=mx/ sqrt(mx*mx+my*my+1);
////    double y=my/ sqrt(mx*mx+my*my+1);
//    pt_3d << mx,my,1;
//}
//
//void fish_cam_to_fish_pix(const Eigen::Vector3d &pt_3d_fish,const cv::Mat &K,Eigen::Vector2d &fish_pix)
//{
//    //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
//    double cx=K.at<double>(0,2);
//    double cy=K.at<double>(1,2);
//    double fx=K.at<double>(0,0);
//    double fy=K.at<double>(1,1);
////    double alpha=0.59417224;
////    double ksi=-0.0420613;
////    double alpha=0.59323367;
////    double ksi=-0.04599436;//new
////    double alpha=0.57716941;
////    double ksi=-0.12002858;
////    double alpha=0.60078736;
////    double ksi=-0.00981989;//new1
////    double alpha=0.60343655;
////    double ksi=0.00358328 ;//new2
////    double alpha=0.59708318;
////    double ksi=-0.0276233;//new3
////    double alpha=0.60717885;
////    double ksi=0.03229258;//new4
////    double alpha=0.60458527;
////    double ksi=0.01238863;//new5
////    double alpha=0.60098396;
////    double ksi=-0.00888748;//new6
////    double alpha=0.59861597;
////    double ksi=-0.00809314;//april1
////    double alpha=0.5971184;
////    double ksi=-0.01777219;//april2
//    double alpha=0.63222804;
//    double ksi=-0.00441318;//real
////    double alpha=0.63246005;
////    double ksi=-0.00355231 ;//april2
//
//
//
//    double xx=pt_3d_fish.x();
//    double yy=pt_3d_fish.y();
//    double zz=pt_3d_fish.z();
//    double d1= sqrt(xx*xx+yy*yy+zz*zz);
//    double d2= sqrt(xx*xx+yy*yy+(ksi*d1+zz)*(ksi*d1+zz));
//    double u_fish=fx*xx/(alpha*d2+(1-alpha)*(ksi*d1+zz))+cx;
//    double v_fish=fy*yy/(alpha*d2+(1-alpha)*(ksi*d1+zz))+cy;
//    fish_pix<<u_fish,v_fish;
//}
