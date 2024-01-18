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

#include "BA_func.h"
//#include "BA_func_2.h"
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include <chrono>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace cv;

void E_tR_scale(Mat &R, Mat &t, Mat &t_x, Mat &essential_matrix, Mat &scale);
void euler_angle(Mat &R,double &roll,double &yaw,double &pitch);
void pFp(Mat &fundamental_matrix,vector<DMatch> &better_matches,vector<Point2f> &points11,vector<Point2f> &points22);

void virtual_pix_to_virtual_cam(const Eigen::Vector2d &pt_virtual_2d,const cv::Mat &K_virtual,Eigen::Vector3d &pt_3d);
void fish_cam_to_fish_pix(const Eigen::Vector3d &pt_3d_fish,const cv::Mat &K,Eigen::Vector2d &fish_pix);

void find_feature_matches_sift(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &good_matches);

void find_feature_matches_orb(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &good_matches);

void triangulation_self(Mat &R, Mat &t, Mat &K,
                        vector<Point2f> &points11,vector<Point2f> &points22,
                        vector<DMatch> &better_matches,vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,
                        Mat &img_1,Mat &img_2,
                        VecVector3d &pts_3d_eigen);

/// 作图用
inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> good_matches,
        Mat &R, Mat &t, Mat &K, Mat &essential_matrix, Mat &fundamental_matrix);

void pose_SVD_self(Mat &essential_matrix, Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);
//像素坐标转相机坐标(假想曲面上的xp)
Point3d pixel2cam2(const Point2d &p, const Mat &K);

double epipolar_constraint_limit=0.4;//0.6

int main(int argc, char **argv) {

    //-- 读取图像
    Mat img_1 = imread("/media/hl/Stuff/ubuntu_share_2/img/path_ue/stereo-high/left_noshake_higher/path3.0086.png", CV_LOAD_IMAGE_COLOR);//
    Mat img_2 = imread("/media/hl/Stuff/ubuntu_share_2/img/path_ue/stereo-high/right_shake_higher/path3.0086.png", CV_LOAD_IMAGE_COLOR);//100,

    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;//关键点（位置，朝向，大小）
    std::vector<cv::DMatch> good_matches;//用特征点匹配里面的约束限制得到的特征点
    find_feature_matches_sift(img_1,img_2,keypoints_1,keypoints_2,good_matches);
    //find_feature_matches_orb(img_1,img_2,keypoints_1,keypoints_2,good_matches);

    //-- 估计两张图像间运动
    Mat R, t;
    Mat essential_matrix;
    Mat fundamental_matrix;
    Mat K = (Mat_<double>(3, 3) << 985.33890829, 0, 2160.97272569, 0, 985.31527764, 2159.74359191, 0, 0, 1);//DS Model stereo

    pose_estimation_2d2d(keypoints_1, keypoints_2, good_matches, R, t, K, essential_matrix, fundamental_matrix);

    //-- 验证E=t^R*scale
    Mat t_x, scale;
    E_tR_scale(R,t,t_x, essential_matrix, scale);
    cout<<"scale = "<<scale<<endl;//输出scale看是否接近1

    //-- 对极约束限制
    vector<DMatch> better_matches;//再用对极约束限制过滤得到的特征点
    Mat d;
    vector<Point2d> keypoint1,keypoint2;
    for (DMatch m: good_matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//像素平面ud,vd——>归一化平面Pd，就差一个K，所以pixel2cam里只用一个K不用畸变参数
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat dd = y2.t() * t_x * R * y1;     //对极误差
        //cout << "epipolar constraint = " << dd << endl;
        d.push_back(dd);//d格式为[a;b;c;d...]
        keypoint1.push_back(keypoints_1[m.queryIdx].pt);
        keypoint2.push_back(keypoints_2[m.queryIdx].pt);
        //cout<<"特征点坐标: x="<<keypoints_1[m.queryIdx].pt.x<<", y="<<keypoints_1[m.queryIdx].pt.y<<endl;
    }

//    for(int h=0;h<keypoint2.size();h++)
//    {
//        cout<<"特征点坐标: x="<<keypoint2.at(h).x<<endl;//发现keypoint2有问题，为什么很多点x=0，而且有超出图像大小的？
//    }

cout<<img_1.cols<<"aaaaaaa"<<endl;
    Mat d1;//去掉过大异常值的匹配点后，其余点的对极误差。后面在此基础上优化
    for (int i = 0; i < good_matches.size(); i++)
    {
        double d_exam=abs(d.at<double>(i,0));//对极误差绝对值
        //if (d_exam <= epipolar_constraint_limit  && keypoint1.at(i).x>img_1.cols*0.5 && keypoint2.at(count).x<img_2.cols*0.5 && keypoint1.at(count).y>img_1.cols*0.2 && keypoint1.at(count).y<img_1.cols*0.8 && keypoint2.at(count).y>img_1.cols*0.2 && keypoint2.at(count).y<img_1.cols*0.8) {
        if ( d_exam <= epipolar_constraint_limit  && (int(keypoint1.at(i).x)>img_1.cols*0.2) ) {
            better_matches.push_back(good_matches[i]);
            //cout<<"epipolar constraint ="<<d.at<double>(i,0)<<endl;//和上面求的一样，只是去掉了差很大的，这些可能是误匹配，去掉误匹配的得到better_match再用这个匹配去优化对极误差
            d1.push_back(d.at<double>(i,0));
        }
        //}
    }

    cout << "一共找到了" << better_matches.size() << "组更好的匹配点" << endl;
    // 画匹配图
    cv::Mat img_matches_better;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, better_matches, img_matches_better, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    namedWindow("sift_matches_better",0);//代表图片可压缩，否则太大显示不全
    cv::imshow("sift_matches_better", img_matches_better);
    //cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/sift_feature_better_virtual.jpg", img_matches_better);
    //cv::waitKey(0);

    //添加约束：过滤掉对极约束误差很大的点

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points11;
    vector<Point2f> points22;
    Mat fundamental_matrix2;
    Mat essential_matrix2;
    //void pose_estimation_2d2d(keypoints_1, keypoints_2, better_matches, R, t, K, essential_matrix2, fundamental_matrix2);//转换成函数不方便

    for (int i = 0; i < (int) better_matches.size(); i++) {
        points11.push_back(keypoints_1[better_matches[i].queryIdx].pt);
        points22.push_back(keypoints_2[better_matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    fundamental_matrix2 = findFundamentalMat(points11, points22, CV_FM_8POINT);//或FM_RANSAC等方法
    //cout << "better fundamental_matrix is " << endl << fundamental_matrix2 << endl;

    //-- 计算本质矩阵
    vector<Point2f> xp11;
    vector<Point2f> xp22;
    for (int i = 0; i < (int) better_matches.size(); i++)
    {
        xp11.push_back(pixel2cam(points11[i], K));
        xp22.push_back(pixel2cam(points22[i], K));
    }
    essential_matrix2 = findFundamentalMat(xp11, xp22, CV_FM_8POINT);//或FM_RANSAC等方法
    //cout << "better essential_matrix is " << endl << essential_matrix2 << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    // 法一：自己用SVD分解实现
    pose_SVD_self(essential_matrix2,R,t);//测试发现和下面的方法结果一样
    // 法二
    Mat R3,t3;
    Mat KK=(Mat_<double>(3, 3) << 1,0,0,
            0,1,0,
            0,0,1);
    recoverPose(essential_matrix2, xp11, xp22, KK,  R3, t3);
    cout << "R3 is " << endl << R3 << endl;
    cout << "t3 is " << endl << t3 << endl;
    R=R3;
    t=t3;//优化前的位姿

    //固定基线长度
    Mat t_40=(Mat_<double>(3,1)<<t.at<double>(0,0)*40, t.at<double>(1,0)*40, t.at<double>(2,0)*40);//双目t假设定值不变，为40cm。
    t=t_40;
    cout<<"set t="<<t<<endl;

    //-- 验证E=t^R*scale
    Mat t_x2, scale2;
    E_tR_scale(R,t,t_x2, essential_matrix2, scale2);//输出scale看是否接近1
    //cout<<"scale2 = "<<scale2<<endl;//有问题，总有一个数差异很大
    //验证p2' * F * p1：发现p2' * F * p1的结果都接近0
    //pFp(fundamental_matrix2,better_matches,points11,points22);//报错，但是已经在matlab解决

    //优化前欧拉角
    double roll,yaw,pitch;
    euler_angle(R,roll,yaw,pitch);
    cout<<"优化前: roll="<<roll<<", yaw="<<yaw<<", pitch="<<pitch<<endl;

    //三角测量
    VecVector3d pts_3d_eigen;//第一个相机坐标系下的3d坐标(在三角测量里赋值)
    VecVector2d pts_2d_eigen;//第二个相机下的像素坐标

    vector<Point2f> pts_2d;
    for (int i = 0; i < (int) better_matches.size(); i++) {
        pts_2d.push_back(keypoints_2[better_matches[i].trainIdx].pt);
    }
    for (size_t i = 0; i < (int) better_matches.size(); ++i) {
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    //-- 三角化
    triangulation_self(R, t, K,points11,points22,better_matches,keypoints_1,keypoints_2,img_1,img_2,pts_3d_eigen);


    //solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    VecVector3d x1;
    VecVector3d x2;//归一化平面坐标
//用去掉过大异常值的匹配点，求E，然后算对极误差
    Mat d2;
    for(DMatch m: better_matches) {
        Point2d pt11 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//像素平面ud,vd——>归一化平面Pd
        Mat y11 = (Mat_<double>(3, 1) << pt11.x, pt11.y, 1);
        Point2d pt22 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y22 = (Mat_<double>(3, 1) << pt22.x, pt22.y, 1);
        //Mat dd = y22.t() * essential_matrix2 * y11;
        Mat dd = y22.t() * t_x2 * R * y11;
        //cout << "new epipolar constraint = " << dd << endl;
        //cout << "epipolar constraint = " << abs(dd.at<double>(0,0)) << endl;
        d2.push_back(dd);//d2格式为[a;b;c;d...].还没使用

//        x1.push_back(Eigen::Vector3d(y11.at<double>(0,0),y11.at<double>(0,1),y11.at<double>(0,2)));
//        x2.push_back(Eigen::Vector3d(y22.at<double>(0,0),y22.at<double>(0,1),y22.at<double>(0,2)));
        x1.push_back(Eigen::Vector3d(y11.at<double>(0,0),y11.at<double>(1,0),y11.at<double>(2,0)));
        x2.push_back(Eigen::Vector3d(y22.at<double>(0,0),y22.at<double>(1,0),y22.at<double>(2,0)));
    }


    //BA优化得到更好的位姿
    cout << "calling bundle adjustment by g2o" << endl;
    //Sophus::SE3d pose_g2o;
    Sophus::SO3d R_ba;
    Sophus::Matrix3d t_ba;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(x1, x2, R,t, R_ba,t_ba);//BA_func_1，优化对极误差
    //bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);//BA_func_2，优化重投影误差：输入3d点和2d点和相机内参矩阵，输出pose为优化变量 使用BA方法
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve BA by g2o cost time: " << time_used.count() << " seconds." << endl;

//// 视差图
    Mat K_virtual = (Mat_<double>(3, 3) << 800, 0, 460, 0, 800, 600, 0, 0, 1);
    cv::Mat image_re_left = cv::Mat(cv::Size(920,960), img_1.type(), cv::Scalar(0, 0, 0));// 创建三通道黑色图像
    cv::Mat image_re_right = cv::Mat(cv::Size(920,960), img_2.type(), cv::Scalar(0, 0, 0));// 创建三通道黑色图像

    //将R_ba转换为mat
    cv::Mat R_ba_mat;
    cv::eigen2cv(R_ba.matrix(), R_ba_mat);
    double roll_ba,yaw_ba,pitch_ba;
    euler_angle(R_ba_mat,roll_ba,yaw_ba,pitch_ba);

//用鱼眼图像求得的位姿：BA roll yaw pitch = 1.6591, 121.852, 4.83652（真值：0,120,4）
    //用BA_test_10求图4共视区图像位姿,epipolar_constraint_limit=0.4，better roll yaw pitch = 4.21665, 179.997, 0.000628198；BA roll yaw pitch = 3.60562, 179.875, 2.13035
    Eigen::Vector3d euler_angles_ba; // 021:XZY顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    euler_angles_ba[0]=(yaw_ba-120)*M_PI/180;
    euler_angles_ba[1]=-pitch_ba*M_PI/180;
    euler_angles_ba[2]=roll_ba*M_PI/180;
//    euler_angles_ba[2]=-(-yaw_ba+120)*M_PI/180;//yaw，虚拟相机已经分别转了60度，故要减去120
    //T = (Mat_<double>(3, 1) << 28.74392189525434/40, 5.341121286223731/40, -27.29943914232243/40);//图4

    Mat RR;//虚拟相机之间的旋转矩阵
    Eigen::Matrix3d rotation_matrix_set;
    rotation_matrix_set = Eigen::AngleAxisd(euler_angles_ba[0], Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(euler_angles_ba[1], Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(euler_angles_ba[2], Eigen::Vector3d::UnitZ());
    cout << "虚拟相机之间的 yaw pitch roll =\n" << euler_angles_ba[0]<<"," <<euler_angles_ba[1]<<","<<euler_angles_ba[2]<< endl;
    cout << "虚拟相机之间的 rotation matrix =\n" << rotation_matrix_set << endl;
    //将eigen转换成mat
    cv::eigen2cv(rotation_matrix_set, RR);

    euler_angle(RR,roll_ba,yaw_ba,pitch_ba);//输出：虚拟相机之间的 roll yaw pitch

    Eigen::Matrix3d R_virtual = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_virtual2 = Eigen::Matrix3d::Identity();
    R_virtual << 0.5, 0, -0.866, 0, 1, 0, 0.866, 0, 0.5;//绕y轴转60度的旋转矩阵
    R_virtual2 << 0.5, 0, 0.866, 0, 1, 0, -0.866, 0, 0.5;//绕y轴转-60度的旋转矩阵

    //遍历每个像素点
    for (int v = 0; v < image_re_left.rows-0 ; v++)
    {
        for (int u = 0; u < image_re_left.cols-0; u++)
        {
            Eigen::Vector2d pt_virtual_2d;  //虚拟相机像素坐标
            pt_virtual_2d << u, v;

            //虚拟相机像素坐标————虚拟相机下归一化坐标
            Eigen::Vector3d pt_3d;  //虚拟相机下归一化坐标
            virtual_pix_to_virtual_cam(pt_virtual_2d,K_virtual,pt_3d);


            //虚拟相机下归一化坐标————鱼眼相机下归一化坐标
            Eigen::Vector3d pt_2d_fish=R_virtual2*pt_3d;    //鱼眼相机下归一化坐标(2d)  2
            double x_virtual = pt_2d_fish.x()/pt_2d_fish.z();//虚拟相机坐标系下的归一化坐标
            double y_virtual = pt_2d_fish.y()/pt_2d_fish.z();//虚拟相机坐标系下的归一化坐标
            Eigen::Vector3d pt_3d_fish;
            pt_3d_fish<<x_virtual,y_virtual,1;

            //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
            Eigen::Vector2d fish_pix;//鱼眼相机像素坐标
            fish_cam_to_fish_pix(pt_3d_fish,K,fish_pix);

            //传递颜色
            if(fish_pix.x()>img_1.cols*1/3 && fish_pix.y()>img_1.rows*0/3 && fish_pix.x()<(img_1.cols+0) && fish_pix.y()<img_1.rows*3/3)
            {
                image_re_left.at<Vec3b>(v, u)[0] = img_1.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[0];
                image_re_left.at<Vec3b>(v, u)[1] = img_1.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[1];
                image_re_left.at<Vec3b>(v, u)[2] = img_1.at<Vec3b>(int(fish_pix.y()), int(fish_pix.x()))[2];
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
            virtual_pix_to_virtual_cam(pt_virtual_2d_,K_virtual,pt_3d_);


            //虚拟相机下归一化坐标————鱼眼相机下归一化坐标
            Eigen::Vector3d pt_2d_fish_=R_virtual*pt_3d_;    //鱼眼相机下归一化坐标(2d)
            double x_virtual_ = pt_2d_fish_.x()/pt_2d_fish_.z();//虚拟相机坐标系下的归一化坐标
            double y_virtual_ = pt_2d_fish_.y()/pt_2d_fish_.z();//虚拟相机坐标系下的归一化坐标
            Eigen::Vector3d pt_3d_fish_;
            pt_3d_fish_<<x_virtual_,y_virtual_,1;

            //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
            Eigen::Vector2d fish_pix_;//鱼眼相机像素坐标
            fish_cam_to_fish_pix(pt_3d_fish_,K,fish_pix_);

            //传递颜色
            if(fish_pix_.x()>0 && fish_pix_.y()>img_2.rows*0/3 && fish_pix_.x()<(img_2.cols*2/3) && fish_pix_.y()<img_2.rows*3/3)
            {
                image_re_right.at<Vec3b>(v_, u_)[0] = img_2.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[0];
                image_re_right.at<Vec3b>(v_, u_)[1] = img_2.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[1];
                image_re_right.at<Vec3b>(v_, u_)[2] = img_2.at<Vec3b>(int(fish_pix_.y()), int(fish_pix_.x()))[2];
            }
            else
            {
                image_re_right.at<Vec3b>(v_, u_)[0] = 0;
                image_re_right.at<Vec3b>(v_, u_)[1] = 0;
                image_re_right.at<Vec3b>(v_, u_)[2] = 0;
            }
        }
    }
    imshow("Left Image Reshaped", image_re_left);
    imshow("Right Image Reshaped", image_re_right);

    ////from disparity_only_1.cpp

    cv::Mat left_image;
    cv::Mat right_image;

    // RGB转换成GRAY
    cv::cvtColor(image_re_left,left_image,cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_re_right,right_image,cv::COLOR_BGR2GRAY);

    // Load camera matrices
    Mat camera_matrix1 = K_virtual; // Load camera matrix for left camera
    Mat camera_matrix2 = K_virtual; // Load camera matrix for right camera
    Mat distortion_coeffs1 = (Mat_<double>(4, 1) << 0,0,0,0); // Load distortion coefficients for left camera
    Mat distortion_coeffs2 = (Mat_<double>(4, 1) << 0,0,0,0); // Load distortion coefficients for right camera

    // Compute relative essential matrix and fundamental matrix
    // Mat relative_rotation, relative_translation;
    // Mat essential_matrix = findEssentialMat(..., ...);
    // Mat fundamental_matrix = findFundamentalMat(..., ...);
    Mat R_0,T;
    R_0 = (Mat_<double>(3, 3) << 1,0,0,0,1,0,0,0,1);//用noshake图像时，用R_0
    //平移向量
    T = (Mat_<double>(3, 1) << 0, 1, 0);


    // Compute rectification transform matrices
    Mat R1, R2, P1, P2, Q;
    stereoRectify(camera_matrix1, distortion_coeffs1, camera_matrix2, distortion_coeffs2,
                  left_image.size(), RR, T, R1, R2, P1, P2, Q);
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

//    // 对图像进行高斯滤波
//    GaussianBlur(left_image_rectified, left_image_rectified, Size(5, 5), 0);
//    /* 高斯滤波核的大小是根据图像的特征和噪声程度来决定的。一般来说，核的大小越大，滤波效果越平滑，但也会导致图像细节的丢失。因此，在选择核的大小时需要权衡平滑程度和细节保留程度。对于一般的图像，常见的核大小为3x3、5x5、7x7等。*/
//    GaussianBlur(right_image_rectified, right_image_rectified, Size(5, 5), 0);

    // Display rectified images
    imshow("Left Image Rectified", left_image_rectified);
    imshow("Right Image Rectified", right_image_rectified);
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
    int numDisparity=16*5;//numDisparity:视差搜索范围长度，其值必须为16的整数倍。确定立体/深度图的分辨率。最大视差 maxDisparity = minDisparity + numDisparities -1；
    int blockSize=3*3;//SAD代价计算窗口大小,默认大小为5。窗口大小为奇数，一般在3*3 到21*21之间。
    int disp12MaxDiff=200;//disp12MaxDiff：左右一致性检测最大容许误差阈值,默认为-1，即不执行左右视差检查。如果大于0，那么将会被执行。
    int preFilterCap=0;//preFilterCap：预过滤器的截断值。预过滤器是为了去除图像噪声。它的值越大，图像越清晰，但是会增加计算量。
    int uniquenessRatio=2;//uniquenessRatio：uniquenessRatio主要可以防止误匹配。视差唯一性百分比，通常为5~15.
    int speckleWindowSize=130;//视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
    int speckleRange=25;//speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            //0, 96, 9, 4 * 3 * 9, 16 * 3 * 9, 20, 0, 0, 27, 3);    // 神奇的参数 8*10* blockSize
            minDisparity, numDisparity, blockSize, 6 * 4 * blockSize, 24 * 4 * blockSize, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange);    // 神奇的参数
    //0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 20, 0, 0, 27, 3);    // 神奇的参数
    //0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数


    cv::Mat disparity_sgbm, disparity;
//    sgbm->compute(image_re_left, image_re_right, disparity_sgbm);
    sgbm->compute(left_image_rectified, right_image_rectified, disparity_sgbm);
    //sgbm->compute(image_re_left, image_re_right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    namedWindow("左目原图",CV_WINDOW_NORMAL);
    namedWindow("右目原图",CV_WINDOW_NORMAL);
    namedWindow("左目修正",CV_WINDOW_NORMAL);
    namedWindow("右目修正",CV_WINDOW_NORMAL);
    namedWindow("视差图",CV_WINDOW_NORMAL);
    cv::imshow("左目原图",img_1);
    cv::imshow("右目原图",img_2);
    cv::imshow("左目修正",image_re_left);
    cv::imshow("右目修正",image_re_right);
    cv::imshow("disparity", disparity / 96.0);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/virtual_reticify_left_127.png", image_re_left);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/virtual_reticify_right_127.png", image_re_right);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/virtual_epip_left_50.png", left_image_rectified);
//    cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/virtual_epip_right_50.png", right_image_rectified);
    cv::waitKey(0);

    //保存视差图（https://blog.csdn.net/nzise_se/article/details/78489554）
    cv::Mat disp8U = Mat(disparity.rows, disparity.cols, CV_8UC1);       //显示
    normalize(disparity, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    //cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/ue_disp_ba4.png", disp8U);


    return 0;
}

void find_feature_matches_sift(const Mat &img_1, const Mat &img_2,
                               std::vector<KeyPoint> &keypoints_1,
                               std::vector<KeyPoint> &keypoints_2,
                               std::vector<DMatch> &good_matches) {
    cv::Ptr<cv::SiftFeatureDetector> sift = cv::SiftFeatureDetector::create();
    //提取SIFT关键点
    sift->detect(img_1, keypoints_1);
    sift->detect(img_2, keypoints_2);
    //计算特征描述子
    cv::Mat descriptors1, descriptors2;//描述子（周围像素信息）
    sift->compute(img_1, keypoints_1, descriptors1);
    sift->compute(img_2, keypoints_2, descriptors2);
    //特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    //KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn_matches;
    const float ratio_thresh = 0.7f;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
    for (auto & knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    cout << "sift一共找到了" << good_matches.size() << "组匹配点" << endl;

    // 画匹配图
    cv::Mat img_matches_knn;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches_knn, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    namedWindow("sift_matches",0);//代表图片可压缩，否则太大显示不全
    cv::imshow("sift_matches", img_matches_knn);
    //cv::imwrite("/home/hl18/Desktop/share/img/real_fisheye/result/sift_feature_virtual.jpg", img_matches_knn);
    //cv::waitKey(0);

}

void find_feature_matches_orb(const Mat &img_1, const Mat &img_2,
                              std::vector<KeyPoint> &keypoints_1,
                              std::vector<KeyPoint> &keypoints_2,
                              std::vector<DMatch> &good_matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;//描述子（周围像素信息）
    Ptr<FeatureDetector> detector = ORB::create();//申请ORB detector
    Ptr<DescriptorExtractor> descriptor = ORB::create();//申请ORB descriptor
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//申请匹配（匹配在构造时可以将距离度量范数传入）
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);//提取keypoint

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);//根据keypoint求descriptor

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    namedWindow("ORB features",0);//代表图片可压缩，否则太大显示不全
    imshow("ORB features", outimg1);//画出提取的ORB特征点
    //waitKey();

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, matches);//用match对descriptor算匹配，返回DMatch结构

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;
//    auto min_max = minmax_element(matches.begin(), matches.end(),
//                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
//    double min_dist = min_max.first->distance;
//    double max_dist = min_max.second->distance;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    //绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);//所有匹配点
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);//经过orb内部参数筛选过的特征点
    namedWindow("all matches",0);
    imshow("all matches", img_match);
    namedWindow("good matches",0);
    imshow("good matches", img_goodmatch);
    //cv::imwrite("save.jpg", img_goodmatch);
    //waitKey(0);
}

void E_tR_scale(Mat &R, Mat &t, Mat &t_x, Mat &essential_matrix, Mat &scale)
{
    t_x =   //t^
            (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    Mat t_R=t_x * R;
    scale =   //t^
            (Mat_<double>(3, 3) << essential_matrix.at<double>(0,0)/t_R.at<double>(0,0), essential_matrix.at<double>(0,1)/t_R.at<double>(0,1), essential_matrix.at<double>(0,2)/t_R.at<double>(0,2),
                    essential_matrix.at<double>(1,0)/t_R.at<double>(1,0), essential_matrix.at<double>(1,1)/t_R.at<double>(1,1), essential_matrix.at<double>(1,2)/t_R.at<double>(1,2),
                    essential_matrix.at<double>(2,0)/t_R.at<double>(2,0), essential_matrix.at<double>(2,1)/t_R.at<double>(2,1), essential_matrix.at<double>(2,2)/t_R.at<double>(2,2));
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {//得到归一化坐标

//    double alpha=0.59417224;
//    double ksi=-0.0420613;
    double alpha=0.59561123;
    double ksi=-0.01119064;//april2
    double u=p.x;
    double v=p.y;
    double cx=K.at<double>(0,2);
    double cy=K.at<double>(1,2);
    double fx=K.at<double>(0,0);
    double fy=K.at<double>(1,1);//https://blog.csdn.net/duiwangxiaomi/article/details/93075571
    double mx=(u-cx)/fx;
    double my=(v-cy)/fy;
    double r =sqrt(mx*mx+my*my);
    double mz=(1-alpha*alpha*r*r)/(alpha*sqrt(1-(2*alpha-1)*r*r)+1-alpha);
    double xp_x=(mz*ksi+sqrt(mz*mz+(1-ksi*ksi)*r*r))/(mz*mz+r*r)*mx;
    double xp_y=(mz*ksi+sqrt(mz*mz+(1-ksi*ksi)*r*r))/(mz*mz+r*r)*my;
    double xp_z=(mz*ksi+sqrt(mz*mz+(1-ksi*ksi)*r*r))/(mz*mz+r*r)*mz-ksi;

    return Point2d(xp_x/xp_z,xp_y/xp_z);//归一化坐标
}

Point3d pixel2cam2(const Point2d &p, const Mat &K) {//得到未归一化的坐标，即(xp_x,xp_y,xp_z)，也即(mx,my,mz)

//    double alpha=0.59417224;
//    double ksi=-0.0420613;
    double alpha=0.59561123;
    double ksi=-0.01119064;//april2
    double u=p.x;
    double v=p.y;
    double cx=K.at<double>(0,2);
    double cy=K.at<double>(1,2);
    double fx=K.at<double>(0,0);
    double fy=K.at<double>(1,1);//https://blog.csdn.net/duiwangxiaomi/article/details/93075571
    double mx=(u-cx)/fx;
    double my=(v-cy)/fy;
    double r =sqrt(mx*mx+my*my);
    double mz=(1-alpha*alpha*r*r)/(alpha*sqrt(1-(2*alpha-1)*r*r)+1-alpha);
    double xp_x=(mz*ksi+sqrt(mz*mz+(1-ksi*ksi)*r*r))/(mz*mz+r*r)*mx;
    double xp_y=(mz*ksi+sqrt(mz*mz+(1-ksi*ksi)*r*r))/(mz*mz+r*r)*my;
    double xp_z=(mz*ksi+sqrt(mz*mz+(1-ksi*ksi)*r*r))/(mz*mz+r*r)*mz-ksi;

    return Point3d(xp_x,xp_y, xp_z);//归一化坐标
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> good_matches,
                          Mat &R, Mat &t, Mat &K, Mat &essential_matrix, Mat &fundamental_matrix) {

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) good_matches.size(); i++) {
        points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    //Mat essential_matrix;
    vector<Point2f> xp1;
    vector<Point2f> xp2;
    for (int i = 0; i < (int) good_matches.size(); i++) {
//        double alpha=0.59417224;
//        double ksi=-0.0420613;
//        double alpha=0.57716941;
//        double ksi=-0.12002858;
        double alpha=0.59561123;
        double ksi=-0.01119064;//april2
        double u1=points1[i].x;
        double v1=points1[i].y;//https://www.coder.work/article/1229297#:~:text=%E6%9C%80%E4%BD%B3%E7%AD%94%E6%A1%88%20%E5%BD%93%E6%88%91%E9%9C%80%E8%A6%81%E4%BB%8E%E5%83%8F%E6%82%A8%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E7%BB%84%E4%B8%AD%E5%88%86%E5%88%AB%E6%8F%90%E5%8F%96%20X%20%E5%92%8C%20Y%20%E5%80%BC%E6%97%B6%EF%BC%8C%E6%88%91%E6%98%AF%E8%BF%99%E6%A0%B7%E5%81%9A%E7%9A%84%3A%20std%20%3A%3A,y%20%3D%20corners%20%5Bk%5D.y%3B%20%2F%2Fsecond%20value%20%2F%2Fstuff%20%7D
        double u2=points2[i].x;
        double v2=points2[i].y;
        double cx=K.at<double>(0,2);
        double cy=K.at<double>(1,2);
        double fx=K.at<double>(0,0);
        double fy=K.at<double>(1,1);//https://blog.csdn.net/duiwangxiaomi/article/details/93075571
        double mx1=(u1-cx)/fx;
        double my1=(v1-cy)/fy;
        double mx2=(u2-cx)/fx;
        double my2=(v2-cy)/fy;
        double r1 =sqrt(mx1*mx1+my1*my1);
        double r2 =sqrt(mx2*mx2+my2*my2);
        double mz1=(1-alpha*alpha*r1*r1)/(alpha*sqrt(1-(2*alpha-1)*r1*r1)+1-alpha);
        double mz2=(1-alpha*alpha*r2*r2)/(alpha*sqrt(1-(2*alpha-1)*r2*r2)+1-alpha);
        double xp1_x=(mz1*ksi+sqrt(mz1*mz1+(1-ksi*ksi)*r1*r1))/(mz1*mz1+r1*r1)*mx1;
        double xp1_y=(mz1*ksi+sqrt(mz1*mz1+(1-ksi*ksi)*r1*r1))/(mz1*mz1+r1*r1)*my1;
        double xp1_z=(mz1*ksi+sqrt(mz1*mz1+(1-ksi*ksi)*r1*r1))/(mz1*mz1+r1*r1)*mz1-ksi;
        double xp2_x=(mz2*ksi+sqrt(mz2*mz2+(1-ksi*ksi)*r2*r2))/(mz2*mz2+r2*r2)*mx2;
        double xp2_y=(mz2*ksi+sqrt(mz2*mz2+(1-ksi*ksi)*r2*r2))/(mz2*mz2+r2*r2)*my2;
        double xp2_z=(mz2*ksi+sqrt(mz2*mz2+(1-ksi*ksi)*r2*r2))/(mz2*mz2+r2*r2)*mz2-ksi;
        Point2f xp1_norm=Point2f(xp1_x/xp1_z,xp1_y/xp1_z);//归一化坐标
        Point2f xp2_norm=Point2f(xp2_x/xp2_z,xp2_y/xp2_z);//归一化坐标
        xp1.push_back(xp1_norm);
        xp2.push_back(xp2_norm);
    }

    //essential_matrix = findFundamentalMat(xp1, xp2, CV_FM_8POINT);//这个函数好像是直接调用八点法求解。后面证实是对的：这里基础矩阵等于本质矩阵, 因为归一化坐标已经去除了内参的影响（https://blog.csdn.net/m0_47096428/article/details/119809340）
    essential_matrix = findFundamentalMat(xp1, xp2);//这个函数好像是直接调用八点法求解。后面证实是对的：这里基础矩阵等于本质矩阵, 因为归一化坐标已经去除了内参的影响（https://blog.csdn.net/m0_47096428/article/details/119809340）
    cout << "essential_matrix is " << endl << essential_matrix << endl;

//    //-- 计算单应矩阵
//    //-- 但是本例中场景不是平面，单应矩阵意义不大
//    Mat homography_matrix;
//    homography_matrix = findHomography(points1, points2, RANSAC, 3);
//    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    Mat KK=(Mat_<double>(3, 3) << 1,0,0,
            0,1,0,
            0,0,1);
    recoverPose(essential_matrix, xp1, xp2,  KK, R, t);
//    cout << "R is " << endl << R << endl;
//    cout << "t is " << endl << t << endl;

}

void pFp(Mat &fundamental_matrix,vector<DMatch> &better_matches,vector<Point2f> &points11,vector<Point2f> &points22)
{
    //验证p2' * F * p1：发现p2' * F * p1的结果都接近0
    Mat F=Mat::ones(3,3,CV_64F);
    F =(Mat_<float>(3, 3) << fundamental_matrix.at<double>(0, 0), fundamental_matrix.at<double>(0, 1), fundamental_matrix.at<double>(0, 2),
            fundamental_matrix.at<double>(1, 0), fundamental_matrix.at<double>(1, 1), fundamental_matrix.at<double>(1, 2),
            fundamental_matrix.at<double>(2, 0), fundamental_matrix.at<double>(2, 1), fundamental_matrix.at<double>(2, 2));
    cout<<"F = "<<F<<endl;
    for (int i = 0; i < (int) better_matches.size(); i++)
    {
        Mat p1=Mat::ones(3,1,CV_64F);
        Mat p2=Mat::ones(3,1,CV_64F);
        Mat pFp=Mat::ones(1,1,CV_32F);
        p1 =(Mat_<float>(3,1)<<points11[i].x,points11[i].y,1);
        p2 =(Mat_<float>(3,1)<<points22[i].x,points22[i].y,1);
        cout<<"p1 = "<<p1<<endl;
        cout<<"p2 = "<<p2<<endl;
        cout<<"fundamental_matrix2 = "<<F<<endl;
        std::cout << "p1类型： " << p1.type() << std::endl;
        std::cout << "p2类型： " << p2.type() << std::endl;
        std::cout << "F类型： " << F.type() << std::endl;
        std::cout << "pFp类型： " << pFp.type() << std::endl;//https://blog.csdn.net/iLOVEJohnny/article/details/105658771

        Mat aaa =p1.t() * fundamental_matrix * p2;
        pFp=(Mat_<float>(1,1)<<aaa.at<Vec3b>(0, 0)[0]);
        //pFp=p1.t() * fundamental_matrix2 * p2;
        cout<<"p2^T * F * p1 ="<< pFp <<endl;

//失败，一直报错error: (-215:Assertion failed) type == B.type() in function 'gemm'。改为在matlab求解
    }
}

void euler_angle(Mat &R, double &roll,double &yaw,double &pitch)
{
    // 欧拉角 四元数
    Eigen::Matrix<double, 3, 3> R_matrix;
    cv::cv2eigen(R, R_matrix); // cv::Mat 转换成 Eigen::Matrix
    //Eigen::Vector3d euler_angles = R_matrix.eulerAngles(0, 2, 1); // 021:XZY顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    Eigen::Vector3d euler_angles = R_matrix.eulerAngles(1, 0,2); // 021:ZYX顺序，即rpy顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    yaw=euler_angles[0]*180/M_PI;
    pitch=euler_angles[1]*180/M_PI;
    roll=euler_angles[2]*180/M_PI;
    //cout << "better yaw pitch roll = " << euler_angles.transpose() << endl;

    //通过下面判断处理得到的欧拉角，得到的正负号可能会有问题，再用其复成旋转矩阵，可能会出问题，因此专门建一个变量用来显示欧拉角数值，但不更改直接求到的欧拉角，这样此函数传回的欧拉角虽然看起来数字可能不直观，但能够准确回复出旋转矩阵
    double roll_fake=roll;
    double yaw_fake=yaw;
    double pitch_fake=pitch;
    if(roll_fake<-90)//-170
    {
        roll_fake=180+roll_fake;
    }
    if(roll_fake>90)//170
    {
        roll_fake=180-roll_fake;
    }
    if(pitch_fake>90)//150
    {
        pitch_fake=180-pitch_fake;
    }
    if(pitch_fake<-90)//-150
    {
        pitch_fake=180+pitch_fake;
    }
    if(yaw_fake>0&&yaw_fake<90)//60
    {
        yaw_fake=180-yaw_fake;
    }
    if(yaw_fake>-90&&yaw_fake<0)//-60
    {
        yaw_fake=180+yaw_fake;
    }
    roll=roll_fake;
    yaw=yaw_fake;
    pitch=pitch_fake;
    cout << "yaw pitch roll = " << yaw_fake <<", "<< pitch_fake <<", "<< roll_fake << endl;//转化成角度制显示
    Eigen::Quaterniond qq=Eigen::Quaterniond(R_matrix);
    cout << "quaternion(x,y,z,w) = " << qq.coeffs().transpose() << endl;//
    //cout<<"better t="<<t<<endl;
}

void pose_SVD_self(Mat &essential_matrix, Mat &R, Mat &t)
{
    //https://zhuanlan.zhihu.com/p/434787470
    //-- 从本质矩阵中恢复旋转和平移信息.
    Mat R11,R22,tt= cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat uu,ww,vtt;
    cv::SVD::compute(essential_matrix,ww,uu,vtt);
    uu.col(2).copyTo(tt);
    tt=tt/cv::norm(tt);

    Mat WW =
            (Mat_<double>(3, 3) << 0, -1, 0,
                    1, 0, 0,
                    0, 0, 1);

    R11 = uu*WW*vtt;

    if(cv::determinant(R11)<0)
        R11=-R11;

    R22 = uu*WW.t()*vtt;
    if(cv::determinant(R22)<0)
        R22=-R22;
    cout << "R11 is " << endl << R11 << endl;
    cout << "R22 is " << endl << R22 << endl;
    cout << "tt is " << endl << tt << endl;
    //R=R22;//要通过结果判断选哪一个R
    //t=tt;

    ///////////////////////
/*    Mat R1,R2;
    Mat U, W, VT;
    SVD::compute(essential_matrix, W, U, VT);//https://blog.csdn.net/u012198575/article/details/99548136
    cout<<"W = "<<W<<endl;
    cout<<"U = "<<U<<endl;
    cout<<"VT = "<<VT<<endl;
    Mat W_x=(Mat_<double>(3, 3) << 0, -W.at<double>(2, 0), W.at<double>(1, 0),//W转置
            W.at<double>(2, 0), 0, -W.at<double>(0, 0),
            -W.at<double>(1, 0), W.at<double>(0, 0), 0);
    Mat RzT =
            (Mat_<double>(3, 3) << 0, -1, 0,
                    1, 0, 0,
                    0, 0, 1);
    R1=U*RzT*VT;//这句有问题，求的还是不对
    R2=U*RzT.t()*VT;//https://blog.csdn.net/m0_47096428/article/details/119809340
    R=R1;
    //R=R2;//二选一，看哪个结果是对的
    cout << "R is " << endl << R << endl;
    //cout << "t is " << endl << t << endl;//t先不求了*/

    //void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
}


void triangulation_self(Mat &R, Mat &t, Mat &K,vector<Point2f> &points11,vector<Point2f> &points22,vector<DMatch> &better_matches,vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,Mat &img_1,Mat &img_2,VecVector3d &pts_3d_eigen)
{
    //-- 三角化
    vector<Point3d> points;//存放3d点
    //triangulation(keypoints_1, keypoints_2, better_matches, R, t, K, points);
    vector<Point3f> xp1_3d;
    vector<Point3f> xp2_3d;//未归一化的xp
    //vector<double> disp;
    for (int i = 0; i < (int) better_matches.size(); i++)
    {
        xp1_3d.push_back(pixel2cam2(points11[i], K));
        xp2_3d.push_back(pixel2cam2(points22[i], K));
    }

    Mat T1=(Mat_<double>(3,4)<<1,0,0,0,
            0,1,0,0,
            0,0,1,0);
    Mat T2=(Mat_<double>(3,4) << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),t.at<double>(0,0),
            R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),t.at<double>(1,0),
            R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),t.at<double>(2,0));
    //cout<<"T2 = "<<T2<<endl;

    fstream f;
    for (int i = 0; i < better_matches.size(); i++)//对每一组匹配点计算深度
    {
        Mat A1=(Mat_<double>(1,4) << xp1_3d[i].x * T1.at<double>(2,0) - xp1_3d[i].z * T1.at<double>(0,0),
                xp1_3d[i].x * T1.at<double>(2,1) - xp1_3d[i].z * T1.at<double>(0,1),
                xp1_3d[i].x * T1.at<double>(2,2) - xp1_3d[i].z * T1.at<double>(0,2),
                xp1_3d[i].x * T1.at<double>(2,3) - xp1_3d[i].z * T1.at<double>(0,3));
        Mat A2=(Mat_<double>(1,4) << xp1_3d[i].y * T1.at<double>(2,0) - xp1_3d[i].z * T1.at<double>(1,0),
                xp1_3d[i].y * T1.at<double>(2,1) - xp1_3d[i].z * T1.at<double>(1,1),
                xp1_3d[i].y * T1.at<double>(2,2) - xp1_3d[i].z * T1.at<double>(1,2),
                xp1_3d[i].y * T1.at<double>(2,3) - xp1_3d[i].z * T1.at<double>(1,3));
        Mat A3=(Mat_<double>(1,4) << xp2_3d[i].x * T2.at<double>(2,0) - xp2_3d[i].z * T2.at<double>(0,0),
                xp2_3d[i].x * T2.at<double>(2,1) - xp2_3d[i].z * T2.at<double>(0,1),
                xp2_3d[i].x * T2.at<double>(2,2) - xp2_3d[i].z * T2.at<double>(0,2),
                xp2_3d[i].x * T2.at<double>(2,3) - xp2_3d[i].z * T2.at<double>(0,3));
        Mat A4=(Mat_<double>(1,4) << xp2_3d[i].y * T2.at<double>(2,0) - xp2_3d[i].z * T2.at<double>(1,0),
                xp2_3d[i].y * T2.at<double>(2,1) - xp2_3d[i].z * T2.at<double>(1,1),
                xp2_3d[i].y * T2.at<double>(2,2) - xp2_3d[i].z * T2.at<double>(1,2),
                xp2_3d[i].y * T2.at<double>(2,3) - xp2_3d[i].z * T2.at<double>(1,3));
        Mat A;//用于Ax=0解x
        cv::Mat matArray[] = { A1,A2,A3,A4};
        cv::vconcat(matArray,4,A); // 等同于A=[A1 ; A2; A3; A4] https://blog.csdn.net/jndingxin/article/details/119597245
        cv::Mat U,W,VT;
        cv::SVD::compute(A,W,U,VT);
        Mat V = VT.t();
        Mat world_point_4d;
        V.col(3).copyTo(world_point_4d);//未归一化
        world_point_4d /= world_point_4d.at<double>(3,0);
        Point3d world_point(world_point_4d.at<double>(0,0),
                            world_point_4d.at<double>(1,0),
                            world_point_4d.at<double>(2,0));
        //cout<<world_point<<endl;
        points.push_back(world_point);//得到的是在第一个相机坐标系下的相机坐标

        //追加写入,在原来基础上加了ios::app
        f.open("/home/hl18/projects/fisheye_project/data.txt",ios::out|ios::app);
        //输入你想写入的内容
        //f<<world_point.x<<" "<<world_point.y<<" "<<world_point.z<<endl;
        f.close();

        double distance = sqrt(world_point.x*world_point.x+world_point.y*world_point.y+world_point.z*world_point.z);
        //cout << "distance: " << distance << endl;//第一个相机坐标系下的坐标到第一个相机原点的距离

        //double depth=world_point.x*0.5+world_point.z*0.866;
        double depth=world_point.x*0.866+world_point.z*0.5;//从matlab画图结果反推得到坐标系的样子
        //cout << "depth: " << depth << endl;//好像能达到基本都是正数，但也有出现几个负数的情况。

        pts_3d_eigen.push_back(Eigen::Vector3d(world_point.x, world_point.y, world_point.z));
        //disp.push_back(K.at<double>(0,0)*40/depth);   //depth = ( f * baseline) / disp
    }
    //cv::imshow("disparity", disp / 96.0);

    ///////////////////////////////////////////////////////

    //验证三角化点与特征点的重投影关系(通过triangulation函数最后的归一化操作，把3D坐标重投影到两个相机的归一化平面上，从而计算重投影误差)
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    for (int i = 0; i < better_matches.size(); i++) {
        //for (int i = 0; i < 2; i++) {
        // 第一个图
        //float depth1 = points[i].z;
        float depth1 = points[i].x*0.866+points[i].z*0.5;
        //cout << "depth: " << depth1 << endl;
        //Point2d pt1_cam = pixel2cam(keypoints_1[better_matches[i].queryIdx].pt, K);//这句干啥用的？源代码就这么写的
        cv::circle(img1_plot, keypoints_1[better_matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // 第二个图
/*        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;//通过方程 x2 = R*x1 + t 对相机1下的3D相机坐标系坐标进行重投影，得到相机2下的3D坐标
        //float depth2 = pt2_trans.at<double>(2, 0);
        float depth2 = pt2_trans.at<double>(0, 0)*0.866+pt2_trans.at<double>(2, 0)*0.5;//有问题
        cout << "depth2: " << depth2 << endl;
        cv::circle(img2_plot, keypoints_2[better_matches[i].trainIdx].pt, 2, get_color(depth2), 2);//两个图上的匹配点颜色不一样，由深度决定*/
    }
//    for (int i = 0; i < better_matches.size(); i++) {
//        // 第一个图
//        float depth1 = points[i].z;//单位是什么？
//        cout << "depth: " << depth1 << endl;
//        //Point2d pt1_cam = pixel2cam(keypoints_1[better_matches[i].queryIdx].pt, K);//这句干啥用的？源代码就这么写的
//        cv::circle(img1_plot, keypoints_1[better_matches[i].queryIdx].pt, 2, get_color(depth1), 2);
//
//        // 第二个图
//        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;//通过方程 x2 = R*x1 + t 对第一帧3D坐标进行重投影，得到第二帧3D坐标
//        float depth2 = pt2_trans.at<double>(2, 0);
//        cv::circle(img2_plot, keypoints_2[better_matches[i].trainIdx].pt, 2, get_color(depth2), 2);//两个图上的匹配点颜色不一样，由深度决定
//    }

//    cv::imshow("img 1", img1_plot);
//    cv::imshow("img 2", img2_plot);
    //cv::waitKey();

    vector<Point3d> points2;//存放3d点(相机2坐标系下)
    for (int i = 0; i < better_matches.size(); i++)
    {
        Mat point2 = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;//通过方程 x2 = R*x1 + t 对相机1下的3D相机坐标系坐标进行重投影，得到相机2下的3D坐标
        Point3d world_point2(point2.at<double>(0,0),
                             point2.at<double>(1,0),
                             point2.at<double>(2,0));
        points2.push_back(world_point2);//得到的是在第一个相机坐标系下的相机坐标

        //代入DS模型正投影方程，得到相机2下的重投影像素坐标
        // ...

    }
}

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

void fish_cam_to_fish_pix(const Eigen::Vector3d &pt_3d_fish,const cv::Mat &K,Eigen::Vector2d &fish_pix)
{
    //鱼眼相机下归一化坐标————鱼眼相机像素坐标（鱼眼相机投影函数）
    double cx=K.at<double>(0,2);
    double cy=K.at<double>(1,2);
    double fx=K.at<double>(0,0);
    double fy=K.at<double>(1,1);

    double alpha=0.59561123;
    double ksi=-0.01119064;//april2





    double xx=pt_3d_fish.x();
    double yy=pt_3d_fish.y();
    double zz=pt_3d_fish.z();
    double d1= sqrt(xx*xx+yy*yy+zz*zz);
    double d2= sqrt(xx*xx+yy*yy+(ksi*d1+zz)*(ksi*d1+zz));
    double u_fish=fx*xx/(alpha*d2+(1-alpha)*(ksi*d1+zz))+cx;
    double v_fish=fy*yy/(alpha*d2+(1-alpha)*(ksi*d1+zz))+cy;
    fish_pix<<u_fish,v_fish;
}