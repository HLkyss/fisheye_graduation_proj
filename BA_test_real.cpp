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

//Point2d pixel2cam_pin(const Point2d &p, const Mat &K);
void E_tR_scale(Mat &R, Mat &t, Mat &t_x, Mat &essential_matrix, Mat &scale);
void euler_angle(Mat &R);
void pFp(Mat &fundamental_matrix,vector<DMatch> &better_matches,vector<Point2f> &points11,vector<Point2f> &points22);

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

void triangulation_self(Mat &R, Mat &t, Mat &K1,Mat &K2,double &alpha1,double &alpha2,double &ksi1,double &ksi2,
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
        Mat &R, Mat &t, Mat &K1,Mat &K2,const double &alpha1,const double &alpha2,const  double &ksi1,const double &ksi2, Mat &essential_matrix, Mat &fundamental_matrix);

void pose_SVD_self(Mat &essential_matrix, Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K, double &alpha, double &ksi);
//像素坐标转相机坐标(假想曲面上的xp)
Point3d pixel2cam2(const Point2d &p, const Mat &K, double &alpha, double &ksi);

double epipolar_constraint_limit=0.8;//0.6

int main(int argc, char **argv) {

    //-- 读取图像
//    Mat img_1 = imread("/home/hl18/Desktop/share/img/real_fisheye/result/unfish_left_out2.png", CV_LOAD_IMAGE_COLOR);
//    Mat img_2 = imread("/home/hl18/Desktop/share/img/real_fisheye/result/unfish_right_out2.png", CV_LOAD_IMAGE_COLOR);//修正后图像
    Mat img_1 = imread("/media/hl/Stuff/ubuntu_share_2/img/real_fisheye/out_left1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread("/media/hl/Stuff/ubuntu_share_2/img/real_fisheye/out_right1.jpg", CV_LOAD_IMAGE_COLOR);//鱼眼原图


    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;//关键点（位置，朝向，大小）
    std::vector<cv::DMatch> good_matches;//用特征点匹配里面的约束限制得到的特征点
    find_feature_matches_sift(img_1,img_2,keypoints_1,keypoints_2,good_matches);
    //find_feature_matches_orb(img_1,img_2,keypoints_1,keypoints_2,good_matches);

    //-- 估计两张图像间运动
    Mat R, t;
    Mat essential_matrix;
    Mat fundamental_matrix;

    Mat K1 = (Mat_<double>(3, 3) << 396.01307103, 0, 708.54459392, 0, 395.81900138, 464.01861501, 0, 0, 1);//鱼眼相机1
    Mat K2 = (Mat_<double>(3, 3) << 416.68602176, 0, 617.57644155, 0, 415.75414779, 449.3526344, 0, 0, 1);//鱼眼相机2
//    Mat K1 = (Mat_<double>(3, 3) << 800, 0, 700, 0, 800, 800, 0, 0, 1);//虚拟相机1
//    Mat K2 = (Mat_<double>(3, 3) << 800, 0, 350, 0, 800, 800, 0, 0, 1);//虚拟相机2
    double alpha1=0.61896438;
    double ksi1=-0.03234126;
    double alpha2=0.64304057;
    double ksi2=-0.00438942;

    pose_estimation_2d2d(keypoints_1, keypoints_2, good_matches, R, t, K1,K2, alpha1,alpha2,ksi1,ksi2, essential_matrix, fundamental_matrix);

    //-- 验证E=t^R*scale
    Mat t_x, scale;
    E_tR_scale(R,t,t_x, essential_matrix, scale);
    cout<<"scale = "<<scale<<endl;//输出scale看是否接近1

    //-- 对极约束限制
    vector<DMatch> better_matches;//再用对极约束限制过滤得到的特征点
    Mat d;
    vector<Point2d> keypoint1,keypoint2;
    for (DMatch m: good_matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K1,alpha1,ksi1);//像素平面ud,vd——>归一化平面Pd，就差一个K，所以pixel2cam里只用一个K不用畸变参数
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K2,alpha2,ksi2);
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

    Mat d1;//去掉过大异常值的匹配点后，其余点的对极误差。后面在此基础上优化
/*    for (int i = 0; i < good_matches.size(); i++)
    //for (int i = 0; i < 25; i++)//用/home/hl18/projects/fisheye_project下的1.png和2.png，对极约束限制为0.1，正好能检测到一小块集中位置的匹配点，用于检验深度计算是否正确(200)，但少量点算出的R和t也不准
    //for (int i =55; i < 65; i++)//用/home/hl18/projects/fisheye_project下的1.png和2.png，对极约束限制为0.3，正好能检测到一小块集中位置的匹配点，用于检验深度计算是否正确(514)
    {
        double d_exam=abs(d.at<double>(i,0));//对极误差绝对值
        if (d_exam <= epipolar_constraint_limit) {
            better_matches.push_back(good_matches[i]);
            cout<<"epipolar constraint ="<<d.at<double>(i,0)<<endl;//和上面求的一样，只是去掉了差很大的，这些可能是误匹配，去掉误匹配的得到better_match再用这个匹配去优化对极误差
            d1.push_back(d.at<double>(i,0));
        }
    }*/

    for (int i = 0; i < good_matches.size(); i++)
    {
        //if (i==6||i==10||i==16||i==17||i==18||i==19||i==20||i==21||i==22||i==23||i==24||i==27||i==28||i==29||i==30||i==31||i==32||i==33||i==35||i==36||i==39||i==41||i==42||i==45)
        //{
            double d_exam=abs(d.at<double>(i,0));//对极误差绝对值

            //if (d_exam <= epipolar_constraint_limit) {
            //if (d_exam <= epipolar_constraint_limit  && keypoint1.at(i).x>img_1.cols*0.5 && keypoint2.at(count).x<img_2.cols*0.5 && keypoint1.at(count).y>img_1.cols*0.2 && keypoint1.at(count).y<img_1.cols*0.8 && keypoint2.at(count).y>img_1.cols*0.2 && keypoint2.at(count).y<img_1.cols*0.8) {
            if ( d_exam <= epipolar_constraint_limit  && (int(keypoint1.at(i).x)>img_1.cols*0.4) ) {  //&& (int(keypoint1.at(i).y)<750)
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
    cv::waitKey(0);

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
        xp11.push_back(pixel2cam(points11[i], K1,alpha1,ksi1));
        xp22.push_back(pixel2cam(points22[i], K2,alpha2,ksi2));
    }
    essential_matrix2 = findFundamentalMat(xp11, xp22, CV_FM_8POINT);//或FM_RANSAC等方法
    //essential_matrix2 = findEssentialMat(points11, points22, CV_FM_8POINT);//虚拟相机
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
    //recoverPose(essential_matrix2, points11, points22, KK,  R3, t3);//虚拟相机
    cout << "R3 is " << endl << R3 << endl;
    cout << "t3 is " << endl << t3 << endl;
    R=R3;
    t=t3;

    //代入先前用更多匹配点算出的位姿数据
//    R=(Mat_<double>(3,3)<<-0.5257678124608629, 0.001501777755276956, -0.8506267995094515,
//    0.00266868448820201, 0.9999964323283758, 0.0001159897524669581,
//    0.8506239389431862, -0.002209070866677358, -0.5257699444649576);
//    t=(Mat_<double>(3,1)<<-0.4721633184634196,-0.04013369250561849,-0.8805970062539821);

    // 欧拉角 四元数
    euler_angle(R);

    Mat t_40=(Mat_<double>(3,1)<<t.at<double>(0,0)*40, t.at<double>(1,0)*40, t.at<double>(2,0)*40);//将求出的t缩放尺度
//    t=(Mat_<double>(3, 1) << 34.64,0,-20);//双目t假设定值不变，为40cm。
//    t=(Mat_<double>(3, 1) << -0.00632816,-0.00006507,-0.01463635);
    t=t_40;
    cout<<"set t="<<t<<endl;

    //-- 验证E=t^R*scale
    Mat t_x2, scale2;
    E_tR_scale(R,t,t_x2, essential_matrix2, scale2);//输出scale看是否接近1
    //cout<<"scale2 = "<<scale2<<endl;//有问题，总有一个数差异很大

    //验证p2' * F * p1：发现p2' * F * p1的结果都接近0
    //pFp(fundamental_matrix2,better_matches,points11,points22);//报错，但是已经在matlab解决

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
    triangulation_self(R, t, K1,K2,alpha1,alpha2,ksi1,ksi2,points11,points22,better_matches,keypoints_1,keypoints_2,img_1,img_2,pts_3d_eigen);


    //solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
//    VecVector3d pts_3d_eigen;
//    VecVector2d pts_2d_eigen;
//    VecVector3d x1;
//    VecVector3d x2;//归一化平面坐标
    VecVector3d x1;
    VecVector3d x2;//归一化平面坐标
//用去掉过大异常值的匹配点，求E，然后算对极误差
    Mat d2;
    for(DMatch m: better_matches) {
        Point2d pt11 = pixel2cam(keypoints_1[m.queryIdx].pt, K1,alpha1,ksi1);//像素平面ud,vd——>归一化平面Pd
        //Point2d pt11 = pixel2cam_pin(keypoints_1[m.queryIdx].pt, K1);//虚拟相机1 像素平面ud,vd——>归一化平面Pd
        Mat y11 = (Mat_<double>(3, 1) << pt11.x, pt11.y, 1);
        Point2d pt22 = pixel2cam(keypoints_2[m.trainIdx].pt, K2,alpha2,ksi2);
        //Point2d pt22 = pixel2cam_pin(keypoints_2[m.trainIdx].pt, K2);//虚拟相机2
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


//    for (size_t i = 0; i < pts_3d.size(); ++i) {
//        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
//        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
//    }

    //BA优化得到更好的位姿
    //t=(Mat_<double>(3,1)<<-0.4721633184634196,-0.04013369250561849,-0.8805970062539821);//固定t
    //t=(Mat_<double>(3,1)<<-0.00632816 -0.00006507 -0.01463635);//固定t
    //t=(Mat_<double>(3,1)<<39.33115291137872,3.316482824793844,-6.485472406432146);//固定t
    //t=(Mat_<double>(3,1)<<38.87935081855338,2.89009259037867,8.946700215548258);//固定t
//    t=(Mat_<double>(3,1)<<35.00200389531651,4.021284938569589,-18.93908632313295);//固定t
    //t=(Mat_<double>(3,1)<<40,0,0);//固定t
    //t=(Mat_<double>(3,1)<<34.641016,0,-20);//固定t

    cout << "calling bundle adjustment by g2o" << endl;
    //Sophus::SE3d pose_g2o;
    Sophus::SO3d R_ba;
    Sophus::Matrix3d t_ba;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(x1, x2, R,t, R_ba,t_ba);//BA_func_1，优化对极误差
    //bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);//BA_func_2，优化重投影误差：输入3d点和2d点和相机内参矩阵，输出pose为优化变量 使用BA方法
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;


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
    waitKey(0);
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

//Point2d pixel2cam_pin(const Point2d &p, const Mat &K) {
//    return Point2d
//            (
//                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
//                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
//            );
//}

Point2d pixel2cam(const Point2d &p, const Mat &K, double &alpha, double &ksi) {//得到归一化坐标
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

Point3d pixel2cam2(const Point2d &p, const Mat &K, double &alpha, double &ksi) {//得到未归一化的坐标，即(xp_x,xp_y,xp_z)，也即(mx,my,mz)
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
                          Mat &R, Mat &t, Mat &K1,Mat &K2,const double &alpha1,const double &alpha2,const  double &ksi1,const double &ksi2, Mat &essential_matrix, Mat &fundamental_matrix) {

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

        double u1=points1[i].x;
        double v1=points1[i].y;//https://www.coder.work/article/1229297#:~:text=%E6%9C%80%E4%BD%B3%E7%AD%94%E6%A1%88%20%E5%BD%93%E6%88%91%E9%9C%80%E8%A6%81%E4%BB%8E%E5%83%8F%E6%82%A8%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E7%BB%84%E4%B8%AD%E5%88%86%E5%88%AB%E6%8F%90%E5%8F%96%20X%20%E5%92%8C%20Y%20%E5%80%BC%E6%97%B6%EF%BC%8C%E6%88%91%E6%98%AF%E8%BF%99%E6%A0%B7%E5%81%9A%E7%9A%84%3A%20std%20%3A%3A,y%20%3D%20corners%20%5Bk%5D.y%3B%20%2F%2Fsecond%20value%20%2F%2Fstuff%20%7D
        double u2=points2[i].x;
        double v2=points2[i].y;
        double cx1=K1.at<double>(0,2);
        double cx2=K2.at<double>(0,2);
        double cy1=K1.at<double>(1,2);
        double cy2=K2.at<double>(1,2);
        double fx1=K1.at<double>(0,0);
        double fx2=K2.at<double>(0,0);
        double fy1=K1.at<double>(1,1);//https://blog.csdn.net/duiwangxiaomi/article/details/93075571
        double fy2=K2.at<double>(1,1);//https://blog.csdn.net/duiwangxiaomi/article/details/93075571
        double mx1=(u1-cx1)/fx1;
        double my1=(v1-cy1)/fy1;
        double mx2=(u2-cx2)/fx2;
        double my2=(v2-cy2)/fy2;
        double r1 =sqrt(mx1*mx1+my1*my1);
        double r2 =sqrt(mx2*mx2+my2*my2);
        double mz1=(1-alpha1*alpha1*r1*r1)/(alpha1*sqrt(1-(2*alpha1-1)*r1*r1)+1-alpha1);
        double mz2=(1-alpha2*alpha2*r2*r2)/(alpha2*sqrt(1-(2*alpha2-1)*r2*r2)+1-alpha2);
        double xp1_x=(mz1*ksi1+sqrt(mz1*mz1+(1-ksi1*ksi1)*r1*r1))/(mz1*mz1+r1*r1)*mx1;
        double xp1_y=(mz1*ksi1+sqrt(mz1*mz1+(1-ksi1*ksi1)*r1*r1))/(mz1*mz1+r1*r1)*my1;
        double xp1_z=(mz1*ksi1+sqrt(mz1*mz1+(1-ksi1*ksi1)*r1*r1))/(mz1*mz1+r1*r1)*mz1-ksi1;
        double xp2_x=(mz2*ksi2+sqrt(mz2*mz2+(1-ksi2*ksi2)*r2*r2))/(mz2*mz2+r2*r2)*mx2;
        double xp2_y=(mz2*ksi2+sqrt(mz2*mz2+(1-ksi2*ksi2)*r2*r2))/(mz2*mz2+r2*r2)*my2;
        double xp2_z=(mz2*ksi2+sqrt(mz2*mz2+(1-ksi2*ksi2)*r2*r2))/(mz2*mz2+r2*r2)*mz2-ksi2;
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

    cout << "better roll yaw pitch = " << roll <<", "<< yaw <<", "<< pitch << endl;//转化成角度制显示
    Eigen::Quaterniond qq=Eigen::Quaterniond(R_matrix);
    cout << "better quaternion(x,y,z,w) = " << qq.coeffs().transpose() << endl;//
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


void triangulation_self(Mat &R, Mat &t, Mat &K1,Mat &K2,double &alpha1,double &alpha2,double &ksi1,double &ksi2,vector<Point2f> &points11,vector<Point2f> &points22,vector<DMatch> &better_matches,vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,Mat &img_1,Mat &img_2,VecVector3d &pts_3d_eigen)
{
    //-- 三角化
    vector<Point3d> points;//存放3d点
    //triangulation(keypoints_1, keypoints_2, better_matches, R, t, K, points);
    vector<Point3f> xp1_3d;
    vector<Point3f> xp2_3d;//未归一化的xp
    //vector<double> disp;
    for (int i = 0; i < (int) better_matches.size(); i++)
    {
    xp1_3d.push_back(pixel2cam2(points11[i], K1,alpha1,ksi1));
    xp2_3d.push_back(pixel2cam2(points22[i], K2,alpha2,ksi2));
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
    cv::waitKey();

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







////////////////////////////////////////////////////////

//画深度图  区分视差图 深度图
/*
    StereoSGBM是OpenCV提供的用于立体匹配的类,可将两幅由处在同一水平线的不同摄像机拍摄的图像进行匹配,比较物体在两幅图像中的相对位置,计算求得其视差图.
    用这个函数不太行，还不知道为什么，先用上面分部求得的深度画图试试
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(img_1, img_2, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);//非平行双目应该不能用视差*/

//点云(先解决深度问题再考虑)
/*    vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;
    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < img_1.rows; v++)
        for (int u = 0; u < img_2.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            Eigen::Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;
            double x = pixel2cam((points11[i], K));

            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;
            point_plot[0] = points[i].x;
            point_plot[1] = y * depth;
            point_plot[2] = depth;
            pointcloud.push_back(point_plot);
        }

    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // 画出点云
    //showPointCloud(pointcloud);*/

/*    //点云(先解决深度问题再考虑)
//    vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;
//    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
// //    for (int v = 0; v < img_1.rows; v++)
// //        for (int u = 0; u < img_2.cols; u++) {
//    //if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;
//    for (int i = 0; i < better_matches.size(); i++) {
//        for (int v = 0; v < img_1.rows; v+=2)
//            for (int u = 0; u < img_1.cols; u+=2) {
//                Eigen::Vector4d point_plot(0, 0, 0, 200); // 前三维为xyz,第四维为颜色
//                //Eigen::Vector4d point_plot(0, 0, 0, img_1.at<uchar>(keypoints_1[better_matches[i].queryIdx].pt) / 255.0); // 前三维为xyz,第四维为颜色
//                //Eigen::Vector4d point_plot(0, 0, 0, img_1.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色
//
//                point_plot[0] = points[i].x;
//                point_plot[1] = points[i].y;
//                point_plot[2] = points[i].z;
//                pointcloud.push_back(point_plot);
//            }
//    }
//    //cv::imshow("disparity", disparity / 96.0);
//    cv::waitKey(0);
//    // 画出点云
//    showPointCloud(pointcloud);//画的是零散的点，看不清，不画点云，还是要改画视差图*/

//原三角测量
/*void triangulation(
        const vector<KeyPoint> &keypoint_1,
        const vector<KeyPoint> &keypoint_2,
        const std::vector<DMatch> &matches,
        const Mat &R, const Mat &t, const Mat &K,
        vector<Point3d> &points) {
    Mat T1 = (Mat_<float>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    vector<Point2f> pts_1, pts_2;
    for (DMatch m:matches) {
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat pts_4d;//齐次坐标
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);//DS model不能直接用
    //代入3x4的[R|T]位姿矩阵、2d坐标（归一化相机坐标），输出齐次坐标（共四个维度，因此需要将前三个维度除以第四个维度以得到非齐次坐标xyz）
    // 直接求的深度s，不是最优解

    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
        );
        points.push_back(p);//得到的是在第一个相机下的坐标？
    }
}*/

//void showPointCloud(在pangolin中画图，已写好，无需调整)
/*
void showPointCloud(const vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}*/
