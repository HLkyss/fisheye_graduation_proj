#include "BA_func.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/hyper_graph.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/base_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
//#include <sophus/se3.hpp>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
//using namespace cv;

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

/// vertex and edges used in g2o ba
// 曲线模型的顶点，待优化的变量，模板参数：优化变量维度和数据类型
/*class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {//VertexPose顶点，6维pose，顶点更新
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //virtual void setToOriginImpl(const Mat &R,const Mat &t) override {   //重置
    virtual void setToOriginImpl(const Mat &R,const Mat &t)  {   //重置
        //  从旋转矩阵进行构建SE(3)
        //_estimate = Sophus::SE3d();   //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
        Eigen::Matrix<double, 3, 3> RR;
        cv::eigen2cv(RR, R);
        Eigen::Matrix<double, 3, 1> tt;
        cv::eigen2cv(tt, t);
        Sophus::SE3d SE3_Rt(RR,tt);
        _estimate = SE3_Rt;//将优化前求出的R和t作为初值
        //_estimate = Sophus::SE3d(R,t);
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {     //更新
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    // 存盘和读盘：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};*/

class VertexPoseR : public g2o::BaseVertex<3, Sophus::SO3d> {//VertexPoseR顶点，3维R，顶点更新
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //virtual void setToOriginImpl(const Mat &R,const Mat &t) override {   //重置
    virtual void setToOriginImpl() override {   //重置    //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
/*        //  从旋转矩阵进行构建SE(3)
        //_estimate = Sophus::SE3d();
        Eigen::Matrix<double, 3, 3> RR;
        //cv::eigen2cv(RR, R);
        cv::cv2eigen(R, RR);
        Eigen::Matrix<double, 3, 1> tt;
        cv::cv2eigen(t, tt);
        Sophus::SO3d SO3_R(RR);
        _estimate = Sophus::SO3d(R,t);//将优化前求出的R和t作为初值*/
        _estimate = Sophus::SO3d();//将优化前求出的R和t作为初值
        //_estimate = Sophus::SE3d(R,t);
    }

    // left multiplication on SO3
    // right multiplication on Matrix
    virtual void oplusImpl(const double *update) override {     //更新
        Eigen::Matrix<double, 3, 1> update_eigen;
        update_eigen << update[0], update[1], update[2];
        _estimate = _estimate * Sophus::SO3d::exp(update_eigen);
        //_estimate = Sophus::SO3d::exp(update_eigen) * _estimate;
    }

    // 存盘和读盘：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};

/*class VertexPoset : public g2o::BaseVertex<3, Sophus::Matrix3d> {//VertexPoset顶点，3维t，顶点更新
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //virtual void setToOriginImpl(const Mat &R,const Mat &t) override {   //重置
    //virtual void setToOoverrideriginImpl() override {   //重置
    virtual void setToOriginImpl() override {   //重置
*//*        //  从旋转矩阵进行构建SE(3)
        //_estimate = Sophus::SE3d();   //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
        Eigen::Matrix<double, 3, 3> RR;
        cv::cv2eigen(R, RR);
        //Eigen::Matrix<double, 3, 1> tt;
        //Eigen::Vector3d tt;
        Sophus::Matrix3d tt;
        Mat t_x =   //t^
                (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                        -t.at<double>(1, 0), t.at<double>(0, 0), 0);
        //Eigen::MatrixXd tt(3, 1);//为什么不能用vector3d？程序未运行时，大小未知，所以用动态？https://blog.csdn.net/MaybeTnT/article/details/110868488
        cv::cv2eigen(t_x, tt);
        //Sophus::Matrix3d Matrix_t(tt);
        //_estimate << Matrix_t[0],Matrix_t[1],Matrix_t[2];//将优化前求出的R和t作为初值
        //_estimate=Sophus::Vector3d(tt);//将优化前求出的R和t作为初值
        _estimate=tt;//将优化前求出的R和t作为初值*//*
        _estimate=Sophus::Matrix3d();//将优化前求出的R和t作为初值
        //Matrix
        //_estimate = Sophus::SE3d(R,t);
    }

    // left multiplication on Matrix
    virtual void oplusImpl(const double *update) override {     //更新
        //Eigen::Matrix<double, 3, 1> update_eigen;
        //Eigen::Vector3d update_eigen;
        Sophus::Matrix3d update_eigen;
        update_eigen << update[0], update[1], update[2],update[3],update[4],update[5],update[6],update[7],update[8];
        //_estimate = Sophus::Matrix3d::exp(update_eigen) * _estimate;
        //_estimate += Eigen::Vector3d(update);
        _estimate += Sophus::Matrix3d(update_eigen);
        //_estimate += Eigen::MatrixXd(update_eigen);//为什么换成MatrixXd就好了？
    }

    // 存盘和读盘：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};*/

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
typedef Eigen::Matrix<double,1,1> Vector1d;//观测值为对极误差，为一个接近0的常数
//class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector3d, VertexPose> {//EdgeProjection边，边的误差计算
//class EdgeProjection : public g2o::BaseBinaryEdge<1, Vector1d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ> {//EdgeProjection边，边的误差计算
//class EdgeProjection : public g2o::BaseBinaryEdge<3, VecVector3d, VertexPoseR, VertexPoset> {//EdgeProjection边，边的误差计算
class EdgeProjection : public g2o::BaseUnaryEdge<1, Vector1d, VertexPoseR> {//EdgeProjection边，边的误差计算
    //EdgeProjection边，边的误差计算
    // 1:观测值(归一化坐标x1,x2)的维度（对极误差结果）
    // Vector1d：观测值类型，一个接近0的常数
    // VertexPoseR：第一个顶点类型（要优化）
    // VertexPoset：第二个顶点类型（要优化）
    // https://blog.csdn.net/QLeelq/article/details/115497273
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}
    EdgeProjection(const Eigen::Vector3d &x1,const Eigen::Vector3d &x2,Sophus::Matrix3d &t) : _x1(x1), _x2(x2),_t(t) {}
    //EdgeProjection(const Sophus::Vector3d &x1,const Sophus::Vector3d &x2) : _x1(x1), _x2(x2) {}
    //EdgeProjection(const g2o::VertexSBAPointXYZ &x1,const g2o::VertexSBAPointXYZ &x2) : _x1(x1), _x2(x2) {}

    virtual void computeError() override {  //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数

        const VertexPoseR *v1 = static_cast<VertexPoseR *> (_vertices[0]);//创建指针v

        Sophus::SO3d R_ = v1->estimate();//估计量，位姿
        //Eigen::Matrix<double,3,3> R_ = v1->estimate();//估计量，位姿
        //Sophus::Matrix3d R_ = v1->estimate();//估计量，位姿
//        Sophus::Matrix3d t_ = v2->estimate();//估计量，位姿
        //Sophus::Vector3d t = v2->estimate();//估计量，位姿
        //Eigen::MatrixXd t = v2->estimate();//估计量，位姿

//        const g2o::VertexSBAPointXYZ *v1 = static_cast<const g2o::VertexSBAPointXYZ*> (_vertices[0]);
//        const g2o::VertexSBAPointXYZ *v2 = static_cast<const g2o::VertexSBAPointXYZ*> (_vertices[1]);


//        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
//        pos_pixel /= pos_pixel[2];
//        _error = _measurement - pos_pixel.head<2>();// .head<2>表示向量里的前两个数 https://blog.csdn.net/hltt3838/article/details/105332576


//        Eigen::Matrix<double, 3, 3> RR;
//        cv::eigen2cv(RR, R);
//        Eigen::Matrix<double, 3, 1> tt;
//        cv::eigen2cv(tt, t);
//        Sophus::SO3d SO3_R(RR);              // Sophus::SO3d可以直接从旋转矩阵构造

        //Eigen::Matrix3d E_=t_ * R_;
        Sophus::Matrix3d E_=_t * R_.matrix();
        Vector1d epi=_x2.transpose() * E_ * _x1;
        //double epi=_x2.transpose() * t_ * R_ * _x1;

        //_error = _measurement - Sophus::SO3d::hat(_x2) * t * R * _x1;
        _error = (_measurement - epi)*10000;
    }

/*    virtual void linearizeOplus() override {    //计算雅可比矩阵
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
//        Eigen::Vector3d pos_cam = T * _pos3d;

//        double Z2 = Z * Z;
//        _jacobianOplusXi
//                << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
//                0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }*/

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

private:
    Eigen::Vector3d _pos3d;
    Eigen::Vector3d _x1;
    Eigen::Vector3d _x2;
    //Eigen::Matrix3d _K;
    Sophus::Matrix3d _t;
};

void bundleAdjustmentG2O(
        const VecVector3d &x1,
        const VecVector3d &x2,
        const Mat &R, const Mat &t,
        Sophus::SO3d &R_ba,Sophus::Matrix3d &t_ba) {

    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> BlockSolverType;  //每个优化变量的维度为1，误差项维度为1   pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // vertex   往图中增加顶点
    //VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    VertexPoseR *vertex_pose_R = new VertexPoseR();
    vertex_pose_R->setId(0);

    Eigen::Matrix<double, 3, 3> RR;
    //代入含有人为添加误差的R作为初值测试效果
    Mat R_error_1=(Mat_<double>(3, 3) << 0.9843458450955395, -0.001284868080891974, 0.176243032081169,
    0.0002934824995656224, 0.9999839895893146, 0.005651055924366041,
    -0.1762474712192257, -0.005510869173966465, 0.9843304624005982);//y10
    Mat R_error_2=(Mat_<double>(3, 3) << 0.8313132326332605, 0.03917026952127017, -0.5544222210864836,
    -0.04236351680666305, 0.999077286193542, 0.007064605857326768,
    0.5541873705639726, 0.01761437474845043, 0.8322055588072045);//y-40
    //cv::cv2eigen(R_error_1, RR);
    cv::cv2eigen(R, RR);
    Sophus::SO3d SO3_R(RR);

    vertex_pose_R->setEstimate(SO3_R);
    optimizer.addVertex(vertex_pose_R);

    /////////////////////////////////////////////////////////////

    //  从旋转矩阵进行构建SE(3)
    //_estimate = Sophus::SE3d();   //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
    //Eigen::Matrix<double, 3, 1> tt;
    //Eigen::Vector3d tt;
    //Eigen::Matrix<double, 3, 1> tt;
    Sophus::Matrix3d tt;//3x3反对称矩阵
    Mat t_x =   //t^
            (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    //Eigen::MatrixXd tt(3, 1);//为什么不能用vector3d？程序未运行时，大小未知，所以用动态？https://blog.csdn.net/MaybeTnT/article/details/110868488
    cv::cv2eigen(t_x, tt);
    //Sophus::Matrix3d Matrix_t(tt);
    //_estimate << Matrix_t[0],Matrix_t[1],Matrix_t[2];//将优化前求出的R和t作为初值
    //_estimate=Sophus::Vector3d(tt);//将优化前求出的R和t作为初值

/////////////////////////////////////////////////////////////////////////////

    // edges    往图中增加边
    int index = 1;
    for (size_t i = 0; i < x1.size(); ++i) {
//        auto p2d = points_2d[i];
//        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(x1[i],x2[i],tt);
        //edge->setId(index);
        edge->setVertex(0, vertex_pose_R);    //设置连接的顶点
        //edge->setVertex(1, vertex_pose_t);    //设置连接的顶点

        edge->setMeasurement(Vector1d(0));    //观测数值
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity());//信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(50);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "R estimated by g2o =\n" << vertex_pose_R->estimate().matrix() << endl;
    //cout << "t estimated by g2o =\n" << vertex_pose_t->estimate().matrix() << endl;
    R_ba = vertex_pose_R->estimate();
    //t_ba = vertex_pose_t->estimate();

    Eigen::Matrix<double,3,3> R_ba_=R_ba.matrix();
//    Eigen::Vector3d euler_angles_ba = R_ba_.eulerAngles(0, 2, 1); // 210:ZYX顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    Eigen::Vector3d euler_angles_ba = R_ba_.eulerAngles(1, 0, 2); // 210:ZYX顺序，即ryp顺序(这个函数转换出来的欧拉角可能不稳定:https://blog.csdn.net/qingtian11112/article/details/105246247)
    double yaw_=euler_angles_ba[0]*180/M_PI;
    double pitch_=euler_angles_ba[1]*180/M_PI;
    double roll_=euler_angles_ba[2]*180/M_PI;
    if(roll_<-90)//-170
    {
        roll_=180+roll_;
    }
    if(roll_>90)//170
    {
        roll_=180-roll_;
    }
    if(pitch_>90)//150
    {
        pitch_=180-pitch_;
    }
    if(pitch_<-90)//-150
    {
        pitch_=180+pitch_;
    }
    if(yaw_>0 && yaw_<90)//60
    {
        yaw_=180-yaw_;
    }
    if(yaw_>-90&&yaw_<0)//-60
    {
        yaw_=180+yaw_;
    }//注：通过这种判断处理得到的欧拉角，得到的正负号可能会有问题，再用其复成旋转矩阵，可能会出问题

    cout << "BA yaw pitch roll = " << yaw_ <<", "<< pitch_ << ", "<< roll_ << endl;//转化成角度制显示
    Eigen::Quaterniond qq=Eigen::Quaterniond(R_ba_);
    cout << "BA quaternion(x,y,z,w) = " << qq.coeffs().transpose() << endl;
    //cout<<"better t="<<t<<endl;

    cout<<"from euler to R ="<<endl;//验证和R estimated by g2o结果是否一致，验证这种处理方法对各个欧拉角的理解对不对
    Eigen::Matrix3d rotation_matrix_4;
    rotation_matrix_4 = Eigen::AngleAxisd(euler_angles_ba[0], Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(euler_angles_ba[1], Eigen::Vector3d::UnitX()) *
                        Eigen::AngleAxisd(euler_angles_ba[2], Eigen::Vector3d::UnitZ());
    cout << rotation_matrix_4 << endl;
}


//chi2是最小二乘问题中该边的代价，omega是该边的信息矩阵

/*

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 重置
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    // 更新
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update);
    }

    // 存盘和读盘：留空
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // 计算曲线模型误差
    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // 计算雅可比矩阵
    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

public:
    double _x;  // x 值， y 值为 _measurement
};

void BA()
{
    cout<<"hello"<<endl;
}*/
