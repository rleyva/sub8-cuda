// Generic includes
#include <boost/bind.hpp>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ROS includes
#include <image_geometry/stereo_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

// Transport includes
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// Sensor-Type includes
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/image_encodings.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// PCL includes
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

bool l_info_recieved, r_info_recieved = false;

class Sub8StereoHandler {
private:
    // TODO: Generalize image size for cv matrices
    
    /* OpenCV Structures */
    cv::Mat left, right;
    cv::Mat left_src, right_src;
    cv::Mat disp;
    cv::Mat pdisp;
    cv::Mat Q_; // Reprojection matrix
    
    cv::gpu::GpuMat d_disp;          // Disparity matrix
    cv::gpu::GpuMat d_left, d_right; // Left & Right GPU matricies
    cv::gpu::GpuMat gpu_pdisp;
    
    cv::gpu::StereoBM_GPU bm;
    cv::gpu::StereoBeliefPropagation bp;
    cv::gpu::StereoConstantSpaceBP csbp;
    
    /* Stereo-Matching options */
    int ndisp;
    int matching_method;
    
    /* PCL */
    // TODO: Add the option to bypass ROS and display on PCL viewer
    bool gen_viewer;
    bool pcl_init_request;
    int pc_reduction_factor;
    
    /* ROS Elements */
    image_geometry::StereoCameraModel stereo_cam_;
    sensor_msgs::PointCloud2 stereo_pc;
    sensor_msgs::CameraInfo lc_info, rc_info;
    
    void fetch_ros_cam_info(const sensor_msgs::CameraInfoConstPtr &mesg,
                            std::string cam);
    
    // Function prototypes
public:
    Sub8StereoHandler();
    void fetch_camera_parameters(ros::NodeHandle &nh);
    void generate_point_cloud(const sensor_msgs::ImageConstPtr &msg_l,
                              const sensor_msgs::ImageConstPtr &msg_r,
                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_ptr,
                              ros::Publisher &pc_pub);
};

Sub8StereoHandler::Sub8StereoHandler() {
    this->pc_reduction_factor = 1;
    this->Q_ = cv::Mat(4, 4, CV_32F);
    this->disp = cv::Mat(644, 482, CV_32F);
    this->pdisp = cv::Mat(644, 482, CV_32FC3);
    
    this->d_disp = cv::gpu::GpuMat(644, 482, CV_16S);
    this->gpu_pdisp = cv::gpu::GpuMat(644, 482, CV_32FC3);
    
    /* Stereo Matching paramteres */
    // Block-Matching
    this->matching_method = 0;
    
    this->bm.preset = 0;
    this->bm.ndisp = 80;
    this->bm.winSize = 45;
    
    // Belief-Propagation
    this->bp.ndisp = 80;
    
    // Constant Stereo Belief-Propagation
    this->csbp.ndisp = 80;
    
    this->gen_viewer = false;
    this->pcl_init_request = false;
    this->pc_reduction_factor = 1;
}

void Sub8StereoHandler::fetch_camera_parameters(ros::NodeHandle &nh) {
    // Subscibe to each camera info message
    ros::Subscriber lc_info_sub = nh.subscribe<sensor_msgs::CameraInfo>(
                                                                        "/stereo/left/camera_info", 1,
                                                                        boost::bind(&Sub8StereoHandler::fetch_ros_cam_info, this, _1, "left"));
    ros::Subscriber rc_info_sub = nh.subscribe<sensor_msgs::CameraInfo>(
                                                                        "stereo/right/camera_info", 1,
                                                                        boost::bind(&Sub8StereoHandler::fetch_ros_cam_info, this, _1, "right"));
    
    while (!(l_info_recieved && r_info_recieved)) {
        ros::spinOnce();                         // tomfoolery
        ROS_WARN("Camera Info is not recieved"); // Block while image information
        // for both cameras is not recieved
    }
    
    lc_info_sub.shutdown();
    rc_info_sub.shutdown();
    
    this->stereo_cam_.fromCameraInfo(this->lc_info, this->rc_info);
    cv::Matx44f Q = this->stereo_cam_.reprojectionMatrix();
    this->Q_ = (cv::Mat)Q;
    std::cout << "Reprojection Matrix: \n" << this->Q_ << std::endl;
}

void Sub8StereoHandler::fetch_ros_cam_info(
                                           const sensor_msgs::CameraInfoConstPtr &mesg, std::string cam) {
    if (cam == "left") {
        l_info_recieved = true;
        this->lc_info = *mesg;
        ROS_WARN("Recieved left message");
    } else if (cam == "right") {
        r_info_recieved = true;
        this->rc_info = *mesg;
        ROS_WARN("Recieved right message");
    } else {
        ROS_WARN("We fucked up");
    }
}

void Sub8StereoHandler::generate_point_cloud(
                                             const sensor_msgs::ImageConstPtr &msg_l,
                                             const sensor_msgs::ImageConstPtr &msg_r,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_ptr, ros::Publisher &pc_pub) {
    // NOTE: Calls to this function will block until the Synchronizer is able to
    // secure
    //       the corresponding messages.
    float px, py, pz;
    cv_bridge::CvImagePtr cv_ptr_l;
    cv_bridge::CvImagePtr cv_ptr_r;
    cv_bridge::CvImagePtr cv_ptr_bgr;
    
    try {
        cv_ptr_l = cv_bridge::toCvCopy(msg_l, sensor_msgs::image_encodings::MONO8);
        cv_ptr_r = cv_bridge::toCvCopy(msg_r, sensor_msgs::image_encodings::MONO8);
        
        // Used to reproject color onto pointcloud (We'll just use the left image)
        cv_ptr_bgr = cv_bridge::toCvCopy(msg_l, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::gpu::Stream pc_stream = cv::gpu::Stream();
    this->d_left.upload(cv_ptr_l->image);
    this->d_right.upload(cv_ptr_r->image);
    
    switch (this->matching_method) {
        case 0:
            if (this->d_left.channels() > 1 || this->d_right.channels() > 1) {
                // TODO: Add handle for these cases
                ROS_WARN("Block-matcher does not support color images");
            }
            this->bm(this->d_left, this->d_right, this->d_disp, pc_stream);
            break; // BM
        case 1:
            this->bp(this->d_left, this->d_right, this->d_disp, pc_stream);
            break; // BP
        case 2:
            this->csbp(this->d_left, this->d_right, this->d_disp, pc_stream);
            break; // CSBP
    }
    
    // cv::Mat h_disp, h_3d_img, h_point;
    
    cv::gpu::reprojectImageTo3D(this->d_disp, this->gpu_pdisp, this->Q_, 4,
                                pc_stream);
    this->gpu_pdisp.download(this->pdisp);
    
    for (int i = 0; i < this->pdisp.rows; i++) {
        // For future reference OpenCV is ROW-MAJOR
        for (int j = 0; j < this->pdisp.cols; j++) {
            cv::Point3f ptxyz = this->pdisp.at<cv::Point3f>(i, j);
            
            if (!isinf(ptxyz.x) && !isinf(ptxyz.y) && !isinf(ptxyz.z)) {
                // Reproject color points onto distances
                cv::Vec3b color_info = cv_ptr_bgr->image.at<cv::Vec3b>(i, j);
                pcl::PointXYZRGB pcl_pt;
                
                pcl_pt.x = ptxyz.x;
                pcl_pt.y = ptxyz.y;
                pcl_pt.z = ptxyz.z;
                
                uint32_t rgb = (static_cast<uint32_t>(color_info[2]) << 16 |
                                static_cast<uint32_t>(color_info[1]) << 8 |
                                static_cast<uint32_t>(color_info[0]));
                pcl_pt.rgb = *reinterpret_cast<float *>(&rgb);
                pcl_ptr->points.push_back(pcl_pt);
            }
        }
    }
    
    pcl_ptr->width = (int)pcl_ptr->points.size();
    pcl_ptr->height = 1;
    
    pcl::toROSMsg(*pcl_ptr, this->stereo_pc);
    this->stereo_pc.header.frame_id = "/stereo_front";
    pcl_ptr->clear();
    pc_pub.publish(this->stereo_pc);
}

int main(int argc, char **argv) {
    // TODO: Convert this to a Nodelet
    // TODO: Add launch file & remove hardcoded topics
    // TODO: Add camera service for its parameters (CameraInfoManager only seems
    // to set info)
    sensor_msgs::CameraInfo linfo, rinfo;
    Sub8StereoHandler stereo_handler;
    
    ros::init(argc, argv, "Sub8_TX1");
    ros::NodeHandle nh;
    
    ros::Publisher pc_pub =
    nh.advertise<sensor_msgs::PointCloud2>("/stereo/point_cloud", 1);
    stereo_handler.fetch_camera_parameters(nh);
    
    ROS_WARN("Initializing point cloud generation");
    
    message_filters::Subscriber<sensor_msgs::Image> l_image_sub(
                                                                nh, "/stereo/left/image_rect_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> r_image_sub(
                                                                nh, "/stereo/right/image_rect_color", 1);
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
    sensor_msgs::Image>
    stereo_sync;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_ptr(
                                                   new pcl::PointCloud<pcl::PointXYZRGB>);
    
    message_filters::Synchronizer<stereo_sync> sync(stereo_sync(3), l_image_sub,
                                                    r_image_sub);
    sync.registerCallback(boost::bind(&Sub8StereoHandler::generate_point_cloud,
                                      &stereo_handler, _1, _2, pcl_ptr, pc_pub));
    
    ros::spin();
    return 0;
}