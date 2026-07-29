#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "dre/msg/drl_estimate.hpp"
#include "dre/msg/local_map_info.hpp"

#include <ba/map/voxel_map.hpp>
#include <ba/scans/local_map_scan.hpp>
#include <ba/utils/ba_config.hpp>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/so2/Rotation.hpp>


class LocNode : public rclcpp::Node {
public:
    LocNode() : Node("loc_node") {
        // Load config
        load_config();

        estimate_pub_ = create_publisher<dre::msg::DRLEstimate>("/drl_estimate", 10);

        cv::setNumThreads(num_threads_);

        voxel_map_.load_from_file(map_path_);
        RCLCPP_INFO(get_logger(), "Voxel map loaded: %zu voxels", voxel_map_.size());

        // Publish the map path once on boot (transient_local so late-joining
        // subscribers, e.g. map_viz_node, still receive it).
        rclcpp::QoS map_path_qos(1);
        map_path_qos.transient_local();
        map_path_pub_ = create_publisher<std_msgs::msg::String>("/map_path", map_path_qos);
        std_msgs::msg::String map_path_msg;
        map_path_msg.data = map_path_;
        map_path_pub_->publish(map_path_msg);

        // Subscribe to initial pose (transient_local to get latched messages)
        rclcpp::QoS qos(10);
        qos.transient_local();
        initial_pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", qos,
            std::bind(&LocNode::initial_pose_callback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Waiting for initial pose on /initialpose...");

        image_sub_.subscribe(this, "/dro_local_map_image");
        info_sub_.subscribe(this, "/dro_local_map_info");

        sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), image_sub_, info_sub_);
        sync_->registerCallback(std::bind(&LocNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));
    }

private:
    void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        if (!has_initial_pose_) {
            initial_pose_x_ = msg->pose.pose.position.x;
            initial_pose_y_ = msg->pose.pose.position.y;

            // Extract yaw from quaternion
            auto q = msg->pose.pose.orientation;
            initial_pose_yaw_ = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));

            RCLCPP_INFO(get_logger(), "Received initial pose: x=%.2f, y=%.2f, yaw=%.2f",
                        initial_pose_x_, initial_pose_y_, initial_pose_yaw_);
            has_initial_pose_ = true;
        }
    }

    void load_config() {
        // Config path comes from a ROS parameter (set by the launch file via
        // FindPackageShare), same as mapping_node/pogo_node — not derived from
        // __FILE__, which would bake the build machine's source path into the
        // compiled binary and break on any other machine or path.
        this->declare_parameter<std::string>("config_file", "");
        std::string full_config_path = this->get_parameter("config_file").as_string();
        if (full_config_path.empty()) {
            throw std::runtime_error("Config file path must be provided as a parameter 'config_file'.");
        }

        YAML::Node config = YAML::LoadFile(full_config_path);

        map_path_ = config["map_path"].as<std::string>();
        voxel_map_res_ = config["voxel_map_resolution"].as<double>();
        num_threads_ = config["num_threads"].as<int>();

        max_dist_ = config["frame_processing"]["max_dist"].as<double>();
        gauss_blur_sigma_ = config["frame_processing"]["gauss_blur_sigma"].as<double>();
        gauss_blur_ksize_ = config["frame_processing"]["gauss_blur_ksize"].as<int>();

        max_iter_ = config["optimization"]["max_iterations"].as<int>();
        conv_tol_ = config["optimization"]["convergence_tol"].as<double>();
        alpha_ = config["optimization"]["alpha"].as<double>();
        meas_std_ = config["optimization"]["meas_std"].as<double>();
        range_factor_ = config["optimization"]["range_factor"].as<double>();
    }

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, dre::msg::LocalMapInfo>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    void sync_callback(const sensor_msgs::msg::Image::ConstPtr& img_msg,
                       const dre::msg::LocalMapInfo::ConstPtr& info_msg) {
        if (!has_initial_pose_) {
            return;  // Skip processing until initial pose is received
        }

        auto t_total_start = std::chrono::high_resolution_clock::now();

        // Image conversion and blur
        cv::Mat cv_img(img_msg->height, img_msg->width, CV_8U, (void*)img_msg->data.data());
        cv::Mat cv_float;
        cv_img.convertTo(cv_float, CV_32F, 1.0 / 255.0);

        int blur_factor = (frame_count_ < 80) ? 3 : 1;
        int sigma = static_cast<int>(gauss_blur_sigma_ * blur_factor);
        int ksize = gauss_blur_ksize_ * blur_factor;
        cv::GaussianBlur(cv_float, cv_float, cv::Size(ksize, ksize), sigma);

        Eigen::MatrixXf local_map;
        cv::cv2eigen(cv_float, local_map);

        // Scan setup
        ba::OptimizationOptions opts;
        opts.meas_std = meas_std_;
        opts.range_factor = range_factor_;
        opts.cumul_thresh = 1.0;
        opts.zero_thresh = 1.0;

        lgmath::so2::Rotation C(info_msg->theta);
        Eigen::Vector2d r(info_msg->x, info_msg->y);
        lgmath::se2::Transformation curr_dro_pose(C.matrix(), -C.matrix().transpose() * r);

        lgmath::se2::Transformation init_pose(Eigen::Vector3d(0.0, 0.0, 0.0));
        if (has_prev_loc_ && has_prev_dro_) {
            lgmath::se2::Transformation dro_rel_pose = prev_dro_pose_.inverse() * curr_dro_pose;
            init_pose = prev_loc_pose_ * dro_rel_pose;
        } else if (!first_frame_) {
            init_pose = curr_dro_pose;
        } else {
            // First frame: use the initial pose from the pose selector
            lgmath::so2::Rotation C(initial_pose_yaw_);
            Eigen::Vector2d r(initial_pose_x_, initial_pose_y_);
            init_pose = lgmath::se2::Transformation(C.matrix(), -C.matrix().transpose() * r);
            first_frame_ = false;
        }

        ba::LocalMapScan scan(0, 0, voxel_map_res_, opts, init_pose.toSE3(), init_pose.toSE3(), local_map);

        // Optimization loop
        double cost = 0.0;
        int final_iter = 0;
        int total_voxels_used = 0;

        // Fetch voxels once (pose changes are small, so voxel set remains valid)
        auto voxel_keys = voxel_map_.get_voxels_in_range(scan.pose2d(), max_dist_);

        for (int iter = 0; iter < max_iter_; ++iter) {
            final_iter = iter;
            Eigen::Matrix3d lhs = Eigen::Matrix3d::Zero();
            Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
            cost = 0.0;
            int num_voxels_used = 0;

            auto t_interp = std::chrono::high_resolution_clock::now();

            // Parallel voxel loop with per-thread accumulation
            std::vector<Eigen::Matrix3d> lhs_threads(num_threads_, Eigen::Matrix3d::Zero());
            std::vector<Eigen::Vector3d> rhs_threads(num_threads_, Eigen::Vector3d::Zero());
            std::vector<double> cost_threads(num_threads_, 0.0);
            std::vector<int> count_threads(num_threads_, 0);

            #pragma omp parallel for num_threads(num_threads_)
            for (size_t i = 0; i < voxel_keys.size(); ++i) {
                int tid = omp_get_thread_num();
                const auto& idx = voxel_keys[i];
                double voxel_x = static_cast<double>(idx.first) * voxel_map_.res();
                double voxel_y = static_cast<double>(idx.second) * voxel_map_.res();
                double vox_intensity = voxel_map_.at(idx);

                std::optional<ba::Scan::Measurement> interp_meas = scan.interpolate(voxel_x, voxel_y);
                if (!interp_meas.has_value()) {
                    continue;
                }

                double I_meas = interp_meas->intensity;
                double meas_cov = interp_meas->covariance;
                double err_weight = 1.0 / meas_cov;

                double err = (vox_intensity - I_meas);
                double err_weight_sqrt = std::sqrt(err_weight);

                Eigen::Matrix<double, 1, 3> d_beta_d_T = - interp_meas->jacobian * err_weight_sqrt;
                err *= err_weight_sqrt;

                lhs_threads[tid] += d_beta_d_T.transpose() * d_beta_d_T;
                rhs_threads[tid] += d_beta_d_T.transpose() * err;
                cost_threads[tid] += 0.5 * std::pow(err, 2);
                count_threads[tid]++;
            }

            // Merge thread results
            for (int t = 0; t < num_threads_; ++t) {
                lhs += lhs_threads[t];
                rhs += rhs_threads[t];
                cost += cost_threads[t];
                num_voxels_used += count_threads[t];
            }
            total_voxels_used += num_voxels_used;

            if (num_voxels_used == 0 || lhs.determinant() == 0.0) {
                break;
            }

            Eigen::Vector3d delta = - alpha_ * lhs.inverse() * rhs;
            scan.update_pose(delta);

            if (iter != 0 && delta.norm() < conv_tol_) break;
        }

        double t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();

        if (frame_count_ == 0) {
            stats_start_time_ = std::chrono::high_resolution_clock::now();
        }
        ++frame_count_;
        sum_runtime_ms_ += t_total_ms;

        if (frame_count_ % 50 == 0) {
            double elapsed_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stats_start_time_).count();
            double avg_fps = elapsed_s > 0.0 ? static_cast<double>(frame_count_) / elapsed_s : 0.0;
            double avg_runtime_ms = sum_runtime_ms_ / static_cast<double>(frame_count_);
            RCLCPP_INFO(get_logger(), "Frames processed: %llu | avg FPS: %.2f | avg runtime/frame: %.1f ms",
                        static_cast<unsigned long long>(frame_count_), avg_fps, avg_runtime_ms);
        }

        lgmath::se2::Transformation final_pose = scan.pose2d();

        // Store current localization and odometry for next iteration
        prev_loc_pose_ = final_pose;
        prev_dro_pose_ = curr_dro_pose;
        has_prev_loc_ = true;
        has_prev_dro_ = true;

        auto estimate_msg = std::make_shared<dre::msg::DRLEstimate>();
        estimate_msg->header = img_msg->header;
        estimate_msg->x = final_pose.r_ab_inb()(0);
        estimate_msg->y = final_pose.r_ab_inb()(1);
        estimate_msg->theta = final_pose.vec()(2);
        estimate_msg->iterations = final_iter + 1;
        estimate_msg->cost = cost;
        estimate_pub_->publish(*estimate_msg);
    }

    rclcpp::Publisher<dre::msg::DRLEstimate>::SharedPtr estimate_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr map_path_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<dre::msg::LocalMapInfo> info_sub_;
    std::shared_ptr<Synchronizer> sync_;
    ba::VoxelMap voxel_map_{1.0};

    bool has_initial_pose_ = false;
    double initial_pose_x_ = 0.0;
    double initial_pose_y_ = 0.0;
    double initial_pose_yaw_ = 0.0;
    bool first_frame_ = true;

    lgmath::se2::Transformation prev_loc_pose_{Eigen::Vector3d(0.0, 0.0, 0.0)};
    lgmath::se2::Transformation prev_dro_pose_{Eigen::Vector3d(0.0, 0.0, 0.0)};
    bool has_prev_loc_ = false;
    bool has_prev_dro_ = false;

    // Frame processing stats
    uint64_t frame_count_ = 0;
    double sum_runtime_ms_ = 0.0;
    std::chrono::high_resolution_clock::time_point stats_start_time_;

    // Config parameters
    std::string map_path_;
    double voxel_map_res_;
    int num_threads_;
    double max_dist_;
    double gauss_blur_sigma_;
    int gauss_blur_ksize_;
    int max_iter_;
    double conv_tol_;
    double alpha_;
    double meas_std_;
    double range_factor_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocNode>());
    rclcpp::shutdown();
    return 0;
}
