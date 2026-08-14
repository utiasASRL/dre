#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>

// Replace these with the actual generated headers for your custom messages
#include "dre/msg/radar_info.hpp"
#include "dre/msg/local_map_info.hpp"
#include "navtech_msgs/msg/radar_b_scan_msg.hpp"
#include "dro/dro_wrapper.hpp"

#include <pybind11/embed.h> // Required for embedding Python in a C++ main binary

#include <deque>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <optional>


namespace py = pybind11;

// ==========================================
// ROS 2 NODE
// ==========================================

class DroNode : public rclcpp::Node {
public:
    using ImuData = DroWrapper::ImuData;
    using RadarData = DroWrapper::RadarData;

    DroNode() : Node("dro_node"), initialized_(false), first_(true), 
                imu_wait_timeout_sec_(0.2), frame_count_(0), sum_runtime_(0.0) {
        
        RCLCPP_INFO(this->get_logger(), "DroNode has been started.");

        // Subscriptions
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/w200_0066/sensors/imu_0/data", 1000,
            std::bind(&DroNode::imuCallback, this, std::placeholders::_1));

        radar_subscription_ = this->create_subscription<navtech_msgs::msg::RadarBScanMsg>(
            "/radar_data/b_scan_msg", 10,
            std::bind(&DroNode::radarCombinedCallback, this, std::placeholders::_1));

        // Publishers
        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("dro_odometry", 10);
        local_map_odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("dro_local_map_odometry", 10);

        local_map_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("dro_local_map_image", 10);
        cumulated_returns_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("dro_cumulated_returns_image", 10);
        local_map_info_publisher_ = this->create_publisher<dre::msg::LocalMapInfo>("dro_local_map_info", 10);

        // TF Broadcasters
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

        // Publish static initial transform
        geometry_msgs::msg::TransformStamped start_transform;
        start_transform.header.stamp.sec = 0;
        start_transform.header.stamp.nanosec = 0;
        start_transform.header.frame_id = "odom";
        start_transform.child_frame_id = "radar";
        start_transform.transform.rotation.w = 1.0;
        static_tf_broadcaster_->sendTransform(start_transform);

        // Parameters
        this->declare_parameter<std::string>("output_path", "output");
        output_path_ = this->get_parameter("output_path").as_string();


        // Initialize Dro
        dro_opts_ = loadDroOpts();
        dro_ = std::make_unique<DroWrapper>(dro_opts_);
        RCLCPP_INFO(this->get_logger(), "DRO ready");
    }

private:

    /**
     * @brief Reads ROS 2 parameters from a node and converts them into a pybind11::dict
     *        matching the structure and default values of Python's kDefaultDroOpts.
     * @param node Shared pointer to the rclcpp::Node instance.
     * @return py::dict Nested dictionary ready for pybind11 / Python.
     */
    py::dict loadDroOpts() {
        py::dict opts;

        // -------------------------------------------------------------------------
        // 1. estimation
        // -------------------------------------------------------------------------
        py::dict estimation;
        estimation["use_gyro"] = declare_parameter<bool>("estimation.use_gyro", true);
        estimation["estimate_gyro_bias"] = declare_parameter<bool>("estimation.estimate_gyro_bias", false);
        estimation["estimate_vy_bias"] = declare_parameter<bool>("estimation.estimate_vy_bias", false);
        estimation["vy_bias_prior"] = declare_parameter<double>("estimation.vy_bias_prior", 0.0);
        estimation["max_acceleration"] = declare_parameter<double>("estimation.max_acceleration", 10.0);
        estimation["min_time_bias_init"] = declare_parameter<double>("estimation.min_time_bias_init", 1.0);
        estimation["gyro_bias_alpha"] = declare_parameter<double>("estimation.gyro_bias_alpha", 0.01);

        // Default T_axle_radar: 4x4 Identity matrix flattened to 16 elements
        const std::vector<double> default_t_axle = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        auto t_axle_vec = declare_parameter<std::vector<double>>("estimation.T_axle_radar", default_t_axle);

        // Convert vector to 4x4 NumPy ndarray
        if (t_axle_vec.size() == 16) {
            estimation["T_axle_radar"] = py::array_t<double>({4, 4}, t_axle_vec.data());
        } else {
            RCLCPP_WARN(get_logger(), "'estimation.T_axle_radar' must have 16 elements. Falling back to 4x4 identity matrix.");
            estimation["T_axle_radar"] = py::array_t<double>({4, 4}, default_t_axle.data());
        }
        opts["estimation"] = estimation;

        // -------------------------------------------------------------------------
        // 2. gp
        // -------------------------------------------------------------------------
        py::dict gp;
        gp["lengthscale_az"] = declare_parameter<double>("gp.lengthscale_az", 2.0);
        gp["lengthscale_range"] = declare_parameter<double>("gp.lengthscale_range", 4.0);
        gp["sz"] = declare_parameter<double>("gp.sz", 0.6);
        opts["gp"] = gp;

        // -------------------------------------------------------------------------
        // 3. radar
        // -------------------------------------------------------------------------
        py::dict radar;
        radar["del_f"] = declare_parameter<double>("radar.del_f", 893.0e6);
        radar["ft"] = declare_parameter<double>("radar.ft", 76.04e9);
        radar["meas_freq"] = declare_parameter<double>("radar.meas_freq", 1600.0);
        radar["beta_corr_fact"] = declare_parameter<double>("radar.beta_corr_fact", 0.944);
        radar["range_offset"] = declare_parameter<double>("radar.range_offset", -0.31);

        // Optional parameters defaulting to None in Python
        int64_t nb_az = declare_parameter<int64_t>("radar.nb_azimuths", -1);
        radar["nb_azimuths"] = (nb_az > 0) ? py::cast(nb_az) : py::none();

        double resolution = declare_parameter<double>("radar.resolution", -1.0);
        radar["resolution"] = (resolution > 0.0) ? py::cast(resolution) : py::none();

        radar["doppler_enabled"] = declare_parameter<bool>("radar.doppler_enabled", false);
        opts["radar"] = radar;

        // -------------------------------------------------------------------------
        // 4. direct
        // -------------------------------------------------------------------------
        py::dict direct;
        direct["min_range"] = declare_parameter<double>("direct.min_range", 4.0);
        direct["max_range"] = declare_parameter<double>("direct.max_range", 70.0);
        direct["local_map_res"] = declare_parameter<double>("direct.local_map_res", 0.1);
        direct["max_local_map_range"] = declare_parameter<double>("direct.max_local_map_range", 120.0);
        direct["local_map_update_alpha"] = declare_parameter<double>("direct.local_map_update_alpha", 0.1);
        opts["direct"] = direct;

        // -------------------------------------------------------------------------
        // 5. doppler
        // -------------------------------------------------------------------------
        py::dict doppler;
        doppler["min_range"] = declare_parameter<double>("doppler.min_range", 4.0);
        doppler["max_range"] = declare_parameter<double>("doppler.max_range", 200.0);
        opts["doppler"] = doppler;

        // -------------------------------------------------------------------------
        // 6. solver
        // -------------------------------------------------------------------------
        py::dict solver;
        solver["nb_iter"] = declare_parameter<int64_t>("solver.nb_iter", 250);
        solver["cost_tol"] = declare_parameter<double>("solver.cost_tol", 1e-6);
        solver["step_tol"] = declare_parameter<double>("solver.step_tol", 1e-5);
        opts["solver"] = solver;

        py::dict log;
        log["save_local_maps"] = declare_parameter<bool>("log.save_local_maps", false);
        log["save_cumulative_image"] = declare_parameter<bool>("log.save_cumulative_image", false);
        opts["log"] = log;

        return opts;
    }


    const int64_t NANOSECOND_EPOCH_THRESHOLD = 100000000000000000LL; // 1e17

    void timestampsToMicroseconds(std::vector<uint64_t>& timestamps) {
        if (!timestamps.empty() && timestamps[0] > NANOSECOND_EPOCH_THRESHOLD) {
            for (auto& t : timestamps) {
                t /= 1000;
            }
        }
    }

    void initialize(const std::string& sequence_id) {
        initialized_ = true;
        // Mock init logic
    }

    void radarCallback(const sensor_msgs::msg::Image& image_msg, const dre::msg::RadarInfo& radar_info_msg) {
        if (!initialized_) {
            initialize(radar_info_msg.sequence_id);
        }

        // Decode Image
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat polar_image;
        cv_ptr->image.convertTo(polar_image, CV_32F, 1.0 / 255.0);

        std::vector<uint64_t> timestamps = radar_info_msg.timestamps;
        timestampsToMicroseconds(timestamps);

        int64_t msg_time_us = static_cast<int64_t>(image_msg.header.stamp.sec) * 1000000LL + 
                              static_cast<int64_t>(image_msg.header.stamp.nanosec) / 1000LL;

        RadarData data;
        data.polar = polar_image;
        data.azimuths = radar_info_msg.azimuth;
        data.timestamps = timestamps;
        data.resolution = radar_info_msg.resolution;
        data.chirps = radar_info_msg.chirps;
        data.timestamp = msg_time_us;

        radar_data_buffer_.push_back(data);
        odometryStepIfReady();
    }

    void radarCombinedCallback(const navtech_msgs::msg::RadarBScanMsg::SharedPtr b_scan_msg) {
        sensor_msgs::msg::Image image_msg = b_scan_msg->b_scan_img;
        dre::msg::RadarInfo r_info;
        
        for (const auto& v : b_scan_msg->encoder_values) {
            r_info.azimuth.push_back(2.0f * M_PI * static_cast<float>(v) / 16000.0f);
        }
        r_info.timestamps = b_scan_msg->timestamps;
        r_info.resolution = py::cast<float>(dro_opts_["radar"]["resolution"]);
        r_info.chirps.assign(b_scan_msg->timestamps.size(), 0);

        radarCallback(image_msg, r_info);
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        int64_t time_us = static_cast<int64_t>(msg->header.stamp.sec) * 1000000LL + 
                          static_cast<int64_t>(msg->header.stamp.nanosec) / 1000LL;

        if (last_imu_time_ && time_us <= *last_imu_time_) {
            RCLCPP_WARN(this->get_logger(), "Received out-of-order IMU message. Current time: %ld, Last time: %ld", time_us, *last_imu_time_);
            return;
        }
        last_imu_time_ = time_us;

        ImuData imu_data;
        imu_data.timestamp = time_us;
        imu_data.angular_velocity = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        imu_data.linear_acceleration = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

        imu_data_buffer_.push_back(imu_data);
        odometryStepIfReady();
    }

    double getCurrentTimeSec() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void odometryStepIfReady() {
        if (radar_data_buffer_.size() > 10) {
            radar_data_buffer_.pop_front();
            RCLCPP_WARN(this->get_logger(), "Radar buffer > 10, likely a problem in with IMU data, dropping oldest radar.");
        }

        if (radar_data_buffer_.empty() || imu_data_buffer_.empty()) {
            return;
        }

        int64_t first_radar_time = radar_data_buffer_.front().timestamps.front();
        if (first_ && imu_data_buffer_.front().timestamp > first_radar_time) {
            imu_data_buffer_.front().timestamp = first_radar_time - 1000;
            first_ = false;
        }

        int64_t last_radar_time = radar_data_buffer_.front().timestamps.back() + 2000;
        bool imu_timed_out = false;

        if (imu_data_buffer_.front().timestamp > first_radar_time || imu_data_buffer_.back().timestamp < last_radar_time) {
            if (!imu_wait_start_time_) {
                imu_wait_start_time_ = getCurrentTimeSec();
            }
            if (getCurrentTimeSec() - *imu_wait_start_time_ < imu_wait_timeout_sec_) {
                return;
            }
            imu_timed_out = true;
        }

        imu_wait_start_time_ = std::nullopt;

        // Search boundaries
        auto start_it = std::lower_bound(imu_data_buffer_.begin(), imu_data_buffer_.end(), first_radar_time,
                                         [](const ImuData& a, int64_t b) { return a.timestamp < b; });
        int start_idx = std::distance(imu_data_buffer_.begin(), start_it);
        start_idx = std::max(0, start_idx - 1);

        auto end_it = std::upper_bound(imu_data_buffer_.begin(), imu_data_buffer_.end(), last_radar_time,
                                       [](int64_t a, const ImuData& b) { return a < b.timestamp; });
        int end_idx = std::distance(imu_data_buffer_.begin(), end_it);
        end_idx = std::min((int)imu_data_buffer_.size(), end_idx + 1);

        std::vector<ImuData> relevant_imus(imu_data_buffer_.begin() + start_idx, imu_data_buffer_.begin() + end_idx);

        if (imu_timed_out) {
            if (relevant_imus.front().timestamp > first_radar_time) {
                ImuData synthetic_first = relevant_imus.front();
                synthetic_first.timestamp = first_radar_time - 1000;
                relevant_imus.insert(relevant_imus.begin(), synthetic_first);
            }
            if (relevant_imus.back().timestamp < last_radar_time) {
                ImuData synthetic_last = relevant_imus.back();
                synthetic_last.timestamp = last_radar_time + 1000;
                relevant_imus.push_back(synthetic_last);
            }
        }

        cv::Mat local_map;

        double t1 = getCurrentTimeSec();
        dro_->odometryStep(radar_data_buffer_.front(), relevant_imus, local_map);
        double t2 = getCurrentTimeSec();

        if (frame_count_ == 0) {
            stats_start_time_ = t1;
        }
        frame_count_++;
        sum_runtime_ += (t2 - t1);

        if (frame_count_ % 50 == 0) {
            double elapsed = t2 - stats_start_time_;
            double avg_fps = (elapsed > 0) ? (frame_count_ / elapsed) : 0.0;
            double avg_runtime = sum_runtime_ / frame_count_;
            RCLCPP_INFO(this->get_logger(), "Frames processed: %d | avg FPS: %.2f | avg runtime/frame: %.1f ms",
                        frame_count_, avg_fps, avg_runtime * 1000.0);
        }


        Eigen::Matrix4d current_odometry = dro_->getPose(radar_data_buffer_.front().timestamp);
        publishOdometry(current_odometry, radar_data_buffer_.front().timestamp);

        publishLocalMap(local_map, {0, 0, 0}, radar_data_buffer_.front().timestamp);


        int64_t temp_last_time = radar_data_buffer_.front().timestamps.back();
        radar_data_buffer_.pop_front();

        int64_t next_radar_time = radar_data_buffer_.empty() ? temp_last_time : radar_data_buffer_.front().timestamps.front();
        
        auto next_start_it = std::lower_bound(imu_data_buffer_.begin(), imu_data_buffer_.end(), next_radar_time,
                                              [](const ImuData& a, int64_t b) { return a.timestamp < b; });
        int next_start_idx = std::distance(imu_data_buffer_.begin(), next_start_it);
        int erase_idx = std::max(0, next_start_idx - 1);
        
        if (erase_idx > 0) {
            imu_data_buffer_.erase(imu_data_buffer_.begin(), imu_data_buffer_.begin() + erase_idx);
        }
    }

    void publishOdometry(const Eigen::Matrix4d& pose, int64_t timestamp) {
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp.sec = static_cast<int32_t>(timestamp / 1000000LL);
        odom_msg.header.stamp.nanosec = static_cast<uint32_t>((timestamp % 1000000LL) * 1000);
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "radar";

        odom_msg.pose.pose.position.x = pose(0, 3);
        odom_msg.pose.pose.position.y = pose(1, 3);
        odom_msg.pose.pose.position.z = 0.0;

        Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        odom_msg.pose.pose.orientation.w = q.w();
        odometry_publisher_->publish(odom_msg);

        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = odom_msg.header.stamp;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "radar";
        transform.transform.translation.x = pose(0, 3);
        transform.transform.translation.y = pose(1, 3);
        transform.transform.translation.z = 0.0;
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(transform);
    }

    bool cumulativeReturnsNeeded() {
        return cumulated_returns_image_publisher_->get_subscription_count() > 0;
    }

    void publishLocalMap(const cv::Mat& local_map, const Eigen::Vector3d& xy_theta, int64_t timestamp) {
        if (local_map.type() != CV_8UC1) {
            RCLCPP_ERROR(this->get_logger(), "Local map cv::Mat is not CV_8UC1 (uint8).");
            return;
        }

        std_msgs::msg::Header header;
        header.stamp.sec = static_cast<int32_t>(timestamp / 1000000LL);
        header.stamp.nanosec = static_cast<uint32_t>((timestamp % 1000000LL) * 1000);
        header.frame_id = "radar";

        sensor_msgs::msg::Image::SharedPtr local_map_image_msg = cv_bridge::CvImage(header, "mono8", local_map).toImageMsg();
        local_map_image_publisher_->publish(*local_map_image_msg);

        dre::msg::LocalMapInfo map_info_msg;
        map_info_msg.header = header;
        map_info_msg.x = xy_theta(0);
        map_info_msg.y = xy_theta(1);
        map_info_msg.theta = xy_theta(2);
        map_info_msg.resolution = py::cast<float>(dro_opts_["direct"]["local_map_res"]);
        local_map_info_publisher_->publish(map_info_msg);

        nav_msgs::msg::Odometry local_map_odom_msg;
        local_map_odom_msg.header = header;
        local_map_odom_msg.header.frame_id = "odom";
        local_map_odom_msg.child_frame_id = "radar";
        local_map_odom_msg.pose.pose.position.x = xy_theta(0);
        local_map_odom_msg.pose.pose.position.y = xy_theta(1);
        local_map_odom_msg.pose.pose.position.z = 0.0;

        Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(0.0, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(xy_theta(2), Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;

        local_map_odom_msg.pose.pose.orientation.x = q.x();
        local_map_odom_msg.pose.pose.orientation.y = q.y();
        local_map_odom_msg.pose.pose.orientation.z = q.z();
        local_map_odom_msg.pose.pose.orientation.w = q.w();
        local_map_odometry_publisher_->publish(local_map_odom_msg);
    }

    void publishCumulativeReturns(const cv::Mat& cumulated_returns, int64_t timestamp) {
        if (cumulated_returns.type() != CV_8UC1) {
            RCLCPP_ERROR(this->get_logger(), "Cumulated returns cv::Mat is not CV_8UC1 (uint8).");
            return;
        }

        std_msgs::msg::Header header;
        header.stamp.sec = static_cast<int32_t>(timestamp / 1000000LL);
        header.stamp.nanosec = static_cast<uint32_t>((timestamp % 1000000LL) * 1000);
        header.frame_id = "radar";

        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "mono8", cumulated_returns).toImageMsg();
        cumulated_returns_image_publisher_->publish(*msg);
    }

    // ROS 2 objects
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Subscription<navtech_msgs::msg::RadarBScanMsg>::SharedPtr radar_subscription_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr local_map_odometry_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr local_map_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cumulated_returns_image_publisher_;
    rclcpp::Publisher<dre::msg::LocalMapInfo>::SharedPtr local_map_info_publisher_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // Data structures
    std::deque<RadarData> radar_data_buffer_;
    std::deque<ImuData> imu_data_buffer_;

    // Variables
    std::optional<int64_t> last_imu_time_;
    bool initialized_;
    bool first_;
    double imu_wait_timeout_sec_;
    std::optional<double> imu_wait_start_time_;
    std::string output_path_;
    
    // Stats
    int frame_count_;
    double sum_runtime_;
    double stats_start_time_;

    py::scoped_interpreter guard_;
    py::dict dro_opts_;
    std::unique_ptr<DroWrapper> dro_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DroNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}