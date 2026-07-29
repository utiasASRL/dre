// io_utils.hpp
#pragma once

#include <filesystem>
#include <vector>
#include <lgmath/se3/Transformation.hpp>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace ba {

    Eigen::Matrix3d toRoll(const double &r);
    Eigen::Matrix3d toPitch(const double &p);
    Eigen::Matrix3d toYaw(const double &y);
    Eigen::Matrix3d rpy2rot(const double &r, const double &p, const double &y);
    double roundToPi(double value);

    Eigen::Matrix4d load_T_radar_applanix(const std::filesystem::path &path);

    // Load groundtruth poses and timestamps from gps_post_process.csv file. Time in seconds.
    // If no Boreas groundtruth exists at the path but pogo output does, the pogo
    // trajectory is loaded as the reference instead (for live data without groundtruth).
    void load_groundtruth_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times);

    // Load pogo poses and timestamps from pogo output pose file. Time in seconds.
    void load_pogo_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times);

    // Load DRO poses and timestamps from dro output pose file. Time in seconds.
    void load_dro_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times);

    // Get interpolated pose at query time using linear interpolation for translation and SLERP for rotation
    lgmath::se3::Transformation get_interpolated_pose(
        const std::vector<lgmath::se3::Transformation> &all_poses,
        const std::vector<double> &all_times,
        double query_time);

    // Handle local maps
    void save_img_bin(const std::filesystem::path &filepath, const cv::Mat &img);
    cv::Mat load_img_bin(const std::filesystem::path &filepath);

    // Utility to save Eigen matrix in binary format
    void saveBinary(const Eigen::MatrixXd& M, const std::string& path);
}