#include "ba/utils/io_utils.hpp"
#include <fstream>
#include <iostream>

namespace ba {

Eigen::Matrix3d toRoll(const double &r) {
    Eigen::Matrix3d roll;
    roll << 1, 0, 0, 0, cos(r), sin(r), 0, -sin(r), cos(r);
    return roll;
}

Eigen::Matrix3d toPitch(const double &p) {
    Eigen::Matrix3d pitch;
    pitch << cos(p), 0, -sin(p), 0, 1, 0, sin(p), 0, cos(p);
    return pitch;
}

Eigen::Matrix3d toYaw(const double &y) {
    Eigen::Matrix3d yaw;
    yaw << cos(y), sin(y), 0, -sin(y), cos(y), 0, 0, 0, 1;
    return yaw;
}

Eigen::Matrix3d rpy2rot(const double &r, const double &p, const double &y) {
    return toRoll(r) * toPitch(p) * toYaw(y);
}

double roundToPi(double value) {
    return std::round(value / M_PI) * M_PI;
}

Eigen::Matrix4d load_T_radar_applanix(const std::filesystem::path &path) {
    std::ifstream ifs1(path / "calib" / "T_applanix_lidar.txt", std::ios::in);
    std::ifstream ifs2(path / "calib" / "T_radar_lidar.txt", std::ios::in);

    Eigen::Matrix4d T_applanix_lidar_mat;
    for (size_t row = 0; row < 4; row++)
        for (size_t col = 0; col < 4; col++) ifs1 >> T_applanix_lidar_mat(row, col);

    Eigen::Matrix4d T_radar_lidar_mat;
    for (size_t row = 0; row < 4; row++)
        for (size_t col = 0; col < 4; col++) ifs2 >> T_radar_lidar_mat(row, col);

    return Eigen::Matrix4d(T_radar_lidar_mat * T_applanix_lidar_mat.inverse());
}

void load_groundtruth_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times) {
    std::ifstream ifs(path / "pose_graph_traj.txt", std::ios::in);
    // Clear header line
    std::string line;
    std::getline(ifs, line);
    // Loop through all gt data
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> data;
        for (std::string str; std::getline(ss, str, ' ');)
        data.push_back(std::stod(str));

        // Store pogo pose
        Eigen::Matrix4d T_ab_mat = Eigen::Matrix4d::Identity();
        T_ab_mat.block<3, 1>(0, 3) << data[1], data[2], 0.0;
        T_ab_mat.block<2, 2>(0, 0) << cos(data[3]), -sin(data[3]),
                                   sin(data[3]),  cos(data[3]);
        lgmath::se3::Transformation T_ab = lgmath::se3::Transformation(T_ab_mat);
        all_poses.push_back(T_ab);
        all_times.push_back(data[0] / 1e6);  // convert to seconds
    }
}

void load_pogo_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times) {
    std::ifstream ifs(path / "pose_graph_traj.txt", std::ios::in);
    // Clear header line
    std::string line;
    std::getline(ifs, line);
    // Loop through all gt data
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> data;
        for (std::string str; std::getline(ss, str, ' ');)
        data.push_back(std::stod(str));

        // Store pogo pose
        Eigen::Matrix4d T_ab_mat = Eigen::Matrix4d::Identity();
        T_ab_mat.block<3, 1>(0, 3) << data[1], data[2], 0.0;
        T_ab_mat.block<2, 2>(0, 0) << cos(data[3]), -sin(data[3]),
                                   sin(data[3]),  cos(data[3]);
        lgmath::se3::Transformation T_ab = lgmath::se3::Transformation(T_ab_mat);
        all_poses.push_back(T_ab);
        all_times.push_back(data[0] / 1e6);  // convert to seconds
    }
}

void load_dro_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times) {
    // Full path is in odometry_2d with seq_id.txt file
    std::filesystem::path full_path = path / "odometry_2d" / (path.filename().string() + ".txt");
    std::ifstream ifs(full_path, std::ios::in);
    // Clear header line
    std::string line;
    std::getline(ifs, line);
    // Loop through all gt data
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> data;
        for (std::string str; std::getline(ss, str, ' ');)
        data.push_back(std::stod(str));

        // Store pogo pose
        Eigen::Matrix4d T_ab_mat = Eigen::Matrix4d::Identity();
        T_ab_mat.block<3, 1>(0, 3) << data[1], data[2], 0.0;
        T_ab_mat.block<2, 2>(0, 0) << cos(data[3]), -sin(data[3]),
                                   sin(data[3]),  cos(data[3]);
        lgmath::se3::Transformation T_ab = lgmath::se3::Transformation(T_ab_mat);
        all_poses.push_back(T_ab);
        all_times.push_back(data[0] / 1e6);  // convert to seconds
    }
}

lgmath::se3::Transformation get_interpolated_pose(
    const std::vector<lgmath::se3::Transformation> &all_poses,
    const std::vector<double> &all_times,
    double query_time) {
    const size_t N = all_times.size();
    // Handle exceptions
    if (N == 0) throw std::runtime_error("Empty pose/time vectors");
    if (query_time < all_times.front()) throw std::runtime_error("Query time before first timestamp");
    if (query_time > all_times.back()) throw std::runtime_error("Query time after last timestamp");

    // Sort for nearest idx
    auto it = std::lower_bound(all_times.begin(), all_times.end(), query_time);
    size_t idx = std::distance(all_times.begin(), it);

    // Handle idx exceptions right on boundary
    if (idx == 0) return all_poses.front();
    if (idx >= N) return all_poses.back();

    // We already handled boundaries, so idx >= 1 here
    size_t i = idx - 1;
    // Compute interpolation alpha
    double t0 = all_times[i];
    double t1 = all_times[i + 1];
    double alpha = (query_time - t0) / (t1 - t0);

    Eigen::Matrix4d T_0 = all_poses[i].matrix();
    Eigen::Matrix4d T_1 = all_poses[i + 1].matrix();

    // Interpolate the translation linearly
    Eigen::Vector3d t0_vec = T_0.block<3, 1>(0, 3);
    Eigen::Vector3d t1_vec = T_1.block<3, 1>(0, 3);
    Eigen::Vector3d t_interp = (1.0 - alpha) * t0_vec + alpha * t1_vec;

    // Interpolate the rotation using slerp
    Eigen::Matrix3d R0 = T_0.block<3, 3>(0, 0);
    Eigen::Matrix3d R1 = T_1.block<3, 3>(0, 0);
    Eigen::Quaterniond q0(R0);
    Eigen::Quaterniond q1(R1);
    Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);
    Eigen::Matrix3d R_interp = q_interp.toRotationMatrix();

    // Construct the interpolated transformation
    Eigen::Matrix4d T_interp = Eigen::Matrix4d::Identity();
    T_interp.block<3, 3>(0, 0) = R_interp;
    T_interp.block<3, 1>(0, 3) = t_interp;

    return lgmath::se3::Transformation(T_interp);
}


void save_img_bin(const std::filesystem::path &filepath, const cv::Mat& img) {
    std::ofstream ofs(filepath, std::ios::binary);
    int32_t rows = img.rows;
    int32_t cols = img.cols;
    ofs.write((char*)&rows, sizeof(rows));
    ofs.write((char*)&cols, sizeof(cols));
    ofs.write((char*)img.ptr<float>(), rows * cols * sizeof(float));
}

cv::Mat load_img_bin(const std::filesystem::path &filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open image file: " + filepath.string());
    }
    
    int32_t rows, cols;
    ifs.read((char*)&rows, sizeof(rows));
    ifs.read((char*)&cols, sizeof(cols));
    
    if (!ifs.good()) {
        throw std::runtime_error("Failed to read image dimensions from file: " + filepath.string());
    }
    
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Invalid image dimensions from file " + filepath.string() + 
                                ": rows=" + std::to_string(rows) + ", cols=" + std::to_string(cols));
    }
    
    // Check for unreasonably large dimensions (e.g., > 100,000 pixels in any dimension)
    if (rows > 100000 || cols > 100000) {
        throw std::runtime_error("Image dimensions suspiciously large from file " + filepath.string() + 
                                ": rows=" + std::to_string(rows) + ", cols=" + std::to_string(cols) + 
                                ". This may indicate file corruption.");
    }
    
    cv::Mat img(rows, cols, CV_32F);
    ifs.read((char*)img.ptr<float>(), rows * cols * sizeof(float));
    
    if (!ifs.good()) {
        throw std::runtime_error("Failed to read image data from file: " + filepath.string());
    }
    
    return img;
}

void saveBinary(const Eigen::MatrixXd& M, const std::string& path)
{
    std::ofstream out(path, std::ios::binary);
    int rows = M.rows(), cols = M.cols();
    out.write((char*)&rows, sizeof(int));
    out.write((char*)&cols, sizeof(int));
    out.write((char*)M.data(), rows * cols * sizeof(double));
}

} // namespace ba