#include <ba/solver/result.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <lgmath/se2/Transformation.hpp>
#include <future>
#include <cstdlib>

namespace ba {

void Result::save_rmse_cost_to_csv(const fs::path& optional_output_dir) const {
    fs::path dir = optional_output_dir.empty() ? csv_path_: optional_output_dir;

    std::ofstream file(dir);
    file << "cost,ate,epe,rmse_x,rmse_y,rmse_yaw\n";
    for (std::size_t i = 0; i < rmse_history_.size(); ++i) {
        file << cost_history_[i] << "," << ate_history_[i] << ","
                << epe_history_[i] << ","
                << rmse_history_[i](0) << "," << rmse_history_[i](1)<< ","
                << rmse_history_[i](2) << "\n";
    }
}

void Result::save_poses_to_csv(const fs::path& optional_output_dir) const {
    // Compute ATE
    scan_manager_.compute_ate();
    fs::path dir = optional_output_dir.empty() ? poses_path_ : optional_output_dir;

    std::ofstream file(dir);
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    for (int scan_id : scan_id_list) {
        auto scan = scan_manager_.get_scan(scan_id);
        lgmath::se2::Transformation T_est = scan->pose2d();
        Eigen::Vector2d t_est = T_est.r_ab_inb();
        double yaw_est = T_est.vec()(2);
        file << scan->timestamp() << "," << t_est(0) << "," << t_est(1) << "," << yaw_est << "\n";
    }
}

void Result::save_voxel_map(const fs::path& optional_output_dir) const {
    // Compute ATE to save with voxel map
    scan_manager_.compute_ate();
    fs::path dir = optional_output_dir.empty() ? voxel_path_ : optional_output_dir;
    voxel_map_.save_to_file(dir.string(), scan_manager_);
}

void Result::visualize_map() const {
    // Visualize voxel map
    // voxel_map_.visualize();

    // Check if output_dir is empty
    std::string cmd_errors;
    std::string cmd_map;
    fs::path temp_dir;
    if (output_dir_.empty()) {
        temp_dir = fs::temp_directory_path() / "dr_ba_temp_visualization";
        fs::create_directories(temp_dir);
        std::cout << "Output directory not set. Using temporary directory: " << temp_dir.string() << std::endl;

        // Save all map results to temporary directory
        fs::path temp_poses_path = temp_dir / "scan_poses.csv";
        save_poses_to_csv(temp_poses_path);
        fs::path temp_voxel_path = temp_dir / "voxel_map.bin";
        save_voxel_map(temp_voxel_path);
        cmd_map = "python3 /home/asrl/ros2_ws/dr_ba/ba_py/visualize_voxel_map.py " + temp_dir.string();
    } else {
        cmd_map = "python3 /home/asrl/ros2_ws/dr_ba/ba_py/visualize_voxel_map.py " + output_dir_.string();
    }

    auto f1 = std::async(std::launch::async, [&]() {
        int ret = std::system(cmd_map.c_str());
        if (ret != 0)
            throw std::runtime_error("Error executing command: " + cmd_map);
    });

    f1.get();

    // Clean up temporary directory
    if (output_dir_.empty()) {
        std::cout << "Removing temporary directory: " << temp_dir.string() << std::endl;
        fs::remove_all(temp_dir);
    }
}

void Result::visualize_all_results() const {
    // Check if output_dir is empty
    std::string cmd_errors;
    std::string cmd_map;
    fs::path temp_dir;
    if (output_dir_.empty()) {
        temp_dir = fs::temp_directory_path() / "dr_ba_temp_visualization";
        fs::create_directories(temp_dir);
        std::cout << "Output directory not set. Using temporary directory: " << temp_dir.string() << std::endl;
        // Clear directory if it already exists
        for (const auto& entry : fs::directory_iterator(temp_dir)) {
            fs::remove_all(entry.path());
        }

        // Save all results to temporary directory
        fs::path temp_csv_path = temp_dir / "rmse_cost_history.csv";
        save_rmse_cost_to_csv(temp_csv_path);
        fs::path temp_poses_path = temp_dir / "scan_poses.csv";
        save_poses_to_csv(temp_poses_path);
        fs::path temp_voxel_path = temp_dir / "voxel_map.bin";
        save_voxel_map(temp_voxel_path);
        cmd_errors = "python3 /home/asrl/ros2_ws/dr_ba/ba/app/plot_errors.py " + temp_csv_path.string();
        cmd_map = "python3 /home/asrl/ros2_ws/dr_ba/ba_py/visualize_voxel_map.py " + temp_dir.string();
    } else {
        cmd_errors = "python3 /home/asrl/ros2_ws/dr_ba/ba/app/plot_errors.py " + csv_path_.string() + " " + output_dir_.string();
        cmd_map = "python3 /home/asrl/ros2_ws/dr_ba/ba_py/visualize_voxel_map.py " + output_dir_.string();
    }

    auto f1 = std::async(std::launch::async, [&]() {
        int ret = std::system(cmd_map.c_str());
        if (ret != 0)
            throw std::runtime_error("Error executing command: " + cmd_map);
    });

    auto f2 = std::async(std::launch::async, [&]() {
        int ret = std::system(cmd_errors.c_str());
        if (ret != 0)
            throw std::runtime_error("Error executing command: " + cmd_errors);
    });

    f1.get();
    f2.get();

    // Clean up temporary directory
    if (output_dir_.empty()) {
        std::cout << "Removing temporary directory: " << temp_dir.string() << std::endl;
        fs::remove_all(temp_dir);
    }

}

}   // namespace ba