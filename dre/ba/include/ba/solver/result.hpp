#pragma once

#include <ba/scans/manager.hpp>
#include <ba/map/voxel_map.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace ba {

class Result {
public:
    Result(VoxelMap &voxel_map, ScanManager &scan_manager, const fs::path& output_dir)
        : voxel_map_(voxel_map),
          scan_manager_(scan_manager),
          output_dir_(output_dir) {
    }
        
    // Immutable accessors
    const std::vector<double>& cost_history() const { return cost_history_; }
    const std::vector<Eigen::Vector3d>& rmse_history() const { return rmse_history_; }
    const std::vector<double>& ate_history() const { return ate_history_; }
    const std::vector<double>& epe_history() const { return epe_history_; }
    const std::string output_dir() const { return output_dir_.string(); }

    // Mutable accessors
    VoxelMap& voxel_map() { return voxel_map_; }
    ScanManager& scan_manager() { return scan_manager_; }

    // Adding info
    void add_cost(double cost) {
        cost_history_.push_back(cost);
    }
    void add_rmse(const Eigen::Vector3d& rmse) {
        rmse_history_.push_back(rmse);
    }
    void add_ate(double ate) {
        ate_history_.push_back(ate);
    }
    void add_epe(double epe) {
        epe_history_.push_back(epe);
    }
    
    // Result output
    void save_rmse_cost_to_csv(const fs::path& output_dir = {}) const;
    void save_poses_to_csv(const fs::path& output_path = {}) const;
    void save_voxel_map(const fs::path& optional_output_dir = {}) const;
    void save_full_result() const {
        save_rmse_cost_to_csv();
        save_poses_to_csv();
        save_voxel_map();
    }
    void visualize_map() const;
    void visualize_all_results() const;

private:
    VoxelMap& voxel_map_;
    ScanManager& scan_manager_;
    const fs::path& output_dir_;

    std::vector<double> cost_history_;
    std::vector<Eigen::Vector3d> rmse_history_;
    std::vector<double> ate_history_;
    std::vector<double> epe_history_;
    const fs::path csv_path_ = output_dir_ / "rmse_cost_history.csv";
    const fs::path poses_path_ = output_dir_ / "ba_traj.csv";
    const fs::path voxel_path_ = output_dir_ / "voxel_map.bin";

};

} // namespace ba