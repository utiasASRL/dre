// problem.hpp
#pragma once

#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include <ba/solver/result.hpp>

#include <iostream>
#include <ankerl/unordered_dense.h>
#include <lgmath/se3/Transformation.hpp>
#include <filesystem>
namespace fs = std::filesystem;

namespace ba {

class Problem {
public:
    using PriorMap = ankerl::unordered_dense::map<std::pair<int32_t, int32_t>, lgmath::se3::Transformation>;
    void initialize() {
        validate_opts();
        init_seq_id();
        get_scan_indeces();
        preload_images();
        init_scans_and_map();
        initialized_ = true;
    }

    virtual void finalize() = 0;

    // Automatically clean up temporary directory if it exists
    virtual ~Problem() {
        cleanup_temp_dir();
    }

    // Accessors
    Options& opts() { return opts_; }
    VoxelMap& voxel_map() { return voxel_map_; }
    ScanManager& scan_manager() { return scan_manager_; }
    Result& result() { return result_; }
    PriorMap& pose_priors() { return pose_priors_; }
    bool is_initialized() const { return initialized_; }
    std::string seq_id() const { return seq_id_; }
    std::string type() const { return type_; }

    // Shared functions
    void preload_images();

protected:
    Problem(std::string type, Options& opts)
                : type_(type),
                    opts_(opts),
                    voxel_map_((type == "ba") ? opts.ba_opts.voxel_res :
                                (type == "map") ? opts.map_opts.voxel_res :
                                1.0),
                    scan_manager_((type == "ba") ? opts_.ba_opts.optimization_opts.max_loaded_scans :
                                    (type == "map") ? opts_.map_opts.optimization_opts.max_loaded_scans :
                                    (type == "loc") ? opts_.loc_opts.optimization_opts.max_loaded_scans : 1.0),
                    result_(voxel_map_, scan_manager_, opts_.output_path),
                    initialized_(false) {
                }

    virtual void validate_opts() = 0;
    virtual void init_seq_id() = 0;
    virtual void get_scan_indeces() = 0;
    virtual void init_scans_and_map() = 0;

    void cleanup_temp_dir() noexcept {
        if (!temp_dir_.empty() && fs::exists(temp_dir_)) {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            // intentionally ignore errors in destructor
        }
    }

    std::string type_; // "ba", "map", or "loc"
    Options& opts_;
    std::string seq_id_;
    VoxelMap voxel_map_;
    ScanManager scan_manager_;
    Result result_;
    // TODO: Deal with pose priors better
    PriorMap pose_priors_;
    bool initialized_;
    fs::path temp_dir_;

    std::vector<int> scan_indices_;
    std::vector<fs::path> img_paths_;
    std::vector<fs::path> cumul_paths_;
    std::vector<int64_t> timestamps_;
    std::vector<lgmath::se3::Transformation> T_est_abs_list_;
    std::vector<lgmath::se3::Transformation> T_gt_abs_list_;

    lgmath::se3::Transformation T_est_0_abs_;
    lgmath::se3::Transformation T_gt_0_abs_;
};


}   // namespace ba