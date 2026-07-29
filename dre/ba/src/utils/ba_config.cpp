#include "ba/utils/ba_config.hpp"
#include <stdexcept>

namespace ba {

OptimizationOptions load_optimization_options(const YAML::Node& config) {
    OptimizationOptions opts;

    if (config["max_iterations"])
        opts.max_iterations = config["max_iterations"].as<int>();
    if (config["convergence_tol"])
        opts.convergence_tol = config["convergence_tol"].as<double>();
    if (config["alpha"])
        opts.alpha = config["alpha"].as<double>();
    if (config["max_cost_increases"])
        opts.max_cost_increases = config["max_cost_increases"].as<int>();
    if (config["meas_std"])
        opts.meas_std = config["meas_std"].as<double>();
    if (config["rel_pose_prior"]) {
        if (config["use_pose_prior"])
            opts.use_pose_prior = config["use_pose_prior"].as<bool>();
        if (config["rel_pose_prior_translation_std"])
            opts.rel_pose_prior_translation_std = config["rel_pose_prior_translation_std"].as<double>();
        if (config["rel_pose_prior_rotation_std"])
            opts.rel_pose_prior_rotation_std = config["rel_pose_prior_rotation_std"].as<double>();
    }

    if (config["range_factor"])
        opts.range_factor = config["range_factor"].as<double>();
    if (config["use_cumul_thresh"])
        opts.use_cumul_thresh = config["use_cumul_thresh"].as<bool>();
    if (config["cumul_thresh"])
        opts.cumul_thresh = config["cumul_thresh"].as<double>();
    if (config["zero_thresh"])
        opts.zero_thresh = config["zero_thresh"].as<double>();
    if (config["num_coarse_iterations"])
        opts.num_coarse_iterations = config["num_coarse_iterations"].as<int>();
    if (config["coarse_downsample"])
        opts.coarse_downsample = config["coarse_downsample"].as<double>();
    if (config["refine_downsample"])
        opts.refine_downsample = config["refine_downsample"].as<double>();
    if (config["tile_size"])
        opts.tile_size = config["tile_size"].as<double>();
    if (config["max_loaded_scans"])
        opts.max_loaded_scans = config["max_loaded_scans"].as<int>();

    return opts;
}

FrameProcessingOptions load_frame_processing_options(const YAML::Node& config) {
    FrameProcessingOptions opts;

    if (config["input_type"])
        opts.input_type = config["input_type"].as<std::string>();
    if (config["local_map_res"])
        opts.local_map_res = config["local_map_res"].as<double>();
    if (config["max_dist"])
        opts.max_dist = config["max_dist"].as<double>();
    if (config["gauss_blur_sigma"])
        opts.gauss_blur_sigma = config["gauss_blur_sigma"].as<double>();
    if (config["adaptive_blur"])
        opts.adaptive_blur = config["adaptive_blur"].as<bool>();
    if (config["max_blur_sigma"])
        opts.max_blur_sigma = config["max_blur_sigma"].as<double>();
    if (config["min_int_val_tol"])
        opts.min_int_val_tol = config["min_int_val_tol"].as<double>();
    if (config["min_percent_nonzero"])
        opts.min_percent_nonzero = config["min_percent_nonzero"].as<double>();

    return opts;
}

BAOptions load_ba_options(const YAML::Node& config) {
    BAOptions opts;

    if (config["voxel_res"])
        opts.voxel_res = config["voxel_res"].as<double>();
    if (config["seq_id"])
        opts.seq_id = config["seq_id"].as<std::string>();
    if (config["save_H"])
        opts.save_H = config["save_H"].as<bool>();
    if (config["solver"])
        opts.solver = config["solver"].as<std::string>();
    if (config["init_poses"])
        opts.init_poses = config["init_poses"].as<std::string>();
    if (config["init_translation_std"])
        opts.init_translation_std = config["init_translation_std"].as<double>();
    if (config["init_rotation_std"])
        opts.init_rotation_std = config["init_rotation_std"].as<double>();
    if (config["frame_ranges"]) {
        opts.frame_ranges.clear();
        for (const auto& range_node : config["frame_ranges"]) {
            if (range_node.IsSequence() && range_node.size() == 2) {
                int start_frame = range_node[0].as<int>();
                int end_frame = range_node[1].as<int>();
                opts.frame_ranges.emplace_back(start_frame, end_frame);
            } else {
                throw std::runtime_error("Invalid frame range format in config file.");
            }
        }
    }
    // Keyframing
    if (config["keyframing"]) {
        if (config["keyframing"]["max_kf_dist"])
            opts.max_kf_dist = config["keyframing"]["max_kf_dist"].as<double>();
        if (config["keyframing"]["max_kf_rot"])
            opts.max_kf_rot = config["keyframing"]["max_kf_rot"].as<double>();
        if (config["keyframing"]["fix_first_scan"])
            opts.fix_first_scan = config["keyframing"]["fix_first_scan"].as<bool>();
    }
    // Frame processing
    if (config["frame_processing"]) {
        opts.frame_processing_opts = load_frame_processing_options(config["frame_processing"]);
    }
    // BA optimization
    if (config["optimization"]) {
        opts.optimization_opts = load_optimization_options(config["optimization"]);
    }

    return opts;
}

MappingOptions load_mapping_options(const YAML::Node& config) {
    MappingOptions opts;

    if (config["voxel_res"])
        opts.voxel_res = config["voxel_res"].as<double>();
    if (config["seq_id"])
        opts.seq_id = config["seq_id"].as<std::string>();
    if (config["pose_source"])
        opts.pose_source = config["pose_source"].as<std::string>();
    if (config["init_translation_std"])
        opts.init_translation_std = config["init_translation_std"].as<double>();
    if (config["init_rotation_std"])
        opts.init_rotation_std = config["init_rotation_std"].as<double>();
    if (config["estimate_location"])
        opts.estimate_location = std::filesystem::path(config["estimate_location"].as<std::string>());
    if (config["frame_ranges"]) {
        opts.frame_ranges.clear();
        for (const auto& range_node : config["frame_ranges"]) {
            if (range_node.IsSequence() && range_node.size() == 2) {
                int start_frame = range_node[0].as<int>();
                int end_frame = range_node[1].as<int>();
                opts.frame_ranges.emplace_back(start_frame, end_frame);
            } else {
                throw std::runtime_error("Invalid frame range format in config file.");
            }
        }
    }
    // Keyframing
    if (config["keyframing"]) {
        if (config["keyframing"]["max_kf_dist"])
            opts.max_kf_dist = config["keyframing"]["max_kf_dist"].as<double>();
        if (config["keyframing"]["max_kf_rot"])
            opts.max_kf_rot = config["keyframing"]["max_kf_rot"].as<double>();
        if (config["keyframing"]["fix_first_scan"])
            opts.fix_first_scan = config["keyframing"]["fix_first_scan"].as<bool>();
    }

    // Frame processing
    if (config["frame_processing"]) {
        opts.frame_processing_opts = load_frame_processing_options(config["frame_processing"]);
    }

    // Map optimization
    if (config["optimization"]) {
        opts.optimization_opts = load_optimization_options(config["optimization"]);
    }

    return opts;
}

LocalizationOptions load_localization_options(const YAML::Node& config) {
    LocalizationOptions opts;

    if (config["seq_id"])
        opts.seq_id = config["seq_id"].as<std::string>();
    if (config["map_seq_id"])
        opts.map_seq_id = config["map_seq_id"].as<std::string>();
    if (config["map_location"])
        opts.map_location = std::filesystem::path(config["map_location"].as<std::string>());
    if (config["start_frame"])
        opts.start_frame = config["start_frame"].as<int>();
    if (config["end_frame"])
        opts.end_frame = config["end_frame"].as<int>();
    if (config["odometry_prior"]) {
        if (config["odometry_prior"]["use_odometry_prior"])
            opts.use_odometry_prior = config["odometry_prior"]["use_odometry_prior"].as<bool>();
        if (config["odometry_prior"]["translation_std"])
            opts.odom_translation_std = config["odometry_prior"]["translation_std"].as<double>();
        if (config["odometry_prior"]["rotation_std"])
            opts.odom_rotation_std = config["odometry_prior"]["rotation_std"].as<double>();
    }

    // Localization frame processing
    if (config["frame_processing"]) {
        opts.frame_processing_opts = load_frame_processing_options(config["frame_processing"]);
    }

    // Localization optimization
    if (config["optimization"]) {
        opts.optimization_opts = load_optimization_options(config["optimization"]);
    }

    return opts;
}

Options load_options(const YAML::Node& config) {
    Options opts;

    if (config["num_threads"])
        opts.num_threads = config["num_threads"].as<int>();
    if (config["seed"])
        opts.seed = config["seed"].as<int>();

    if (config["data"]) {
        if (config["data"]["data_path"])
            opts.data_path = std::filesystem::path(config["data"]["data_path"].as<std::string>());
        if (config["data"]["meas_path"])
            opts.meas_path = std::filesystem::path(config["data"]["meas_path"].as<std::string>());
    } else {
        throw std::runtime_error("Config file missing required 'data' section with required fields 'data_path' and 'meas_path'");
    }

    if (config["output"]) {
        if (config["output"]["save_result"])
            opts.save_result = config["output"]["save_result"].as<bool>();
        else
            opts.save_result = false;
        if (config["output"]["output_path"])
            opts.output_path = std::filesystem::path(config["output"]["output_path"].as<std::string>());
        if (config["output"]["visualize"])
            opts.visualize_result = config["output"]["visualize"].as<bool>();
        else
            opts.visualize_result = true;
    }

    if (config["ba"]) {
        opts.ba_opts = load_ba_options(config["ba"]);
    }

    if (config["mapping"]) {
        opts.map_opts = load_mapping_options(config["mapping"]);
    }

    if (config["localization"]) {
        opts.loc_opts = load_localization_options(config["localization"]);
    }

    return opts;
}

} // namespace ba