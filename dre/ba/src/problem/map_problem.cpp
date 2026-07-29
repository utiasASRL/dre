#include <ba/problem/map_problem.hpp>
#include <ba/utils/io_utils.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <ba/scans/local_map_scan.hpp>

namespace ba {

void MapProblem::validate_opts() {
    // Check that mapping opts exist
    if (opts_.map_opts.seq_id.empty()) {
        throw std::invalid_argument("MapProblem: seq_id must be specified in mapping options");
    }
    if (opts_.map_opts.pose_source == "estimate") {
        if (opts_.map_opts.estimate_location.empty()) {
            throw std::invalid_argument("MapProblem: estimate_location must be specified in mapping options when pose_source is 'estimate'");
        }
    }
}

void MapProblem::init_seq_id() {
    seq_id_ = opts_.map_opts.seq_id;
}

void MapProblem::get_scan_indeces() {
    std::cout << "Selecting scan indices based on frame ranges..." << std::endl;

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts_.data_path / seq_id_, all_gt_poses, all_gt_times);

    // Load pogo poses
    std::vector<lgmath::se3::Transformation> all_pogo_poses;
    std::vector<double> all_pogo_times;
    ba::load_pogo_poses_and_times(opts_.meas_path / seq_id_, all_pogo_poses, all_pogo_times);

    // Load DRO poses
    std::vector<lgmath::se3::Transformation> all_dro_poses;
    std::vector<double> all_dro_times;
    ba::load_dro_poses_and_times(opts_.meas_path / seq_id_, all_dro_poses, all_dro_times);

    // Initialize looping through trajectory
    lgmath::se3::Transformation T_est_abs_0(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    lgmath::se3::Transformation T_kf_prev(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));  // Previous keyframe pose
    int kf_prev_id = 0;

    // Loop through all images
    // TODO: This is already done in preload_images, refactor to avoid duplicate work
    fs::path all_img_dir = opts_.meas_path / seq_id_ / opts_.map_opts.frame_processing_opts.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        if (entry.path().extension() != ".png") {
            continue;
        }
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    // Check validity of frame ranges
    int num_scans = files.size();
    int max_frame = 0;
    std::vector<std::pair<int, int>> frame_ranges = opts_.map_opts.frame_ranges;
    for (auto& range : frame_ranges) {
        if (range.second == -1) {
            range.second = num_scans - 1;
        }
        if (range.second < range.first) {
            throw std::invalid_argument("Invalid frame range: [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "]");
        }
        if (range.first < 0 || range.second >= num_scans) {
            throw std::out_of_range("Frame range out of bounds: [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "] for sequence " + seq_id_);
        }
        max_frame = std::max(max_frame, range.second);
    }

    int num_checked = -1;

    if (opts_.map_opts.pose_source == "estimate") {
        // Load in voxel map from estimates
        std::string estimate_location = opts_.map_opts.estimate_location.string() + "/voxel_map.bin";
        std::cout << "Loading voxel map from estimates: " << estimate_location << std::endl;
        voxel_map_.load_poses_from_file(estimate_location);
        const std::vector<int> pose_ids = voxel_map_.pose_ids();
        const std::vector<lgmath::se2::Transformation> poses_se2 = voxel_map_.poses();

        for (size_t i=0; i<files.size(); i++) {
            const auto& path = files[i];
            int64_t timestamp = std::stoll(path.stem().string()); // in micro
            double timestamp_seconds = timestamp / 1e6; // convert to seconds
            // Only consider files ending with .png
            if (path.extension() != ".png") {
                continue;
            }
            num_checked++;

            // Check if frame in desired ranges
            bool in_range = false;
            for (const auto& range : frame_ranges) {
                if (num_checked >= range.first && num_checked <= range.second) {
                    in_range = true;
                    break;
                }
            }

            if (i == 0) {
                T_gt_0_abs_ = ba::get_interpolated_pose(all_gt_poses, all_gt_times, std::stod(files[0].stem().string()) / 1e6);
            }
            // Check if current index is in pose_ids
            if (std::find(pose_ids.begin(), pose_ids.end(), i) == pose_ids.end()) {
                continue;
            }
            if (!in_range) {
                continue;
            }
            lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
            // Get SE2 pose from voxel map
            int pose_idx = std::distance(pose_ids.begin(), std::find(pose_ids.begin(), pose_ids.end(), i));
            lgmath::se2::Transformation T_est_se2 = poses_se2[pose_idx];
            scan_indices_.push_back(i);
            T_est_abs_list_.push_back(T_est_se2.toSE3());
            T_gt_abs_list_.push_back(T_gt_abs);
        }
        return;
    }


    int num_loaded = 0;
    for (const auto& path : files) {
        // Only consider files ending with .png
        if (path.extension() != ".png") {
            continue;
        }
        num_checked++;

        // Check if frame in desired ranges
        bool in_range = false;
        for (const auto& range : frame_ranges) {
            if (num_checked >= range.first && num_checked <= range.second) {
                in_range = true;
                break;
            }
        }

        // Load in scan pose
        int64_t timestamp = std::stoll(path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds

        // Load in gt pose
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);

        // Load in initial guess pose
        lgmath::se3::Transformation T_est_abs;
        if (opts_.map_opts.pose_source == "pogo") {
            T_est_abs = ba::get_interpolated_pose(all_pogo_poses, all_pogo_times, timestamp_seconds);
        } else if (opts_.map_opts.pose_source == "gt") {
            T_est_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        } else if (opts_.map_opts.pose_source == "dro") {
            T_est_abs = ba::get_interpolated_pose(all_dro_poses, all_dro_times, timestamp_seconds);
        } else {
            throw std::invalid_argument("Invalid pose_source option: " + opts_.map_opts.pose_source);
        }

        if (num_loaded != 0) {
            // Check if this pose is a keyframe
            lgmath::se3::Transformation T_kf_rel = T_est_abs.inverse() * T_kf_prev;
            double del_x = T_kf_rel.r_ab_inb()(0);
            double del_y = T_kf_rel.r_ab_inb()(1);
            double del_theta = T_kf_rel.vec()(5); // Yaw angle
            double translation_mag = std::sqrt(std::pow(del_x, 2) + std::pow(del_y, 2));
            double rotation_mag = std::abs(del_theta) * 180.0 / M_PI; // convert to degrees
            if (translation_mag < opts_.map_opts.max_kf_dist && rotation_mag < opts_.map_opts.max_kf_rot) {
                // Not a keyframe, skip
                continue;
            }
            // Set up prior from prev keyframe radar frame to this keyframe radar frame
            pose_priors_[{kf_prev_id, num_checked}] = T_kf_rel;
        }
        // Always want to root map in first pose, whether we use it or not
        if (num_checked == 0) {
            T_est_0_abs_ = T_est_abs;
            T_gt_0_abs_ = T_gt_abs;
        }

        // We've decided this is a keyframe!
        kf_prev_id = num_checked;
        T_kf_prev = T_est_abs;

        if (!in_range) {
            // We want to do keyframing in the same way for all frames, but dont
            // want to load out of range
            continue;
        }

        // Add to list of scan indices to load
        scan_indices_.push_back(num_checked);
        T_est_abs_list_.push_back(T_est_abs);
        T_gt_abs_list_.push_back(T_gt_abs);

        num_loaded++;
    }

    // Sort scan indices
    std::sort(scan_indices_.begin(), scan_indices_.end());
}
    
void MapProblem::init_scans_and_map() {
    if (opts_.map_opts.pose_source == "estimate") {
        init_scans_and_map_from_estimates();
    } else {
        init_scans_and_map_from_data();
    }
}

void MapProblem::init_scans_and_map_from_estimates() {
    // Load in voxel map from estimates
    std::string estimate_location = opts_.map_opts.estimate_location.string() + "/voxel_map.bin";
    std::cout << "Loading voxel map from estimates: " << estimate_location << std::endl;
    std::cout << "Map spanning from " << timestamps_.front() << " to " << timestamps_.back() << " microseconds." << std::endl;
    voxel_map_.load_poses_from_file(estimate_location);
    const std::vector<int> pose_ids = voxel_map_.pose_ids();
    const std::vector<lgmath::se2::Transformation> poses_se2 = voxel_map_.poses();
    // Re-initialize a new voxel_map
    voxel_map_ = VoxelMap(opts_.map_opts.voxel_res);

    for (size_t i=0; i < scan_indices_.size(); i++) {
        int idx = scan_indices_[i];

        // Load in gt pose
        lgmath::se3::Transformation T_gt_rel = T_gt_0_abs_.inverse() * T_gt_abs_list_[i];
        T_gt_rel = T_gt_rel.toSE2().toSE3();

        // Get SE2 pose from voxel map
        int pose_idx = std::distance(pose_ids.begin(), std::find(pose_ids.begin(), pose_ids.end(), idx));
        lgmath::se2::Transformation T_est_se2 = poses_se2[pose_idx];

        // Load in image paths
        const auto& img_path = img_paths_[i];
        std::optional<fs::path> cumul_img_path = std::nullopt;
        if (opts_.map_opts.optimization_opts.use_cumul_thresh) {
            if (cumul_paths_.empty()) {
                throw std::runtime_error("Cumulative image paths are empty but use_cumul_thresh is true.");
            }
            cumul_img_path = cumul_paths_[i];
        }

        // Create LocalMapScan and add to scan manager
        auto scan = std::make_shared<LocalMapScan>(
            timestamps_[i],
            idx,
            opts_.map_opts.frame_processing_opts.local_map_res,
            opts_.map_opts.optimization_opts,
            T_est_se2.toSE3(),
            T_gt_rel,
            img_path,
            cumul_img_path);

        scan_manager_.add_scan(scan);
        voxel_map_.init_map(T_est_se2, opts_.map_opts.frame_processing_opts.max_dist, idx);
    }
}

void MapProblem::init_scans_and_map_from_data() {
    std::cout << "Initializing scans and map from data for sequence: " << seq_id_ << std::endl;

    // Initialize uniform distribution for noise
    std::uniform_real_distribution<double> translation_dist(-opts_.map_opts.init_translation_std, opts_.map_opts.init_translation_std);
    double rotation_std_rad = opts_.map_opts.init_rotation_std * M_PI / 180.0;
    std::uniform_real_distribution<double> rotation_dist(-rotation_std_rad, rotation_std_rad);
    std::mt19937 rng(opts_.seed >= 0 ? opts_.seed : std::random_device{}());

    lgmath::se3::Transformation T_est_abs_0 = T_est_abs_list_[0];
    for (size_t i=0; i < scan_indices_.size(); i++) {
        int idx = scan_indices_[i];
        // Load in estimated pose
        lgmath::se3::Transformation T_est_rel = T_est_0_abs_.inverse() * T_est_abs_list_[i];
        T_est_rel = T_est_rel.toSE2().toSE3();

        // Load in gt pose
        lgmath::se3::Transformation T_gt_rel = T_gt_0_abs_.inverse() * T_gt_abs_list_[i];
        T_gt_rel = T_gt_rel.toSE2().toSE3();

        // Load in image paths
        const auto& img_path = img_paths_[i];
        std::optional<fs::path> cumul_img_path = std::nullopt;
        if (opts_.map_opts.optimization_opts.use_cumul_thresh) {
            if (cumul_paths_.empty()) {
                throw std::runtime_error("Cumulative image paths are empty but use_cumul_thresh is true.");
            }
            cumul_img_path = cumul_paths_[i];
        }

        // Add noise to initial pose estimate if specified
        if (opts_.map_opts.init_translation_std > 0.0 || opts_.map_opts.init_rotation_std > 0.0) {
            // Add noise to gt pose sampled from uniform distribution
            Eigen::Vector3d noise;
            noise << translation_dist(rng), translation_dist(rng), rotation_dist(rng);
            lgmath::se3::Transformation T_noise = lgmath::se2::Transformation(noise).toSE3();
            T_est_rel = T_est_rel * T_noise;
        }

        auto scan = std::make_shared<LocalMapScan>(
            timestamps_[i],
            idx,
            opts_.map_opts.frame_processing_opts.local_map_res,
            opts_.map_opts.optimization_opts,
            T_est_rel,
            T_gt_rel,
            img_path,
            cumul_img_path);

        scan_manager_.add_scan(scan);
    }

    std::cout << "Scan manager has " << scan_manager_.num_scans() << " scans." << std::endl;

    // Initialize voxel map around all scans
    std::cout << "Initializing voxel map..." << std::endl;
    for (int scan_id : scan_manager_.get_all_scan_ids()) {
        auto scan = scan_manager_.get_scan(scan_id);
        voxel_map_.init_map(scan->pose(), opts_.map_opts.frame_processing_opts.max_dist, scan_id);
    }

    std::pair<double, double> x_bounds = voxel_map_.x_bounds();
    std::pair<double, double> y_bounds = voxel_map_.y_bounds();

    std::cout << "Map bounds:" << std::endl;
    std::cout << "X: [" << x_bounds.first << ", " << x_bounds.second << "] meters" << std::endl;
    std::cout << "Y: [" << y_bounds.first << ", " << y_bounds.second << "] meters" << std::endl;
    std::cout << "Initialized voxel map with " << voxel_map_.size() << " voxels." << std::endl;
}

void MapProblem::finalize() {
    // Save results
    if (opts_.save_result) {
        result_.save_full_result();
    }

    // Visualize results
    if (opts_.visualize_result) {
        result_.visualize_map();
    }
}

}   // namespace ba