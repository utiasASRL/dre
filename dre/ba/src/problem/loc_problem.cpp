#include <ba/problem/loc_problem.hpp>
#include <ba/map/voxel_map.hpp>
#include <ba/utils/io_utils.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <ba/scans/local_map_scan.hpp>
#include <ba/scans/scan.hpp>

namespace ba {

void LocProblem::validate_opts() {
    // Check that localization opts exist
    if (opts_.loc_opts.seq_id.empty()) {
        throw std::invalid_argument("LocProblem: seq_id must be specified in localization options");
    }
}

void LocProblem::init_seq_id() {
    seq_id_ = opts_.loc_opts.seq_id;
}

void LocProblem::get_scan_indeces() {
    // For localization load all scans
    fs::path all_img_dir = opts_.meas_path / seq_id_ / opts_.loc_opts.frame_processing_opts.input_type;
    int count = -1;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        // Only consider files ending with .png
        if (entry.path().extension() != ".png") {
            continue;
        }
        if (entry.is_regular_file()) {
            count++;
            if (count < opts_.loc_opts.start_frame) {
                continue;
            }
            scan_indices_.push_back(count);
            if (opts_.loc_opts.end_frame >= 0 && count >= opts_.loc_opts.end_frame) {
                break;
            }
        }
    }
    // Sort scan indices
    std::sort(scan_indices_.begin(), scan_indices_.end());
}
    
void LocProblem::init_scans_and_map() {
    // Load in map from estimates
    load_map_from_estimate();
    // Load in all scan poses to localize
    load_scans();
}

void LocProblem::load_map_from_estimate() {
    std::string map_seq_id = opts_.loc_opts.map_seq_id;
    // Load in voxel map from estimates
    std::string map_location = opts_.loc_opts.map_location.string() + "/voxel_map.bin";
    voxel_map_.load_from_file(map_location);

    std::cout << "Voxel map loaded from: " << map_location << std::endl;

    // Load groundtruth map poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;

    // seq id for map should be encoded in the map object
    ba::load_groundtruth_poses_and_times(opts_.data_path / map_seq_id, all_gt_poses, all_gt_times);

    // Store gt poses corresponding to map poses
    fs::path all_map_img_dir = opts_.meas_path / map_seq_id / opts_.loc_opts.frame_processing_opts.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_map_img_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    const std::vector<int> pose_ids = voxel_map_.pose_ids();
    const std::vector<lgmath::se2::Transformation> map_poses = voxel_map_.poses();

    if (pose_ids.empty()) {
        throw std::runtime_error("voxel_map pose_ids is empty.");
    }

    const int max_pose_id = pose_ids.back();
    if (max_pose_id < 0 || static_cast<std::size_t>(max_pose_id) >= files.size()) {
        throw std::runtime_error("Pose id out of range for image files. Are you sure the map was built from the same sequence? max_pose_id: " + std::to_string(max_pose_id) + ", num_files: " + std::to_string(files.size()) + ", map_seq_id: " + map_seq_id);
    }

    for (size_t i = 0; i < pose_ids.size(); i++) {
        int pose_id = pose_ids[i];
        // Get timestamp from pose_id by looking at the filename for boreas-2024-12-03-12-54
        fs::path img_path = files[pose_id];
        int64_t timestamp = std::stoll(img_path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        gt_map_poses_.push_back(T_gt_abs);
    }

    std::cout << gt_map_poses_.size() << " groundtruth map poses loaded." << std::endl;
    std::cout << voxel_map_.poses().size() << " estimated map poses loaded." << std::endl;
}

void LocProblem::load_scans() {
    std::cout << "Loading scans for localization..." << std::endl;

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts_.data_path / seq_id_, all_gt_poses, all_gt_times);

    // Load DRO poses
    std::vector<lgmath::se3::Transformation> all_dro_poses;
    std::vector<double> all_dro_times;
    ba::load_dro_poses_and_times(opts_.meas_path / seq_id_, all_dro_poses, all_dro_times);

    // Load pogo poses
    std::vector<lgmath::se3::Transformation> all_pogo_poses;
    std::vector<double> all_pogo_times;
    ba::load_pogo_poses_and_times(opts_.meas_path / seq_id_, all_pogo_poses, all_pogo_times);

    // Loop through indeces
    for (size_t i=0; i < scan_indices_.size(); i++) {
        int idx = scan_indices_[i];

        // Load in all relevant poses
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamps_[i]/1e6);
        gt_poses_.push_back(T_gt_abs);
        lgmath::se3::Transformation T_dro_abs = ba::get_interpolated_pose(all_dro_poses, all_dro_times, timestamps_[i]/1e6);
        dro_poses_.push_back(T_dro_abs);

        // Load in image paths
        fs::path img_path = img_paths_[i];
        std::optional<fs::path> cumul_img_path = std::nullopt;
        if (opts_.ba_opts.optimization_opts.use_cumul_thresh) {
            if (cumul_paths_.empty()) {
                throw std::runtime_error("Cumulative image paths are empty but use_cumul_thresh is true.");
            }
            cumul_img_path = cumul_paths_[i];
        }

        // Just initialize all poses with identity matrix. We'll handle initialization later.
        lgmath::se3::Transformation T_init;


        auto scan = std::make_shared<ba::LocalMapScan>(timestamps_[i],
                                                        idx,
                                                        opts_.loc_opts.frame_processing_opts.local_map_res,
                                                        opts_.loc_opts.optimization_opts,
                                                        T_init,
                                                        T_init,
                                                        img_path,
                                                        cumul_img_path);
        scan_manager_.add_scan(scan);
    }

    std::cout << "Scan manager has " << scan_manager_.num_scans() << " scans." << std::endl;
}

void LocProblem::save_loc_results(const fs::path &output_path) {
    const fs::path resolved_output_path = output_path.empty() ? opts_.output_path : output_path;
    fs::path loc_results_path = resolved_output_path / "loc_results.csv";
    // Remove existing file if it exists
    if (fs::exists(loc_results_path)) {
        fs::remove(loc_results_path);
    }
    // Save all data
    std::ofstream ofs(loc_results_path, std::ios::out);
    ofs << "map_id,scan_id,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw,std_x,std_y,std_yaw\n";
    for (const auto& entry : loc_results_) {
        ofs << entry.map_id << "," << entry.scan_id << ","
            << entry.est_x << "," << entry.est_y << "," << entry.est_yaw << ","
            << entry.gt_x << "," << entry.gt_y << "," << entry.gt_yaw << ","
            << entry.std_x << "," << entry.std_y << "," << entry.std_yaw << "\n";
    }
    ofs.close();
}

void LocProblem::visualize_loc_results() {
    std::string cmd;
    fs::path temp_dir;
    bool use_temp_dir = !opts_.save_result;
    if (use_temp_dir) {
        temp_dir = fs::temp_directory_path() / "dr_ba_temp_visualization";
        fs::remove_all(temp_dir);
        fs::create_directories(temp_dir);
        std::cout << "Output directory not set. Using temporary directory: " << temp_dir.string() << std::endl;

        // Save all map results to temporary directory
        save_loc_results(temp_dir);
        // Also copy over voxel_map.bin to temp directory
        fs::path voxel_map_src = opts_.loc_opts.map_location / "voxel_map.bin";
        fs::path voxel_map_dst = temp_dir / "voxel_map.bin";
        fs::copy_file(voxel_map_src, voxel_map_dst);
        cmd = "python3 /home/asrl/ros2_ws/dr_ba/ba_py/visualize_loc_result.py --loc_path " + temp_dir.string();
    } else {
        cmd = "python3 /home/asrl/ros2_ws/dr_ba/ba_py/visualize_loc_result.py --loc_path " + opts_.output_path.string();
    }

    if (opts_.visualize_result) {
        cmd += " --show";
    }

    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error("Error executing command: " + cmd);

    // Clean up temporary directory
    if (use_temp_dir) {
        std::cout << "Removing temporary directory: " << temp_dir.string() << std::endl;
        fs::remove_all(temp_dir);
    }
}

void LocProblem::finalize() {
    if (opts_.save_result) {
        save_loc_results();
    }
    if (opts_.visualize_result || opts_.save_result) {
        visualize_loc_results();
    }
}


}  // namespace ba