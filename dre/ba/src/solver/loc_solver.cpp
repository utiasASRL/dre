#include <ba/solver/loc_solver.hpp>
#include <ba/problem/loc_problem.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <lgmath/se2/Operations.hpp>

namespace ba {

void LocSolver::construct_problem(const std::shared_ptr<Scan>& scan) {
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }
    // Reset cost
    cost_ = 0.0;

    // Initialize matrices
    lhs_.setZero(3, 3);
    rhs_.setZero(3);

    // Add prior
    // if (opts_.use_odometry_prior) {
    //     // Form error between pose prior and current estimate
    //     lgmath::se3::Transformation T_prior_err = scan->pose().inverse() * curr_pose_;
    //     Eigen::Matrix<double, 3, 1> prior_err = T_prior_err.toSE2().vec();
    //     Eigen::Matrix<double, 3, 3> prior_info = curr_cov_.inverse();
    //     // lhs_ += prior_info;
    //     // rhs_ += prior_info * prior_err;

    //     Eigen::Matrix3d dro_process_noise = Eigen::Matrix3d::Zero();
    //     dro_process_noise(0, 0) = std::pow(opts_.odom_translation_std, 2);
    //     dro_process_noise(1, 1) = std::pow(opts_.odom_translation_std, 2);
    //     dro_process_noise(2, 2) = std::pow(opts_.odom_rotation_std * M_PI / 180.0, 2); // convert to radians

    //     lhs_ += dro_process_noise.inverse();
    //     rhs_ += dro_process_noise.inverse() * prior_err;
    // }

    // Loop through all voxels in the scan's coverage
    int num_voxels_used = 0;
    std::vector<double> voxel_errors;
    // Get voxels in range of initial pose
    double max_dist = full_opts_.loc_opts.frame_processing_opts.max_dist;
    std::vector<ba::VoxelMap::Index> voxel_keys = voxel_map_.get_voxels_in_range(scan->pose2d(), max_dist);
    for (const auto& voxel_idx : voxel_keys) {
        double voxel_x = static_cast<double>(voxel_idx.first) * voxel_map_.res();
        double voxel_y = static_cast<double>(voxel_idx.second) * voxel_map_.res();
        double vox_intensity = voxel_map_.at(voxel_idx);

        // Interpolate intensity and Jacobian
        std::optional<Scan::Measurement> interp_meas = scan->interpolate(voxel_x, voxel_y);
        // If scan is outside coverage, no intensity will be provided
        if (!interp_meas.has_value()) {
            continue;
        }

        // Weight everything by square root of measurement covariance
        double I_meas = interp_meas->intensity;
        double meas_cov = interp_meas->covariance;
        double err_weight = 1.0 / meas_cov;

        // Compute unweighted error
        double err = (vox_intensity - I_meas);
        double err_weight_sqrt = std::sqrt(err_weight);

        voxel_errors.push_back(std::abs(err));

        Eigen::Matrix<double, 1, 3> d_beta_d_T = - interp_meas->jacobian * err_weight_sqrt;
        err *= err_weight_sqrt;

        lhs_ += d_beta_d_T.transpose() * d_beta_d_T;
        rhs_ += d_beta_d_T.transpose() * err;

        // Compute cost
        cost_ += 0.5 * std::pow(err, 2) ;
        num_voxels_used++;
    }
    
    if (num_voxels_used == 0) {
        throw std::runtime_error("Error: No voxels used in localization optimization! Check if your map and loc entries overlap?");
    }
}

void LocSolver::compute_errors(LocProblem& loc_problem, const std::shared_ptr<Scan>& scan, int i) {
    // First, find nearest map pose to the scan's estimated pose
    double min_dist = std::numeric_limits<double>::max();
    int best_map_idx = -1;
    lgmath::se3::Transformation nearest_map_gt_pose;
    lgmath::se3::Transformation nearest_map_est_pose;
    for (size_t j = 0; j < loc_problem.gt_map_poses().size(); j++) {
        lgmath::se3::Transformation map_est_pose = loc_problem.voxel_map().poses().at(j).toSE3();
        double dist = (map_est_pose.inverse() * scan->pose()).r_ab_inb().norm();

        // Prefer selecting nodes with similar orientation
        if (std::abs((map_est_pose.vec()(5) - scan->pose().vec()(5))) > M_PI / 2.0) {
            dist += 1000.0;
        }

        if (dist < min_dist) {
            min_dist = dist;
            nearest_map_gt_pose = loc_problem.gt_map_poses().at(j);
            // Also get the estimated map pose
            nearest_map_est_pose = map_est_pose;
            best_map_idx = static_cast<int>(j);
        }
    }
    if (best_map_idx == -1) {
        throw std::runtime_error("Error: Could not find nearest map pose! Check if your map and loc entries overlap?");
    }
    std::cout << "Nearest map pose index: " << best_map_idx << ", distance: " << min_dist << " m." << std::endl;
    // Compute estimated pose within local map
    lgmath::se3::Transformation loc_est_pose = nearest_map_est_pose.inverse() * curr_pose_;
    scan->set_pose(loc_est_pose);

    // Compute gt pose within local map
    lgmath::se3::Transformation loc_gt_pose = nearest_map_gt_pose.inverse() * loc_problem.gt_poses().at(i);
    // Discard 3D info from the relative transform
    loc_gt_pose = loc_gt_pose.toSE2().toSE3();
    scan->set_gt_pose(loc_gt_pose);

    // Store result
    curr_cov_ = lhs_.inverse();
    Eigen::Matrix<double, 3, 1> loc_est_pose_xy = scan->pose().r_ab_inb();
    Eigen::Matrix<double, 3, 1> loc_gt_pose_xy = scan->gt_pose().r_ab_inb();
    double loc_est_yaw = scan->pose().vec()(5);
    double loc_gt_yaw = scan->gt_pose().vec()(5);
    LocProblem::LocResultEntry result_entry;
    result_entry.map_id = loc_problem.voxel_map().pose_ids().at(best_map_idx);
    result_entry.scan_id = scan->id();
    result_entry.est_x = loc_est_pose_xy(0);
    result_entry.est_y = loc_est_pose_xy(1);
    result_entry.est_yaw = loc_est_yaw;
    result_entry.gt_x = loc_gt_pose_xy(0);
    result_entry.gt_y = loc_gt_pose_xy(1);
    result_entry.gt_yaw = loc_gt_yaw;
    result_entry.std_x = std::sqrt(curr_cov_(0,0));
    result_entry.std_y = std::sqrt(curr_cov_(1,1));
    result_entry.std_yaw = std::sqrt(curr_cov_(2,2));
    loc_problem.add_loc_result(result_entry);

    // Periodically save results to memory
    if (save_results_ && (i % 10 == 0 || i == num_scans_ - 1)) {
        loc_problem.save_loc_results();
    }

    // Print pose error for this scan
    Eigen::Matrix<double, 6, 1> pose_error = scan->pose_error();
    std::cout << "Final pose error (m, m, deg): " << pose_error(0) << ", " << pose_error(1) << ", " << pose_error(5) * 180.0 / M_PI << std::endl;
    avg_pose_error_(0) += pose_error(0) * pose_error(0);
    avg_pose_error_(1) += pose_error(1) * pose_error(1);
    avg_pose_error_(2) += pose_error(5) * pose_error(5);

    // If pose error is larger than max_dist, localization has failed and will not recover
    // This prevents wasted time and is not meant to be used in a live system since we use gt
    if (std::sqrt(pose_error(0) * pose_error(0) + pose_error(1) * pose_error(1)) > 15.0) {
        std::cerr << "Error: Localization has diverged! Pose error exceeded maximum map range." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void LocSolver::odometry_step(LocProblem& loc_problem, int i) {
    if (i == 0) {
        // Initialize pose using gt
        // Get the initial pose within the map frame
        lgmath::se3::Transformation first_map_gt_pose = loc_problem.gt_map_poses().at(0);
        lgmath::se3::Transformation first_map_est_pose = loc_problem.voxel_map().poses().at(0).toSE3();
        lgmath::se3::Transformation loc_init_gt_pose = loc_problem.gt_poses().at(0);
        curr_pose_ = first_map_est_pose * first_map_gt_pose.inverse() * loc_init_gt_pose;

        // Project to SE2 to get rid of any gt rounding in 3D dimensions
        curr_pose_ = curr_pose_.toSE2().toSE3();
        curr_cov_ = 0.001 * Eigen::Matrix3d::Identity();
        return;
    }

    // Propagate curr_pose_ using DRO estimates
    lgmath::se3::Transformation dro_rel_pose;
    lgmath::se3::Transformation curr_dro_pose = loc_problem.dro_poses().at(i-1);
    lgmath::se3::Transformation next_dro_pose = loc_problem.dro_poses().at(i);
    dro_rel_pose = curr_dro_pose.inverse() * next_dro_pose;
    Eigen::Vector3d dro_rel_pose_xi = dro_rel_pose.toSE2().vec();
    curr_pose_ = curr_pose_ * dro_rel_pose;

    // Compute Jacobians
    Eigen::Matrix3d dro_noise_jacobian = lgmath::se2::vec2jac(dro_rel_pose_xi);
    Eigen::Matrix3d dro_pose_jacobian = - lgmath::se2::tranAd(dro_rel_pose.toSE2().inverse().matrix());

    // Load in process noise
    Eigen::Matrix3d dro_process_noise = Eigen::Matrix3d::Zero();
    // dro_process_noise(0, 0) = std::pow(opts_.odom_translation_std, 2);
    // dro_process_noise(1, 1) = std::pow(opts_.odom_translation_std, 2);
    // dro_process_noise(2, 2) = std::pow(opts_.odom_rotation_std * M_PI / 180.0, 2); // convert to radians

    // Propagate covariance
    curr_cov_ = dro_pose_jacobian * curr_cov_ * dro_pose_jacobian.transpose() + dro_noise_jacobian * dro_process_noise * dro_noise_jacobian.transpose();

    // For debug, set current pose to groundtruth
    // curr_pose_ = nearest_map_est_pose * nearest_map_gt_pose.inverse() * loc_problem.gt_poses().at(i + 1);
    // curr_pose_ = curr_pose_.toSE2().toSE3();
}

void LocSolver::optimize() {
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }

    avg_pose_error_ = Eigen::Vector3d::Zero();
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    num_scans_ = static_cast<int>(scan_id_list.size());
    int max_id = scan_manager_.get_all_scan_ids().back();

    // Cast to LocProblem to access derived class methods
    auto& loc_problem = static_cast<LocProblem&>(problem_);
    double avg_runtime = 0.0;
    int64_t start_timestamp = scan_manager_.ref_timestamp();
    for (int i = 0; i < num_scans_; i++) {
        // Run odometry step to update pose for next iteration
        odometry_step(loc_problem, i);

        auto start_time = std::chrono::high_resolution_clock::now();
        int scan_id = scan_id_list.at(i);
        auto scan = scan_manager_.get_scan(scan_id);
        scan->set_pose(curr_pose_);
        std::cout << "----------------------------------------" << std::endl;
        int64_t scan_timestamp = scan->timestamp();
        double time_from_start = static_cast<double>(scan_timestamp - start_timestamp) / 1e6;

        std::cout << "Optimizing scan ID: " << scan_id << "/" << max_id
                  << " (timestamp: " << scan->timestamp() << ", " << time_from_start << " s from start)" << std::endl;
        scan->load_data();
        for (int iter = 0; iter < opts_.max_iterations; iter++) {
            // Construct problem
            construct_problem(scan);

            // Solve problem
            Eigen::Vector3d delta_xi = - alpha_ * lhs_.inverse() * rhs_;

            // Update poses
            scan->update_pose(delta_xi);

            if (iter != 0 && (delta_xi.norm() < opts_.convergence_tol || std::abs(prev_cost_ - cost_) < opts_.convergence_tol)) {
                std::cout << "Converged from: " << ((delta_xi.norm() < opts_.convergence_tol ) ? "small pose update." : "small cost change.") << std::endl;
                break;
            }

            prev_cost_ = cost_;
        }
        scan->unload_data();

        // Compute difference between prior and estimate
        lgmath::se3::Transformation pose_diff = scan->pose().inverse() * curr_pose_;
        double pos_diff = pose_diff.r_ab_inb().head<2>().norm();
        double yaw_diff = std::abs(pose_diff.vec()(5)) * 180.0 / M_PI;
        std::cout << "Position difference from prior: " << pos_diff << " m." << std::endl;
        std::cout << "Yaw difference from prior: " << yaw_diff << " deg." <<  std::endl;
        curr_pose_ = scan->pose();

        // Compute errors
        compute_errors(loc_problem, scan, i);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        avg_runtime += static_cast<double>(duration);
        std::cout << "Average scan runtime: " << avg_runtime / static_cast<double>(i + 1) << " ms." << std::endl;
    }
    
    
    std::cout << "Localization of " << num_scans_ << " scans took " << avg_runtime / static_cast<double>(num_scans_) << " ms on average." <<  std::endl;

    avg_pose_error_ /= static_cast<double>(num_scans_);
    avg_pose_error_ = avg_pose_error_.cwiseSqrt();
    avg_pose_error_(2) = avg_pose_error_(2) * 180.0 / M_PI;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "RMSE over all scans (m, m, deg):\n" << avg_pose_error_.transpose() << std::endl;
}


} // namespace ba