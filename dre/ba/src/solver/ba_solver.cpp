#include <ba/solver/ba_solver.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <unsupported/Eigen/SparseExtra>
#include <ba/utils/io_utils.hpp>

namespace ba {

void DrBASolver::tile_problem() {
    std::cout << "Tiling problem..." << std::endl;
    tiles_.clear();

    // Get all voxel keys
    auto all_voxel_keys = voxel_map_.get_sorted_keys_downsampled();

    // Create tiles
    double tile_size = opts_.tile_size;
    if (tile_size <= 0.0) {
        // No tiling
        Tile single_tile;
        single_tile.voxel_indices = all_voxel_keys;
        // All scans in this tile
        for (int scan_id : scan_manager_.get_all_scan_ids()) {
            single_tile.scan_ids.push_back(scan_id);
        }
        tiles_.push_back(single_tile);
    } else {
        // Tiling
        // Determine bounds of the voxel map
        std::pair<double, double> x_bounds = voxel_map_.x_bounds();
        std::pair<double, double> y_bounds = voxel_map_.y_bounds();
        double min_x = x_bounds.first - tile_size; // add margin
        double max_x = x_bounds.second + tile_size;
        double min_y = y_bounds.first - tile_size;
        double max_y = y_bounds.second + tile_size;

        double max_dist = 1.2 * full_opts_.ba_opts.frame_processing_opts.max_dist; // consider scans within this distance
        // Given that max_dist should be >10m and we expect an initial guess within a few meters,
        // we should effectively never need to re-tile the problem after initial tiling
        int num_tiles_x = static_cast<int>(std::ceil((max_x - min_x) / tile_size));
        int num_tiles_y = static_cast<int>(std::ceil((max_y - min_y) / tile_size));
        int max_num_scans_in_tile = 0;
        int max_num_voxels_in_tile = 0;
        for (int tx = 0; tx < num_tiles_x; ++tx) {
            for (int ty = 0; ty < num_tiles_y; ++ty) {
                double tile_min_x = min_x + tx * tile_size;
                double tile_max_x = std::min(tile_min_x + tile_size, max_x);
                double tile_min_y = min_y + ty * tile_size;
                double tile_max_y = std::min(tile_min_y + tile_size, max_y);

                Tile tile;
                // Collect voxel indices in this tile
                for (const auto& voxel_idx : all_voxel_keys) {
                    auto coord = voxel_map_.voxel_to_coord(voxel_idx);
                    if (coord.first >= tile_min_x && coord.first < tile_max_x &&
                        coord.second >= tile_min_y && coord.second < tile_max_y) {
                        tile.voxel_indices.push_back(voxel_idx);
                    }
                }

                // Collect scans that overlap with this tile
                for (int scan_id : scan_manager_.get_all_scan_ids()) {
                    auto scan = scan_manager_.get_scan(scan_id);
                    auto pose = scan->pose2d();
                    auto scan_x = pose.r_ab_inb()(0);
                    auto scan_y = pose.r_ab_inb()(1);
                    if (scan_x + max_dist >= tile_min_x && scan_x - max_dist < tile_max_x &&
                        scan_y + max_dist >= tile_min_y && scan_y - max_dist < tile_max_y) {
                        tile.scan_ids.push_back(scan_id);
                    }
                }
                if (!tile.voxel_indices.empty() && !tile.scan_ids.empty()) {
                    tiles_.push_back(tile);
                    max_num_scans_in_tile = std::max(max_num_scans_in_tile, static_cast<int>(tile.scan_ids.size()));
                    max_num_voxels_in_tile = std::max(max_num_voxels_in_tile, static_cast<int>(tile.voxel_indices.size()));
                }
            }
        }
        std::cout << "Tiling complete: " << tiles_.size() << " tiles created." << std::endl;
        std::cout << "Max scans in a tile: " << max_num_scans_in_tile << std::endl;
        std::cout << "Max voxels in a tile: " << max_num_voxels_in_tile << std::endl;
    }

}

void DrBASolver::construct_problem(double downsample_factor) {
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }
    // Tile problem if not already done
    if (tiles_.empty()) {
        tile_problem();
    }

    // Set up timer
    auto start_time = std::chrono::high_resolution_clock::now();
    double rel_pose_prior_time = 0.0;
    double avg_per_voxel_time = 0.0;

    // Check that tiling has been done
    if (tiles_.empty()) {
        throw std::runtime_error("Error: DrBASolver::construct_problem called before tiling the problem.");
    }

    // Load in constants
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    int states_size = scan_manager_.num_active_scans() * 3; // SE2 poses with first pose fixed

    // Reset cost
    cost_ = 0.0;

    // Downsample desired voxels
    std::vector<ba::VoxelMap::Index> voxel_keys = voxel_map_.get_sorted_keys_downsampled(downsample_factor);
    int voxels_size = static_cast<int>(voxel_keys.size());

    // Initialize matrices
    lhs_.setZero(states_size, states_size);
    rhs_.setZero(states_size);

    // Load in relative SE2 pose priors
    auto rel_pose_start_time = std::chrono::high_resolution_clock::now();
    if (opts_.use_pose_prior) {
        for (const auto& prior : pose_priors_) {
            int scan_id_a = prior.first.first;
            int scan_id_b = prior.first.second;

            // Make sure both scans are in the current problem
            if (scan_manager_.has_scan(scan_id_a) == false ||
                scan_manager_.has_scan(scan_id_b) == false) {
                continue;
            }

            lgmath::se2::Transformation T_prior = prior.second.toSE2();
            // T_prior is the transform from scan A to scan B
            // Load in poses
            lgmath::se2::Transformation T_a = scan_manager_.get_scan(scan_id_a)->pose2d();
            lgmath::se2::Transformation T_b = scan_manager_.get_scan(scan_id_b)->pose2d();
            // Compute error
            lgmath::se2::Transformation T_err = T_prior * (T_b.inverse() * T_a).inverse();
            Eigen::Matrix<double, 3, 1> err_vec = T_err.vec();
            // Compute Jacobians
            Eigen::Matrix<double, 3, 3> Ad_T_b_inv = T_b.inverse().adjoint();
            Eigen::Matrix<double, 3, 3> d_e_d_Ta = Ad_T_b_inv;
            Eigen::Matrix<double, 3, 3> d_e_d_Tb = - Ad_T_b_inv;
            // Weight by prior covariance
            Eigen::Vector3d prior_std_diag;
            prior_std_diag << opts_.rel_pose_prior_translation_std, opts_.rel_pose_prior_translation_std,
                                opts_.rel_pose_prior_rotation_std * M_PI / 180.0;
            int num_steps = scan_id_b - scan_id_a;
            prior_std_diag *= static_cast<double>(num_steps); // Scale covariance with number of steps
            Eigen::Matrix3d prior_cov_sqrt_inv = prior_std_diag.cwiseInverse().asDiagonal();
            // Weight Jacobians and error
            d_e_d_Ta = prior_cov_sqrt_inv * d_e_d_Ta;
            d_e_d_Tb = prior_cov_sqrt_inv * d_e_d_Tb;
            Eigen::Matrix<double, 3, 1> err_vec_weighted = prior_cov_sqrt_inv * err_vec;
            // Assemble into H and J
            // Get scan index from scan id
            int state_idx_a = -1;
            int state_idx_b = -1;
            for (std::size_t idx = 0; idx < scan_id_list.size(); ++idx) {
                if (scan_id_list[idx] == scan_id_a) {
                    state_idx_a = (idx - 1) * 3;
                }
                if (scan_id_list[idx] == scan_id_b) {
                    state_idx_b = (idx - 1) * 3;
                }
            }

            if (state_idx_a < 0 && state_idx_b < 0) {
                // Something is wrong!
                throw std::runtime_error("Both scans in prior are fixed poses.");
            }

            if (state_idx_a < 0) {
                // First pose is fixed, only assemble for b
                lhs_.block<3,3>(state_idx_b, state_idx_b) += d_e_d_Tb.transpose() * d_e_d_Tb;
                rhs_.segment<3>(state_idx_b) += d_e_d_Tb.transpose() * err_vec_weighted;
            } else {
                // a, a contribution
                lhs_.block<3,3>(state_idx_a, state_idx_a) += d_e_d_Ta.transpose() * d_e_d_Ta;
                // b, b contribution
                lhs_.block<3,3>(state_idx_b, state_idx_b) += d_e_d_Tb.transpose() * d_e_d_Tb;
                // a, b contribution
                lhs_.block<3,3>(state_idx_a, state_idx_b) += d_e_d_Ta.transpose() * d_e_d_Tb;
                // b, a contribution
                lhs_.block<3,3>(state_idx_b, state_idx_a) += d_e_d_Tb.transpose() * d_e_d_Ta;
                // rhs
                rhs_.segment<3>(state_idx_a) += d_e_d_Ta.transpose() * err_vec_weighted;
                rhs_.segment<3>(state_idx_b) += d_e_d_Tb.transpose() * err_vec_weighted;
            }

            cost_ += 0.5 * err_vec.norm();
        }
    }
    rel_pose_prior_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - rel_pose_start_time).count();

    // Loop through each tile
    for (const auto& tile : tiles_) {
        // Load in data for scans in this tile
        // Note this will fail if max_loaded_scans is too small
        scan_manager_.load_data(tile.scan_ids);

        // Pre-compute active state indices for this tile (excluding fixed first pose)
        std::vector<int> active_state_indices;
        for (int scan_id : tile.scan_ids) {
            int scan_idx = scan_manager_.id_to_idx(scan_id);
            if (scan_idx > 0) {  // Skip first fixed pose
                active_state_indices.push_back((scan_idx - 1) * 3);
            }
        }
        int num_active_states = active_state_indices.size() * 3;  // Each pose has 3 DOF

        std::unordered_map<int,int> global_to_local;
        global_to_local.reserve(active_state_indices.size());

        
        for (size_t i = 0; i < active_state_indices.size(); ++i)
            global_to_local[active_state_indices[i]] = i * 3;

        // Pre-filter voxels in this tile with downsampled keys (both are sorted)
        std::vector<decltype(tile.voxel_indices)::value_type> filtered_voxels;
        filtered_voxels.reserve(std::min(tile.voxel_indices.size(), voxel_keys.size()));
        std::set_intersection(
            tile.voxel_indices.begin(), tile.voxel_indices.end(),
            voxel_keys.begin(), voxel_keys.end(),
            std::back_inserter(filtered_voxels)
        );
        const auto start_voxel_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel
{
        Eigen::MatrixXd lhs_local = Eigen::MatrixXd::Zero(lhs_.rows(), lhs_.cols());
        Eigen::VectorXd rhs_local = Eigen::VectorXd::Zero(rhs_.size());
        double cost_local = 0.0;

        // Loop through all voxels in this tile
        #pragma omp for schedule(static)
        for (size_t v = 0; v < filtered_voxels.size(); ++v) {
            const auto& voxel_idx = filtered_voxels[v];

            double voxel_x = static_cast<double>(voxel_idx.first) * voxel_map_.res();
            double voxel_y = static_cast<double>(voxel_idx.second) * voxel_map_.res();
            
            // Use local Jacobian with only active states
            int max_scans = tile.scan_ids.size();
            Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(max_scans, num_active_states);
            Eigen::VectorXd B = Eigen::VectorXd::Zero(max_scans);
            Eigen::VectorXd weighted_ones = Eigen::VectorXd::Zero(max_scans);
            int scan_count = 0;

            // Loop through all scans that cover this tile
            for (const int scan_id : tile.scan_ids) {                
                int scan_idx = scan_manager_.id_to_idx(scan_id);
                auto scan = scan_manager_.get_scan(scan_id);

                // Interpolate intensity and Jacobian
                std::optional<Scan::Measurement> interp_meas = scan->interpolate(voxel_x, voxel_y);
                // If scan is outside coverage, no intensity will be provided
                if (!interp_meas.has_value()) {
                    continue;
                }

                // Weight everything by square root of measurement covariance
                double I_meas = interp_meas->intensity;
                double meas_cov = interp_meas->covariance;
                Eigen::Matrix<double, 1, 3> d_beta_d_T = interp_meas->jacobian / std::sqrt(meas_cov);

                // Assemble into local Jacobian
                B(scan_count) = I_meas / std::sqrt(meas_cov);
                weighted_ones(scan_count) = 1.0 / std::sqrt(meas_cov);
                if (scan_idx > 0) {
                    // Find position in local Jacobian
                    auto it = global_to_local.find((scan_idx - 1) * 3);
                    if (it != global_to_local.end())
                        J_local.block<1,3>(scan_count, it->second) = d_beta_d_T;
                }

                scan_count++;
            }
            
            if (scan_count == 0) {
                // No scans cover this voxel
                continue;
            }
            
            // Trim to actual scan count
            J_local.conservativeResize(scan_count, num_active_states);
            B.conservativeResize(scan_count);
            weighted_ones.conservativeResize(scan_count);

            // Compute projection matrix
            Eigen::MatrixXd P = (weighted_ones * weighted_ones.transpose()) / (weighted_ones.squaredNorm()) - Eigen::MatrixXd::Identity(scan_count, scan_count);

            // Compute residual
            Eigen::MatrixXd PtP = P.transpose() * P;

            // Compute local contributions
            Eigen::MatrixXd H_local = J_local.transpose() * PtP * J_local;
            Eigen::VectorXd g_local = J_local.transpose() * PtP * B;

            // Scatter to global lhs_ and rhs_
            for (size_t i = 0; i < active_state_indices.size(); ++i) {
                int gi = active_state_indices[i];

                // diagonal
                lhs_local.block<3,3>(gi, gi) += H_local.block<3,3>(i*3, i*3);

                for (size_t j = i + 1; j < active_state_indices.size(); ++j) {
                    int gj = active_state_indices[j];
                    auto Hij = H_local.block<3,3>(i*3, j*3);

                    lhs_local.block<3,3>(gi, gj) += Hij;
                }

                rhs_local.segment<3>(gi) += g_local.segment<3>(i*3);
            }

            cost_local += 0.5 * (P * B).squaredNorm();
        }
        #pragma omp critical
        {
            lhs_ += lhs_local;
            rhs_ += rhs_local;
            cost_ += cost_local;
        }
}

        const auto end_voxel_time = std::chrono::high_resolution_clock::now();
        avg_per_voxel_time += std::chrono::duration<double>(end_voxel_time - start_voxel_time).count();
    }

    // Finalize timing
    auto end_time = std::chrono::high_resolution_clock::now();
    avg_per_voxel_time = avg_per_voxel_time / static_cast<double>(voxels_size);

    if (full_opts_.ba_opts.save_H && save_results_) {
        std::cout << "number non-zeros in lhs: " << (lhs_.array() != 0.0).count() << std::endl;

        std::string H_save_path = full_opts_.output_path / "H.bin";
        saveBinary(lhs_, H_save_path);
    }
}

bool DrBASolver::solve() {
    // Small regularization for numerical stability
    lhs_ += 1e-8 * Eigen::MatrixXd::Identity(lhs_.rows(), lhs_.cols());

    // Solve
    del_x_ = - alpha_ * lhs_.selfadjointView<Eigen::Upper>().ldlt().solve(rhs_);

    std::cout << "alpha: " << alpha_
            << " |delta|: " << del_x_.norm()
            << " cost: " << cost_ << std::endl;

    return true;
}

void DrBASolver::update_poses() {
    // Update poses
    if (del_x_.size() == 0) {
        throw std::runtime_error("No pose updates available. Have you run solve()?");
    }
    if (del_x_.size() != (scan_manager_.num_scans() - 1) * 3) {
        throw std::runtime_error("Size of pose updates does not match number of scans.");
    }
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    std::vector<lgmath::se2::Transformation> updated_poses;
    // Add first fixed pose to updated poses
    updated_poses.push_back(scan_manager_.get_scan(scan_id_list[0])->pose2d());
    for (int scan_idx = 1; scan_idx < scan_manager_.num_scans(); scan_idx++) {
        // Extract delta for this scan
        int state_idx = (scan_idx - 1) * 3;
        Eigen::Matrix<double, 3, 1> delta_xi = del_x_.segment<3>(state_idx);
        // Load in scan and update pose
        auto scan = scan_manager_.get_scan(scan_id_list[scan_idx]);
        scan->update_pose(delta_xi);
        updated_poses.push_back(scan->pose2d());
    }
    // Update map poses
    voxel_map_.set_poses(scan_id_list, updated_poses);
}

void DrBASolver::update_map() {
    std::cout << "Updating map..." << std::endl;
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }
    // Tile problem if not already done
    if (tiles_.empty()) {
        tile_problem();
    }
    // Zero out voxel map
    voxel_map_.zero_out();

    std::cout << "Looping over tiles..." << std::endl;
    // Print number of threads
    std::cout << "Using " << omp_get_max_threads() << " threads." << std::endl;

    // Loop through each tile
    for (const auto& tile : tiles_) {
        // Load in data for scans in this tile
        // Note this will fail if max_loaded_scans is too small
        scan_manager_.load_data(tile.scan_ids);

        // Pre-compute active state indices for this tile (excluding fixed first pose)
        std::vector<int> active_state_indices;
        for (int scan_id : tile.scan_ids) {
            int scan_idx = scan_manager_.id_to_idx(scan_id);
            if (scan_idx > 0) {  // Skip first fixed pose
                active_state_indices.push_back((scan_idx - 1) * 3);
            }
        }

        // Loop through all voxels in this tile
        #pragma omp parallel for schedule(static)
        for (size_t v = 0; v < tile.voxel_indices.size(); ++v) {
            const auto& voxel_idx = tile.voxel_indices[v];
            double voxel_x = static_cast<double>(voxel_idx.first) * voxel_map_.res();
            double voxel_y = static_cast<double>(voxel_idx.second) * voxel_map_.res();
            
            // Use local Jacobian with only active states
            int max_scans = tile.scan_ids.size();
            Eigen::VectorXd B = Eigen::VectorXd::Zero(max_scans);
            Eigen::VectorXd weighted_ones = Eigen::VectorXd::Zero(max_scans);
            int scan_count = 0;

            // Loop through all scans that cover this tile
            for (const int scan_id : tile.scan_ids) {          
                auto scan = scan_manager_.get_scan(scan_id);      
                // Interpolate intensity and Jacobian
                std::optional<Scan::Measurement> interp_meas = scan->interpolate(voxel_x, voxel_y);
                // If scan is outside coverage, no intensity will be provided
                if (!interp_meas.has_value()) {
                    continue;
                }
                double meas_cov = interp_meas->covariance;
                // Assemble into local Jacobian
                B(scan_count) = interp_meas->intensity / std::sqrt(meas_cov);
                weighted_ones(scan_count) = 1.0 / std::sqrt(meas_cov);
                scan_count++;
            }
            
            if (scan_count == 0) {
                // No scans cover this voxel
                continue;
            }
            
            // Trim to actual scan count
            B.conservativeResize(scan_count);
            weighted_ones.conservativeResize(scan_count);

            // Update voxel intensity
            double new_intensity = (weighted_ones.transpose() * B)(0,0) / (weighted_ones.transpose() * weighted_ones)(0,0);
            voxel_map_.at(voxel_idx) = new_intensity;
        }
    }

    // Unload all scans
    scan_manager_.unload_all_data();
}

void DrBASolver::optimize() {
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }

    // Initialize timer stuff
    double avg_construct_time = 0.0;
    double avg_solve_time = 0.0;
    double avg_update_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;

    // Tile the problem
    tile_problem();
    end =  std::chrono::high_resolution_clock::now();
    std::cout << "Number of tiles: " << tiles_.size() << std::endl;
    std::cout << "Tiling time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    // Compute initial RMSE
    result_.add_rmse(scan_manager_.compute_pose_rmse());
    result_.add_ate(scan_manager_.compute_ate());
    result_.add_epe(scan_manager_.compute_epe());
    std::cout << "Initial Pose Errors (ATE, EPE, RMSE: x, y, yaw): " << scan_manager_.compute_ate() << " | " << scan_manager_.compute_epe() << " | " << scan_manager_.compute_pose_rmse().transpose() << std::endl;
    double downsample_factor = 1.0;
    int num_cost_rises = 0;

    for (int iter = 0; iter < opts_.max_iterations; iter++) {
        std::cout << "Iteration " << iter << " / " << opts_.max_iterations << std::endl;
        downsample_factor = (iter < opts_.num_coarse_iterations) ? opts_.coarse_downsample : opts_.refine_downsample;

        // Construct problem
        start = std::chrono::high_resolution_clock::now();
        construct_problem(downsample_factor);
        end =  std::chrono::high_resolution_clock::now();
        avg_construct_time += std::chrono::duration<double>(end - start).count();

        if (iter > 0 && iter > opts_.num_coarse_iterations && cost_ > prev_cost_) {
            if (num_cost_rises >= opts_.max_cost_increases) {
                std::cout << "Stopping optimization due to repeated cost increases." << std::endl;
                cost_ = prev_cost_; // Revert to previous cost since we failed to find a better solve
                break;
            }
            num_cost_rises++;

            // Undo previous update
            del_x_ = -del_x_;
            update_poses();

            // Recompute full del_x
            del_x_ /= -alpha_;

            // Reduce alpha
            alpha_ *= std::pow(0.5, num_cost_rises);

            // Compute new step
            del_x_ *= alpha_;

            // Try a new update!
            update_poses();

            // Retry with new step
            iter--;
            std::cout << "Cost increased. Reducing step size to alpha: " << alpha_ << " and retrying..." << std::endl;
            continue;
        } else {
            // Solve normally
            num_cost_rises = 0;
            // Solve problem, skip updating if solve failed
            start = std::chrono::high_resolution_clock::now();
            bool success = solve();
            end =  std::chrono::high_resolution_clock::now();
            avg_solve_time += std::chrono::duration<double>(end - start).count();
            if (!success) continue;
            // Save cost
            result_.add_cost(cost_);
        }

        // Update poses
        start = std::chrono::high_resolution_clock::now();
        update_poses();
        end =  std::chrono::high_resolution_clock::now();
        avg_update_time += std::chrono::duration<double>(end - start).count();

        // std::cout << "Cost: " << cost_ << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Pose Errors (ATE, EPE, RMSE: x, y, yaw): " << scan_manager_.compute_ate() << " | " << scan_manager_.compute_epe() << " | " << scan_manager_.compute_pose_rmse().transpose() << std::endl;
        std::cout << "Average Times (s): Construct Problem: " << (avg_construct_time / static_cast<double>(iter + 1))
                  << " | Solve: " << (avg_solve_time / static_cast<double>(iter + 1))
                  << " | Update Poses: " << (avg_update_time / static_cast<double>(iter + 1)) << std::endl;
        result_.add_rmse(scan_manager_.compute_pose_rmse());
        result_.add_ate(scan_manager_.compute_ate());
        result_.add_epe(scan_manager_.compute_epe());
        if (save_results_)
            result_.save_full_result();

        if (iter != 0 && iter > opts_.num_coarse_iterations && (del_x_.norm() < opts_.convergence_tol || std::abs(prev_cost_ - cost_) < opts_.convergence_tol)) {
            std::cout << "Converged from: " << ((del_x_.norm() < opts_.convergence_tol ) ? "small pose update." : "small cost change.") << std::endl;
            break;
        }
        // Only overwrite prev_cost if we are not retrying due to cost increase
        if (cost_ < prev_cost_) {
            prev_cost_ = cost_;
        }
    }

    // Construct problem for final cost
    // construct_problem(downsample_factor);
    result_.add_cost(cost_);
    std::cout << "Final Cost: " << cost_ << std::endl;

    // Update map
    update_map();   
}


} // namespace ba