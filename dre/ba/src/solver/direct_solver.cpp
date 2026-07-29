#include <ba/solver/direct_solver.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <unsupported/Eigen/SparseExtra>
#include <ba/utils/io_utils.hpp>

namespace ba {

void DirectSolver::tile_problem() {
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

void DirectSolver::construct_problem(double downsample_factor) {
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }
    // Tile problem if not already done
    if (tiles_.empty()) {
        tile_problem();
    }

    std::cout << "Constructing problem..." << std::endl;

    // Set up timer
    auto start_time = std::chrono::high_resolution_clock::now();
    double rel_pose_prior_time = 0.0;
    double avg_per_voxel_time = 0.0;

    // Check that tiling has been done
    if (tiles_.empty()) {
        throw std::runtime_error("Error: DirectSolver::construct_problem called before tiling the problem.");
    }

    // Load in constants
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    int states_size = scan_manager_.num_active_scans() * 3; // SE2 poses with first pose fixed

    // Reset cost
    cost_ = 0.0;

    // Downsample desired voxels
    std::vector<ba::VoxelMap::Index> voxel_keys = voxel_map_.get_sorted_keys_downsampled(downsample_factor);
    int voxels_size = static_cast<int>(voxel_keys.size());

    // Keep track of exactly which voxel variables are in this linearized system.
    optimized_voxel_keys_ = voxel_keys;

    // Create voxel lookup for quick access to column indices in the optimization variable
    std::map<VoxelMap::Index, std::size_t> vox_to_global_pos;
    for (std::size_t i = 0; i < voxel_keys.size(); ++i) {
        vox_to_global_pos[voxel_keys[i]] = i;
    }

    // Initialize matrices
    lhs_sp_.resize(states_size + voxels_size, states_size + voxels_size);
    rhs_.setZero(states_size + voxels_size);

    // TODO: Add pose prior terms

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
        int num_active_voxels = filtered_voxels.size();

#pragma omp parallel
{
        std::vector<Eigen::Triplet<double>> local_triplets;
        Eigen::VectorXd rhs_local = Eigen::VectorXd::Zero(rhs_.size());
        double cost_local = 0.0;

        // Loop through all voxels in this tile
        #pragma omp for schedule(static)
        for (size_t v = 0; v < filtered_voxels.size(); ++v) {
            const auto& voxel_idx = filtered_voxels[v];

            double voxel_x = static_cast<double>(voxel_idx.first) * voxel_map_.res();
            double voxel_y = static_cast<double>(voxel_idx.second) * voxel_map_.res();
            double voxel_intensity = voxel_map_.at(voxel_idx);
            
            // Use local Jacobian with only active states
            int max_scans = tile.scan_ids.size();
            Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(max_scans, num_active_states + 1); // +1 for voxel intensity variable
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
                double inv_sqrt_meas_cov = 1 / std::sqrt(interp_meas->covariance);
                Eigen::Matrix<double, 1, 3> d_beta_d_T = interp_meas->jacobian * inv_sqrt_meas_cov;

                // Assemble into local Jacobian
                B(scan_count) = I_meas * inv_sqrt_meas_cov;
                weighted_ones(scan_count) = inv_sqrt_meas_cov;
                if (scan_idx > 0) {
                    // Find position in local Jacobian
                    auto it = global_to_local.find((scan_idx - 1) * 3);
                    if (it != global_to_local.end())
                        J_local.block<1,3>(scan_count, it->second) =  - d_beta_d_T; // Minus since we dont simplify in full problem
                }

                scan_count++;
            }
            
            if (scan_count == 0) {
                // No scans cover this voxel
                // Add a zero prior to keep the variable in the problem and prevent singularity
                int voxel_index = vox_to_global_pos[voxel_idx] + states_size; // voxel variables start after state variables
                local_triplets.emplace_back(voxel_index, voxel_index, 1e-6);
                continue;
            }
            
            // Trim to actual scan count; keep +1 voxel-intensity column.
            J_local.conservativeResize(scan_count, num_active_states + 1);
            B.conservativeResize(scan_count);
            weighted_ones.conservativeResize(scan_count);

            // Add intensity Jacobian column
            J_local.block(0, num_active_states, scan_count, 1) = weighted_ones;

            // Compute local contributions
            Eigen::VectorXd err = weighted_ones * voxel_intensity - B;
            Eigen::MatrixXd H_local = J_local.transpose() * J_local;
            Eigen::VectorXd g_local = J_local.transpose() * err;

            // Scatter to global lhs_ and rhs_
            // Diagonal voxel intensity variable contributions
            int voxel_index = vox_to_global_pos[voxel_idx] + states_size;
            local_triplets.emplace_back(voxel_index, voxel_index, H_local(num_active_states, num_active_states));
            rhs_local(voxel_index) += g_local(num_active_states);

            for (size_t i = 0; i < active_state_indices.size(); ++i) {
                int gi = active_state_indices[i];

                // diagonal
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        local_triplets.emplace_back(gi + r, gi + c, H_local(i*3 + r, i*3 + c));

                // only need upper diagonal since using solver that exploits symmetry
                // off-diagonal between pose variables
                for (size_t j = i + 1; j < active_state_indices.size(); ++j) {
                    int gj = active_state_indices[j];
                    auto Hij = H_local.block<3,3>(i*3, j*3);

                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            local_triplets.emplace_back(gi + r, gj + c, Hij(r, c));
                }

                // off-diagonal between pose variables and voxel intensity variable
                auto Hiv = H_local.block<3,1>(i*3, num_active_states);
                for (int r = 0; r < 3; ++r)
                    local_triplets.emplace_back(gi + r, voxel_index, Hiv(r));

                rhs_local.segment<3>(gi) += g_local.segment<3>(i*3);
            }

            cost_local += 0.5 * (err.squaredNorm());
        }
        #pragma omp critical
        {
            // Add local tile contributions to global matrices
            Eigen::SparseMatrix<double> tmp(states_size + voxels_size, states_size + voxels_size);
            tmp.setFromTriplets(local_triplets.begin(), local_triplets.end());
            lhs_sp_ += tmp;
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
}

bool DirectSolver::solve() {
    // Regularize by adding to diagonal (sparse-friendly)
    for (int i = 0; i < lhs_sp_.rows(); ++i)
        lhs_sp_.coeffRef(i, i) += 1e-8;

    solver_.analyzePattern(lhs_sp_);    // sparsity seems to change every iter... not 100% why
    solver_.factorize(lhs_sp_);        // numeric factorization

    if (solver_.info() != Eigen::Success) {
        std::cerr << "Sparse factorization failed." << std::endl;
        return false;
    }

    del_x_ = -alpha_ * solver_.solve(rhs_);

    std::cout << "number non-zeros in lhs: " << lhs_sp_.nonZeros() << std::endl;

    std::cout << "alpha: " << alpha_
            << " |delta|: " << del_x_.norm()
            << " cost: " << cost_ << std::endl;

    return true;
}

void DirectSolver::update_states() {
    // Update poses
    if (del_x_.size() == 0) {
        throw std::runtime_error("No pose updates available. Have you run solve()?");
    }
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    std::vector<lgmath::se2::Transformation> updated_poses;

    // Save sizes
    int size_pose_state = scan_manager_.num_active_scans() * 3;
    int expected_state_size = size_pose_state + static_cast<int>(optimized_voxel_keys_.size());
    if (del_x_.size() != expected_state_size) {
        throw std::runtime_error("State update size mismatch in DirectSolver::update_states.");
    }

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

    // Update map intensities
    std::vector<ba::VoxelMap::Index> voxel_keys = optimized_voxel_keys_;
    // Create voxel lookup for quick access to column indices in the optimization variable
    std::map<VoxelMap::Index, std::size_t> vox_to_global_pos;
    for (std::size_t i = 0; i < voxel_keys.size(); ++i) {
        vox_to_global_pos[voxel_keys[i]] = i;
    }
    for (const auto& voxel_idx : voxel_keys) {
        int voxel_index = vox_to_global_pos[voxel_idx] + size_pose_state; // voxel variables start after state variables
        double delta_vi = del_x_(voxel_index);
        voxel_map_.at(voxel_idx) += delta_vi;
    }
}

void DirectSolver::optimize() {
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

        if (iter > 2) {
            // Slowly decrease alpha
            alpha_ *= 0.8;
        }

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
            update_states();

            // Recompute full del_x
            del_x_ /= -alpha_;

            // Reduce alpha
            alpha_ *= std::pow(0.5, num_cost_rises);

            // Compute new step
            del_x_ *= alpha_;

            // Try a new update!
            update_states();

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
        update_states();
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
}


} // namespace ba