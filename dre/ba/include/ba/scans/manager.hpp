// loader.hpp
#pragma once

#include <string>
#include <memory>
#include <ba/scans/scan.hpp>
#include <ba/scans/local_map_scan.hpp>
#include <ankerl/unordered_dense.h>
#include <iostream>
#include <queue>

namespace ba {

class ScanManager {
public:
    ScanManager(int max_loaded_scans = 0) {
        if (max_loaded_scans >= 1) {
            max_loaded_scans_ = static_cast<size_t>(max_loaded_scans);
        } else {
            // Set equal to max int to effectively disable limit
            max_loaded_scans_ = std::numeric_limits<size_t>::max();
        }
    }

    void set_max_loaded_scans(int max_loaded_scans) {
        if (max_loaded_scans >= 1) {
            max_loaded_scans_ = static_cast<size_t>(max_loaded_scans);
        } else {
            throw std::invalid_argument("max_loaded_scans must be >= 1");
        }
    }

    void add_scan(std::shared_ptr<Scan> scan) {
        id_to_idx_[scan->id()] = num_scans();
        idx_to_id_.push_back(scan->id());
        scans_.emplace(scan->id(), scan);
    }

    std::shared_ptr<Scan> get_scan(int scan_id) {
        return scans_.at(scan_id);
    }

    std::shared_ptr<const Scan> get_scan(int scan_id) const {
        return scans_.at(scan_id);
    }

    bool has_scan(int scan_id) const {
        return scans_.find(scan_id) != scans_.end();
    }

    int num_scans() const {
        return static_cast<int>(scans_.size());
    }

    int queue_size() const {
        return static_cast<int>(loaded_scan_queue_.size());
    }

    void set_ref_timestamp(int64_t timestamp) {
        ref_timestamp_ = timestamp;
    }

    int64_t ref_timestamp() const {
        return ref_timestamp_;
    }

    int num_active_scans() const {
        int count = 0;
        for (const auto& kv : scans_) {
            if (!kv.second->is_fixed()) {
                count++;
            }
        }
        return count;
    }

    void load_data(std::vector<int> scan_ids = {}) {
        std::vector<int> load_scan_ids;
        if (scan_ids.empty()) {
            load_scan_ids = idx_to_id_;
        } else {
            load_scan_ids = scan_ids;
        }
        // Ensure we do not exceed max_loaded_scans_
        if (load_scan_ids.size() > max_loaded_scans_) {
            throw std::runtime_error("ScanManager: Trying to load " + std::to_string(scans_.size()) + 
                                        " scans with max_loaded_scans set to " + std::to_string(max_loaded_scans_));
        }

        // Load scans, unloading old scans if necessary
        for (int scan_id : load_scan_ids) {
            auto scan = scans_.at(scan_id);
            if (scan->data_loaded()) {
                continue; // already loaded
            }
            if (loaded_scan_queue_.size() >= max_loaded_scans_) {
                // Find scan thats not in the current load list
                while (!loaded_scan_queue_.empty()) {
                    int unload_scan_id = loaded_scan_queue_.front();
                    // Check if this scan is in the current load list
                    if (std::find(load_scan_ids.begin(), load_scan_ids.end(), unload_scan_id) == load_scan_ids.end()) {
                        // Not in load list, unload it
                        loaded_scan_queue_.pop();
                        scans_.at(unload_scan_id)->unload_data();
                        break;
                    } else {
                        // In load list, move to back of queue
                        loaded_scan_queue_.pop();
                        loaded_scan_queue_.push(unload_scan_id);
                    }
                }
            }
            scan->load_data();
            loaded_scan_queue_.push(scan_id);
        }
    }

    void unload_all_data() {
        while (!loaded_scan_queue_.empty()) {
            int scan_id = loaded_scan_queue_.front();
            loaded_scan_queue_.pop();
            scans_.at(scan_id)->unload_data();
        }
    }

    std::vector<int> get_all_scan_ids() const {
        return idx_to_id_;
    }

    int idx_to_id(int scan_idx) const {
        return idx_to_id_.at(scan_idx);
    }

    int id_to_idx(int scan_id) const {
        return id_to_idx_.at(scan_id);
    }

    void apply_noise_to_scans(double pos_stddev, double yaw_stddev) {
        for (const auto& kv : scans_) {
            kv.second->apply_noise_to_pose(pos_stddev, yaw_stddev);
        }
    }

    // Compute RMSE of all scan poses compared to groundtruth (SE2: x, y, yaw)
    Eigen::Matrix<double, 3, 1> compute_pose_rmse() const {
        if (scans_.empty()) {
            return Eigen::Matrix<double, 3, 1>::Zero();
        }
        Eigen::Matrix<double, 3, 1> rmse = Eigen::Matrix<double, 3, 1>::Zero();
        for (const auto& kv : scans_) {
            if (kv.second->is_fixed()) {
                continue;
            }
            Eigen::Matrix<double, 6, 1> err = kv.second->pose_error();
            rmse(0) += err(0) * err(0); // x
            rmse(1) += err(1) * err(1); // y
            rmse(2) += err(5) * err(5); // yaw
        }
        rmse /= static_cast<double>(num_active_scans());
        rmse = rmse.cwiseSqrt();
        rmse(2) *= (180.0 / M_PI); // convert yaw to degrees
        return rmse;
    }

    double compute_epe() const {
        if (scans_.empty()) {
            return 0.0;
        }
        // Get first and last scan poses
        auto first_scan = scans_.at(idx_to_id_.front());
        auto last_scan = scans_.at(idx_to_id_.back());
        auto T_0 = first_scan->pose();
        auto T_N = last_scan->pose();
        auto T_gt_0 = first_scan->gt_pose();
        auto T_gt_N = last_scan->gt_pose();
        
        // Compute EPE
        auto T_est_epe = T_gt_0.inverse() * T_gt_N;
        auto T_gt_epe = T_0.inverse() * T_N;
        auto T_epe_err = T_est_epe.inverse() * T_gt_epe;
        double epe = T_epe_err.r_ab_inb().head<2>().norm();
        return epe;
    }

    double compute_ate() const {
        if (scans_.empty()) {
            return 0.0;
        }
        // Load in array of xy positions
        Eigen::Matrix<double, 2, Eigen::Dynamic> est_positions(2, num_scans());
        Eigen::Matrix<double, 2, Eigen::Dynamic> gt_positions(2, num_scans());
        int idx = 0;
        for (const auto& kv : scans_) {
            lgmath::se2::Transformation T_est = kv.second->pose2d();
            lgmath::se2::Transformation T_gt = kv.second->gt_pose2d();
            est_positions(0, idx) = T_est.r_ab_inb()(0);
            est_positions(1, idx) = T_est.r_ab_inb()(1);
            gt_positions(0, idx) = T_gt.r_ab_inb()(0);
            gt_positions(1, idx) = T_gt.r_ab_inb()(1);
            idx++;
        }

        // Get centroids
        Eigen::Vector2d est_centroid = est_positions.rowwise().mean();
        Eigen::Vector2d gt_centroid = gt_positions.rowwise().mean();

        // Center the positions
        Eigen::Matrix<double, 2, Eigen::Dynamic> est_centered = est_positions.colwise() - est_centroid;
        Eigen::Matrix<double, 2, Eigen::Dynamic> gt_centered = gt_positions.colwise() - gt_centroid;

        // Compute covariance matrix
        Eigen::Matrix2d H = gt_centered * est_centered.transpose();

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();

        // Correct for reflection
        if (R.determinant() < 0) {
            Eigen::Matrix2d V = svd.matrixV();
            V.col(1) *= -1;
            R = V * svd.matrixU().transpose();
        }

        // Compute translation
        Eigen::Vector2d t = est_centroid - R * gt_centroid;

        // Apply transformation to estimated positions
        Eigen::Matrix<double, 2, Eigen::Dynamic> est_aligned = (R * gt_positions).colwise() + t;

        // Compute ATE
        double ate = 0.0;
        for (int i = 0; i < num_scans(); ++i) {
            double ate_contribution = (est_aligned.col(i) - gt_positions.col(i)).squaredNorm();
            int pose_id = idx_to_id_.at(i);
            scans_.at(pose_id)->set_ate_error(std::sqrt(ate_contribution));
            ate += (est_aligned.col(i) - est_positions.col(i)).squaredNorm();
        }
        ate = std::sqrt(ate / static_cast<double>(num_scans()));

        return ate;
    }

    ScanManager deep_copy() const {
        ScanManager copy;
        for (const auto& scan_id : idx_to_id_) {
            const auto& scan = scans_.at(scan_id);
            auto scan_clone = scan->clone();
            copy.add_scan(scan_clone);
        }
        return copy;
    }

private:
    ankerl::unordered_dense::map<int, std::shared_ptr<Scan>> scans_;
    std::vector<int> idx_to_id_;                  // scan_idx -> scan_id
    std::unordered_map<int, std::size_t> id_to_idx_; // scan_id -> scan_idx
    std::queue<int> loaded_scan_queue_;
    size_t max_loaded_scans_ = 0; // 0 means no limit
    int64_t ref_timestamp_ = 0; // The timestamp of the first scan in the sequence, even if its not added
};


} // namespace ba