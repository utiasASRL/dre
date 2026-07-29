// ba_problem.hpp
#pragma once

#include <ba/problem/problem.hpp>

namespace ba {

class LocProblem : public Problem {
public:
    struct LocResultEntry {
        int map_id;
        int scan_id;
        double est_x;
        double est_y;
        double est_yaw;
        double gt_x;
        double gt_y;
        double gt_yaw;
        double std_x;
        double std_y;
        double std_yaw;
    };

    LocProblem(Options& opts) : Problem("loc", opts) {}

    void validate_opts() override;
    void init_seq_id() override;
    void get_scan_indeces() override;
    void init_scans_and_map() override;
    void finalize() override;

    void load_map_from_estimate();
    void load_scans();
    void add_loc_result(const LocResultEntry& entry) {
        loc_results_.push_back(entry);
    }

    // Output
    void save_loc_results(const fs::path &output_path = fs::path());
    void visualize_loc_results();

    std::vector<lgmath::se3::Transformation> gt_map_poses() const { return gt_map_poses_; }
    std::vector<lgmath::se3::Transformation> gt_poses() const { return gt_poses_; }
    std::vector<lgmath::se3::Transformation> dro_poses() const { return dro_poses_; }

private:
    std::vector<lgmath::se3::Transformation> gt_map_poses_;
    std::vector<lgmath::se3::Transformation> gt_poses_;
    std::vector<lgmath::se3::Transformation> dro_poses_;
    std::vector<LocProblem::LocResultEntry> loc_results_;
};

}   // namespace ba