#pragma once

#include <ba/scans/manager.hpp>
#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <Eigen/Dense>
#include <ankerl/unordered_dense.h>
#include <ba/solver/result.hpp>
#include <ba/problem/problem.hpp>
#include <limits>
#include <stdexcept>
#include <Eigen/Sparse>

namespace ba {

class Solver {
public:

    static OptimizationOptions select_optimization_options(Problem& problem) {
        if (problem.type() == "ba") {
            return problem.opts().ba_opts.optimization_opts;
        }
        if (problem.type() == "map") {
            return problem.opts().map_opts.optimization_opts;
        }
        if (problem.type() == "loc") {
            return problem.opts().loc_opts.optimization_opts;
        }
        throw std::invalid_argument("Unknown problem type: " + problem.type());
    }

    Solver(Problem& problem)
        : problem_(problem),
          full_opts_(problem.opts()),
          result_(problem.result()),
          scan_manager_(problem.scan_manager()),
          voxel_map_(problem.voxel_map()),
          pose_priors_(problem.pose_priors()),
          save_results_(problem_.opts().save_result)
        {
            opts_ = select_optimization_options(problem);
            cost_ = std::numeric_limits<double>::max();
            prev_cost_ = std::numeric_limits<double>::max();
            alpha_ = opts_.alpha;
        }

    ~Solver() = default;

    // Functions to be fulfilled by derived classes
    virtual void optimize() = 0;

    // Accessors
    double cost() const { return cost_; }

    // Writers
    void set_alpha(double alpha) { alpha_ = alpha; }

protected:
    Problem& problem_;
    Options& full_opts_;
    Result& result_;
    ScanManager& scan_manager_;
    VoxelMap& voxel_map_;
    const Problem::PriorMap& pose_priors_;
    bool save_results_;
    OptimizationOptions opts_;

    // Variables to be passed around
    double cost_;
    double prev_cost_;
    double alpha_;
    Eigen::MatrixXd lhs_;
    Eigen::VectorXd rhs_;
    // Sparse option
    Eigen::SparseMatrix<double> lhs_sp_;
    std::vector<Eigen::Triplet<double>> triplets_;
};


}