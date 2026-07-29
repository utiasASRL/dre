#pragma once

#include <ba/map/voxel_map.hpp>
#include <Eigen/Dense>
#include <ba/problem/problem.hpp>
#include <ba/solver/solver.hpp>

namespace ba {

class DirectSolver : public Solver {
public:
    struct Tile {
        std::vector<VoxelMap::Index> voxel_indices;
        std::vector<int> scan_ids;
    };

    DirectSolver(Problem& problem) : Solver(problem) {}

    void tile_problem();
    void construct_problem(double downsample_factor = 1.0);
    bool solve();
    void update_states();
    void optimize() override;

    ~DirectSolver() = default;

private:
    // Problem tiling
    std::vector<Tile> tiles_;
    std::vector<VoxelMap::Index> optimized_voxel_keys_;
    
    // Variables to be passed around
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver_;
    Eigen::VectorXd del_x_;
};


}