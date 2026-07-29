// ba_problem.hpp
#pragma once

#include <ba/problem/problem.hpp>

namespace ba {

class MapProblem : public Problem {
public:
    MapProblem(Options& opts) : Problem("map", opts) {}

    void validate_opts() override;
    void init_seq_id() override;
    void get_scan_indeces() override;
    void init_scans_and_map() override;
    void init_scans_and_map_from_estimates();
    void init_scans_and_map_from_data();
    void finalize() override;

private:


};

}   // namespace ba