// voxel_map.hpp
#pragma once

#include <ankerl/unordered_dense.h>

#include <cstdint>
#include <utility>
#include <vector>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/se3/Transformation.hpp>
#include <ba/scans/manager.hpp>

namespace ba {

class VoxelMap {
public:
	using Index = std::pair<int32_t, int32_t>;
	using Coord = std::pair<double, double>;

	explicit VoxelMap(double res);
	explicit VoxelMap(const std::string& filepath);

	// Convert world coordinates (meters) to voxel indices
	Index index(double x, double y) const;

	// Number of stored voxels
	std::size_t size() const;

	std::vector<lgmath::se2::Transformation> poses() const { return poses_; }
	void set_poses(const std::vector<int>& pose_ids, const std::vector<lgmath::se2::Transformation>& poses) {
		pose_ids_ = pose_ids;
		poses_ = poses;
	}
	std::vector<int> pose_ids() const { return pose_ids_; }

	// Return bounds of the voxel map in meters
	std::pair<double, double> x_bounds() const { return x_bounds_; }
	std::pair<double, double> y_bounds() const { return y_bounds_; }

	Coord voxel_to_coord(Index idx) const;

	void init_voxel(Index idx);

	// Create or set a voxel intensity at integer coordinates
	void add_single_voxel(int32_t a, int32_t b, double intensity);

	// Create or set a voxel intensity at cartesian coordinates in meters
	void add_single_voxel(double x, double y, double intensity);

	// Reset all intensities to 0.0
	void zero_out();

	// Fill all existing voxels with random values in [min_val, max_val]
	void randomize(uint32_t seed = 0); /* deterministic if seed!=0 */

	void overwrite(Scan& scan);

	// Initialize empty voxels in a square window around pose within max_dist
	// If SE3 pose provided, only the SE2 components are used
	void init_map(const lgmath::se2::Transformation& pose, double max_dist, int pose_id);
	void init_map(const lgmath::se3::Transformation& pose, double max_dist, int pose_id);

	// Get all voxel indices within max_dist of a given pose
	std::vector<Index> get_voxels_in_range(const lgmath::se2::Transformation& pose, double max_dist) const;

	// Return sorted voxel keys downsampled by factor in [0,1]; 1.0 means no downsample
	std::vector<Index> get_sorted_keys_downsampled(double downsample_factor = 1.0) const;

	// Direct access helpers
	bool contains(Index idx) const;
	double& at(Index idx);
	const double& at(Index idx) const;

	// Resolution access
	double res() const { return res_; }

	// Visualize as pixel image (for debugging)
	void visualize(double downsample_factor = 1.0) const;

	// Save map as a binary file
	void save_to_file(const std::string& filepath, ScanManager& scan_manager) const;

	// Load map from a binary file
	void load_from_file(const std::string& filepath);

	// Load only popses from file
	void load_poses_from_file(const std::string& filepath);

	void clear_pose_data() {
		pose_ids_.clear();
		poses_.clear();
	}

private:
	double res_;
	std::vector<int> pose_ids_;
	std::vector<lgmath::se2::Transformation> poses_;
	ankerl::unordered_dense::map<Index, double> voxels_;
	std::pair<double, double> x_bounds_;
	std::pair<double, double> y_bounds_;
};

} // namespace ba
