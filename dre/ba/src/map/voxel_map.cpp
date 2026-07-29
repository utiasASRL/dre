#include <ba/map/voxel_map.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <lgmath/so2/Rotation.hpp>

namespace ba {

VoxelMap::VoxelMap(double res) : res_(res) {}
VoxelMap::VoxelMap(const std::string& filepath) {
    load_from_file(filepath);
}

VoxelMap::Coord VoxelMap::voxel_to_coord(Index idx) const {
    double x = static_cast<double>(idx.first) * res_;
    double y = static_cast<double>(idx.second) * res_;
    return {x, y};
}

auto VoxelMap::index(double x, double y) const -> Index {
	const int32_t a = static_cast<int32_t>(std::floor(x / res_));
	const int32_t b = static_cast<int32_t>(std::floor(y / res_));
	return {a, b};
}

std::size_t VoxelMap::size() const { return voxels_.size(); }

void VoxelMap::add_single_voxel(int32_t a, int32_t b, double intensity) {
    if (voxels_.contains({a, b})) return;
	voxels_[{a, b}] = intensity;
}

void VoxelMap::add_single_voxel(double x, double y, double intensity) {
    const Index ab = index(x, y);
    if (voxels_.contains(ab)) return;
    voxels_[ab] = intensity;
}

void VoxelMap::zero_out() {
	for (auto& kv : voxels_) kv.second = 0.0;
}

void VoxelMap::randomize(uint32_t seed) {
	std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (auto& kv : voxels_) kv.second = dist(rng);
}

void VoxelMap::overwrite(Scan& scan) {
    for (const auto& kv : voxels_) {
        double voxel_x = static_cast<double>(kv.first.first) * res_;
        double voxel_y = static_cast<double>(kv.first.second) * res_;
        // Interpolate intensity and Jacobian
        std::optional<Scan::Measurement> interp_meas = scan.interpolate(voxel_x, voxel_y);
        // If scan is outside coverage, no intensity will be provided
        if (!interp_meas.has_value()) {
            continue;
        }

        double I_meas = interp_meas->intensity;
        voxels_[kv.first] = I_meas;
    }
}

void VoxelMap::init_voxel(Index idx) {
    if (contains(idx)) return;
    voxels_.emplace(idx, 0.0);
    // Update bounds
    if (size() == 1) {
        Coord coord = voxel_to_coord(idx);
        x_bounds_ = {coord.first, coord.first};
        y_bounds_ = {coord.second, coord.second};
    } else {
        Coord coord = voxel_to_coord(idx);
        x_bounds_.first = std::min(x_bounds_.first, coord.first);
        x_bounds_.second = std::max(x_bounds_.second, coord.first);
        y_bounds_.first = std::min(y_bounds_.first, coord.second);
        y_bounds_.second = std::max(y_bounds_.second, coord.second);
    }
}

void VoxelMap::init_map(const lgmath::se3::Transformation& pose, double max_dist, int pose_id) {
    const Eigen::Matrix<double, 4, 4> pose_mat = pose.matrix();
	const double x_center = pose_mat(0, 3);
	const double y_center = pose_mat(1, 3);
	const auto center = index(x_center, y_center);
	const int32_t n = static_cast<int32_t>(std::ceil(max_dist / res_));
	for (int32_t da = -n; da <= n; ++da) {
		for (int32_t db = -n; db <= n; ++db) {
            if (std::sqrt(std::pow(da * res_, 2) + std::pow(db * res_, 2)) > max_dist) {
                continue;
            }
			const Index idx{center.first + da, center.second + db};
            init_voxel(idx);
		}
	}
    // Store pose
    poses_.push_back(pose.toSE2());
    pose_ids_.push_back(pose_id);
}

void VoxelMap::init_map(const lgmath::se2::Transformation& pose, double max_dist, int pose_id) {
    init_map(pose.toSE3(), max_dist, pose_id);
}

std::vector<VoxelMap::Index> VoxelMap::get_voxels_in_range(const lgmath::se2::Transformation& pose, double max_dist) const {
    std::vector<Index> indices;
    const Eigen::Matrix<double, 3, 3> pose_mat = pose.matrix();
    const double x_center = pose_mat(0, 2);
    const double y_center = pose_mat(1, 2);
    const auto center = index(x_center, y_center);
    const int32_t n = static_cast<int32_t>(std::ceil(max_dist / res_));
    for (int32_t da = -n; da <= n; ++da) {
        for (int32_t db = -n; db <= n; ++db) {
            if (std::sqrt(std::pow(da * res_, 2) + std::pow(db * res_, 2)) > max_dist) {
                continue;
            }
            const Index idx{center.first + da, center.second + db};
            if (contains(idx)) {
                indices.push_back(idx);
            }
        }
    }
    return indices;
}

std::vector<VoxelMap::Index> VoxelMap::get_sorted_keys_downsampled(double downsample_factor) const {
    if (voxels_.empty()) return {};
    if (downsample_factor > 1.0 || downsample_factor <= 0.0) {
        throw std::invalid_argument("Downsample factor must be in (0, 1]");
    }

    std::vector<Index> keys;
    keys.reserve(voxels_.size());

    // Collect and sort all keys
    for (const auto& kv : voxels_) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    // No downsample
    if (downsample_factor >= 1.0) return keys;

    std::vector<Index> downsampled_keys;
    downsampled_keys.reserve(static_cast<std::size_t>(std::ceil(keys.size() * downsample_factor)));

    const std::size_t step = static_cast<std::size_t>(
        std::max<double>(1.0, std::round(1.0 / downsample_factor)));
    for (std::size_t i = 0; i < keys.size(); i += step) {
        downsampled_keys.push_back(keys[i]);
    }
    return downsampled_keys;
}

void VoxelMap::visualize(double downsample_factor) const {
    if (voxels_.empty()) return;

    // Find bounds of the indices
    int32_t min_x = std::numeric_limits<int32_t>::max();
    int32_t max_x = std::numeric_limits<int32_t>::min();
    int32_t min_y = std::numeric_limits<int32_t>::max();
    int32_t max_y = std::numeric_limits<int32_t>::min();

    for (const auto &[idx, val] : voxels_) {
        min_x = std::min(min_x, idx.first);
        max_x = std::max(max_x, idx.first);
        min_y = std::min(min_y, idx.second);
        max_y = std::max(max_y, idx.second);
    }

    int width  = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    cv::Mat img(height, width, CV_64F, cv::Scalar(0)); // use double for intensity

    // Fill image
    for (const auto &[idx, val] : voxels_) {
        int x = idx.first - min_x;
        int y = idx.second - min_y;
        img.at<double>(y, x) = val; // row = y = u, col = x = v
    }

    // Normalize to 0-255 and convert to 8-bit for display
    cv::Mat img8;
    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);
    img.convertTo(img8, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Show image
    cv::namedWindow("Scan", cv::WINDOW_NORMAL);
    cv::resizeWindow("Scan", 480, 480);
    cv::imshow("Scan", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void VoxelMap::save_to_file(const std::string& filepath, ScanManager& scan_manager) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    // Pre-compute ATE per scan
    scan_manager.compute_ate();

    // Write metadata
    ofs.write(reinterpret_cast<const char*>(&res_), sizeof(res_));
    uint32_t num_poses = static_cast<uint32_t>(poses_.size());
    ofs.write(reinterpret_cast<const char*>(&num_poses), sizeof(num_poses));
    uint32_t num_voxels = static_cast<uint32_t>(voxels_.size());
    ofs.write(reinterpret_cast<const char*>(&num_voxels), sizeof(num_voxels));
    // Write poses, pose ids, and ate (for evaluating map quality)
    for (size_t i = 0; i < poses_.size(); ++i) {
        int32_t pose_id = pose_ids_[i];
        ofs.write(reinterpret_cast<const char*>(&pose_id), sizeof(pose_id));
        const auto& pose = poses_[i];
        double x = pose.r_ab_inb()(0);
        double y = pose.r_ab_inb()(1);
        double yaw = pose.vec()(2);
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(x));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(y));
        ofs.write(reinterpret_cast<const char*>(&yaw), sizeof(yaw));
        // Get ATE
        auto scan = scan_manager.get_scan(pose_id);
        double ate = scan->get_ate_error();
        // Zero out ate less than 1e-6 to avoid numerical issues
        if (std::abs(ate) < 1e-6) ate = 0.0;
        ofs.write(reinterpret_cast<const char*>(&ate), sizeof(ate));
    }

    // Write voxel data
    for (const auto& kv : voxels_) {
        int32_t x = kv.first.first;
        int32_t y = kv.first.second;
        double intensity = kv.second;
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(x));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(y));
        ofs.write(reinterpret_cast<const char*>(&intensity), sizeof(intensity));
    }

    ofs.close();
}

void VoxelMap::load_poses_from_file(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    clear_pose_data();
    // Read metadata
    ifs.read(reinterpret_cast<char*>(&res_), sizeof(res_));
    uint32_t num_poses = 0;
    ifs.read(reinterpret_cast<char*>(&num_poses), sizeof(num_poses));
    // Skip voxel count
    uint32_t num_voxels = 0;
    ifs.read(reinterpret_cast<char*>(&num_voxels), sizeof(num_voxels));
    // Read poses
    for (uint32_t i = 0; i < num_poses; ++i) {
        int pose_id;
        ifs.read(reinterpret_cast<char*>(&pose_id), sizeof(pose_id));
        pose_ids_.push_back(pose_id);
        double x, y, yaw;
        ifs.read(reinterpret_cast<char*>(&x), sizeof(x));
        ifs.read(reinterpret_cast<char*>(&y), sizeof(y));
        ifs.read(reinterpret_cast<char*>(&yaw), sizeof(yaw));
        lgmath::so2::Rotation C(yaw);
        Eigen::Vector2d r;
        r << x, y;
        lgmath::se2::Transformation pose(C.matrix(), -C.matrix().transpose() * r);
        poses_.push_back(pose);
        double ate;
        ifs.read(reinterpret_cast<char*>(&ate), sizeof(ate));
    }
    ifs.close();
}

void VoxelMap::load_from_file(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    // Read metadata
    ifs.read(reinterpret_cast<char*>(&res_), sizeof(res_));
    uint32_t num_poses = 0;
    ifs.read(reinterpret_cast<char*>(&num_poses), sizeof(num_poses));
    uint32_t num_voxels = 0;
    ifs.read(reinterpret_cast<char*>(&num_voxels), sizeof(num_voxels));
    // Read poses
    poses_.clear();
    for (uint32_t i = 0; i < num_poses; ++i) {
        int pose_id;
        ifs.read(reinterpret_cast<char*>(&pose_id), sizeof(pose_id));
        pose_ids_.push_back(pose_id);
        double x, y, yaw;
        ifs.read(reinterpret_cast<char*>(&x), sizeof(x));
        ifs.read(reinterpret_cast<char*>(&y), sizeof(y));
        ifs.read(reinterpret_cast<char*>(&yaw), sizeof(yaw));
        lgmath::so2::Rotation C(yaw);
        Eigen::Vector2d r;
        r << x, y;
        lgmath::se2::Transformation pose(C.matrix(), -C.matrix().transpose() * r);
        poses_.push_back(pose);
        double ate;
        ifs.read(reinterpret_cast<char*>(&ate), sizeof(ate));
    }
    std::cout << "Loaded " << num_poses << " poses from file: " << filepath << std::endl;
    // Read voxel data
    voxels_.clear();
    std::cout << "Loading " << num_voxels << " voxels from file: " << filepath << std::endl;
    for (uint32_t i = 0; i < num_voxels; ++i) {
        int32_t x, y;
        double intensity;
        ifs.read(reinterpret_cast<char*>(&x), sizeof(x));
        ifs.read(reinterpret_cast<char*>(&y), sizeof(y));
        ifs.read(reinterpret_cast<char*>(&intensity), sizeof(intensity));
        voxels_[{x, y}] = intensity;
    }
    ifs.close();
}

bool VoxelMap::contains(Index idx) const { return voxels_.find(idx) != voxels_.end(); }

double& VoxelMap::at(Index idx) { return voxels_.at(idx); }

const double& VoxelMap::at(Index idx) const { return voxels_.at(idx); }

}