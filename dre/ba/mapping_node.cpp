#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/path.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "dre/msg/local_map_info.hpp"

#include <ba/map/voxel_map.hpp>
#include <ba/scans/local_map_scan.hpp>
#include <ba/utils/ba_config.hpp>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/so2/Rotation.hpp>

namespace fs = std::filesystem;

// Periodically reconstructs a voxel map directly from DRO's local maps (no
// bundle-adjustment refinement, unlike the offline dr_map pipeline) and saves
// it to disk, publishing the output path whenever the map is updated.
class MappingNode : public rclcpp::Node {
public:
    MappingNode() : Node("mapping_node") {
        this->declare_parameter<std::string>("config_file", "");
        std::string config_file = this->get_parameter("config_file").as_string();
        if (config_file.empty()) {
            throw std::runtime_error("Config file path must be provided as a parameter 'config_file'.");
        }

        this->declare_parameter<std::string>("output_path", "");
        output_path_ = this->get_parameter("output_path").as_string();
        if (output_path_.empty()) {
            throw std::runtime_error("Output path must be provided as a parameter 'output_path'.");
        }
        fs::create_directories(output_path_);
        map_file_path_ = (fs::path(output_path_) / "voxel_map.bin").string();

        // Keyframe local maps are persisted here (same layout and stamp-based
        // naming as DRO's offline save_local_maps output) so LocalMapScan can
        // reload image data from disk on demand.
        local_maps_dir_ = fs::path(output_path_) / "local_maps";
        cumul_dir_ = fs::path(output_path_) / "cumulated_returns";
        fs::create_directories(local_maps_dir_);
        fs::create_directories(cumul_dir_);

        load_config(config_file);

        voxel_map_ = std::make_unique<ba::VoxelMap>(voxel_map_res_);

        // transient_local so late-joining subscribers (map_viz_node) get the
        // latest map path immediately; matches loc_node's /map_path QoS.
        rclcpp::QoS map_path_qos(1);
        map_path_qos.transient_local();
        map_path_pub_ = create_publisher<std_msgs::msg::String>("map_path", map_path_qos);

        sub_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // The three DRO topics share identical stamps, so exact-time sync.
        // (message_filters subscribers use the node's default callback group,
        // which is distinct from timer_group_, so intake still runs while a
        // map recompute is in progress.)
        image_sub_.subscribe(this, "/dro_local_map_image");
        cumul_sub_.subscribe(this, "/dro_cumulated_returns_image");
        info_sub_.subscribe(this, "/dro_local_map_info");
        sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), image_sub_, cumul_sub_, info_sub_);
        sync_->registerCallback(std::bind(&MappingNode::localMapCallback, this,
                                          std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = sub_group_;
        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/pogo_path", 10,
            std::bind(&MappingNode::pogoPathCallback, this, std::placeholders::_1),
            sub_opts);

        // The map update runs on its own thread rather than a ROS timer: the
        // Foxy multi-threaded executor delays timers while the image
        // subscriptions keep it busy, which let the scan queue grow unboundedly.
        // The worker integrates scans as soon as they are queued and paces
        // save/publish by map_update_period_sec. PNG writes for newly-queued
        // keyframes also happen here (see localMapCallback), not on the
        // subscription thread, so intake never blocks on disk I/O.
        update_thread_ = std::thread(&MappingNode::updateLoop, this);

        RCLCPP_INFO(get_logger(), "MappingNode started. Map will be saved to '%s' at most every %.1f s.",
                    map_file_path_.c_str(), map_update_period_sec_);
    }

    ~MappingNode() override {
        {
            std::lock_guard<std::mutex> lock(update_cv_mutex_);
            stop_ = true;
        }
        update_cv_.notify_all();
        if (update_thread_.joinable()) {
            update_thread_.join();
        }
    }

private:
    void load_config(const std::string& config_file) {
        YAML::Node config = YAML::LoadFile(config_file);
        voxel_map_res_ = config["voxel_map_resolution"].as<double>();
        max_dist_ = config["max_dist"].as<double>();
        map_update_period_sec_ = config["map_update_period_sec"].as<double>();
        if (config["max_kf_dist"]) {
            max_kf_dist_ = config["max_kf_dist"].as<double>();
        }
        if (config["max_kf_rot"]) {
            max_kf_rot_ = config["max_kf_rot"].as<double>();
        }
        if (config["cumul_thresh"]) {
            opt_opts_.cumul_thresh = config["cumul_thresh"].as<double>();
        }
        if (config["zero_thresh"]) {
            opt_opts_.zero_thresh = config["zero_thresh"].as<double>();
        }
        if (config["meas_std"]) {
            opt_opts_.meas_std = config["meas_std"].as<double>();
        }
        if (config["range_factor"]) {
            opt_opts_.range_factor = config["range_factor"].as<double>();
        }
        if (config["max_loaded_scans"]) {
            max_loaded_scans_ = config["max_loaded_scans"].as<int>();
        }
        if (config["pose_update_tol"]) {
            pose_update_tol_ = config["pose_update_tol"].as<double>();
        }
        if (config["max_moved_per_update"]) {
            max_moved_per_update_ = config["max_moved_per_update"].as<int>();
        }
        if (config["min_moved_per_update"]) {
            min_moved_per_update_ = config["min_moved_per_update"].as<int>();
        }
        if (config["num_threads"]) {
            // Applied inside updateLoop: omp_set_num_threads only affects the
            // calling thread, and all OpenMP regions run on the worker thread
            num_threads_ = config["num_threads"].as<int>();
        }
        Eigen::setNbThreads(1);
    }

    static int64_t stampToMicroseconds(const builtin_interfaces::msg::Time& stamp) {
        return static_cast<int64_t>(stamp.sec) * 1000000LL + static_cast<int64_t>(stamp.nanosec) / 1000LL;
    }

    // Build an SE(3) scan pose from a 2D (x, y, yaw) vehicle pose; same
    // construction the offline pipeline uses for dro/pogo poses.
    static lgmath::se3::Transformation makePose(double x, double y, double theta) {
        lgmath::so2::Rotation C(theta);
        Eigen::Vector2d r(x, y);
        lgmath::se2::Transformation pose_se2(C.matrix(), -C.matrix().transpose() * r);
        return pose_se2.toSE3();
    }

    static double yawFromQuaternion(double x, double y, double z, double w) {
        const double siny_cosp = 2.0 * (w * z + x * y);
        const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    // Pose graph nodes share the scans' microsecond timestamps, so lookups
    // normally hit exactly; interpolate between neighbours otherwise (as the
    // offline get_interpolated_pose does). Returns nullopt outside the path.
    static std::optional<lgmath::se3::Transformation> lookupPathPose(
            const std::vector<int64_t>& times,
            const std::vector<std::array<double, 3>>& poses_xyt,
            int64_t t_us) {
        if (times.empty() || t_us < times.front() || t_us > times.back()) {
            return std::nullopt;
        }
        auto it = std::lower_bound(times.begin(), times.end(), t_us);
        size_t idx = std::distance(times.begin(), it);
        if (times[idx] == t_us) {
            return makePose(poses_xyt[idx][0], poses_xyt[idx][1], poses_xyt[idx][2]);
        }
        // Not an exact hit: t_us is strictly between times[idx - 1] and times[idx]
        size_t i = idx - 1;
        double alpha = static_cast<double>(t_us - times[i]) / static_cast<double>(times[idx] - times[i]);
        double x = (1.0 - alpha) * poses_xyt[i][0] + alpha * poses_xyt[idx][0];
        double y = (1.0 - alpha) * poses_xyt[i][1] + alpha * poses_xyt[idx][1];
        double dtheta = std::atan2(std::sin(poses_xyt[idx][2] - poses_xyt[i][2]),
                                   std::cos(poses_xyt[idx][2] - poses_xyt[i][2]));
        double theta = poses_xyt[i][2] + alpha * dtheta;
        return makePose(x, y, theta);
    }

    void pogoPathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        std::vector<int64_t> times;
        std::vector<std::array<double, 3>> poses_xyt;
        times.reserve(msg->poses.size());
        poses_xyt.reserve(msg->poses.size());
        for (const auto& pose_stamped : msg->poses) {
            const auto& p = pose_stamped.pose.position;
            const auto& q = pose_stamped.pose.orientation;
            times.push_back(stampToMicroseconds(pose_stamped.header.stamp));
            poses_xyt.push_back({p.x, p.y, yawFromQuaternion(q.x, q.y, q.z, q.w)});
        }

        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            path_times_ = std::move(times);
            path_poses_xyt_ = std::move(poses_xyt);
            path_updated_ = true;
        }
        notifyWorker();
    }

    // Persist a mono8 image message as a PNG (low compression level: several
    // times faster to encode than the default at slightly larger files).
    static bool writeImage(const sensor_msgs::msg::Image& msg, const fs::path& path) {
        cv::Mat img(msg.height, msg.width, CV_8UC1,
                    const_cast<uint8_t*>(msg.data.data()), msg.step);
        return cv::imwrite(path.string(), img, {cv::IMWRITE_PNG_COMPRESSION, 1});
    }

    void localMapCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
                          const sensor_msgs::msg::Image::ConstSharedPtr& cumul_msg,
                          const dre::msg::LocalMapInfo::ConstSharedPtr& info_msg) {
        // Odometry pose from the message; used for the keyframe gate (motion
        // is measured against smooth, always-available odometry) and as the
        // initial scan pose if the pogo path does not cover this stamp yet.
        lgmath::se3::Transformation odom_pose = makePose(info_msg->x, info_msg->y, info_msg->theta);

        // Keyframe gate, same as the offline problem setup: skip the frame
        // unless sufficient motion has accumulated since the last added scan.
        if (have_prev_kf_) {
            lgmath::se3::Transformation T_kf_rel = odom_pose.inverse() * T_kf_prev_;
            double del_x = T_kf_rel.r_ab_inb()(0);
            double del_y = T_kf_rel.r_ab_inb()(1);
            double translation_mag = std::sqrt(std::pow(del_x, 2) + std::pow(del_y, 2));
            double rotation_mag = std::abs(T_kf_rel.vec()(5)) * 180.0 / M_PI; // yaw, degrees
            if (translation_mag < max_kf_dist_ && rotation_mag < max_kf_rot_) {
                return;
            }
        }

        int64_t timestamp_us = stampToMicroseconds(info_msg->header.stamp);

        // Prefer the pose-graph pose for the scan itself. The path message for
        // this stamp may not have arrived yet; the odometry pose then stands in
        // until the next path update corrects it.
        lgmath::se3::Transformation scan_pose = odom_pose;
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            auto path_pose = lookupPathPose(path_times_, path_poses_xyt_, timestamp_us);
            if (path_pose.has_value()) {
                scan_pose = *path_pose;
            }
        }

        // Queue the raw messages for the mapping thread: it writes the PNGs
        // and builds the LocalMapScan itself (see integrateOnce), so this
        // callback never blocks on disk I/O and intake keeps up regardless
        // of write latency.
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_scans_.push_back(PendingScan{
                img_msg, cumul_msg, timestamp_us, next_scan_id_,
                info_msg->resolution, scan_pose, std::chrono::steady_clock::now()});
        }

        T_kf_prev_ = odom_pose;
        have_prev_kf_ = true;
        RCLCPP_INFO(get_logger(), "keyframe %d queued (stamp %ld)", next_scan_id_, timestamp_us);
        ++next_scan_id_;
        notifyWorker();
    }

    // Allocate voxels and accumulator entries within max_dist_ of pose, so
    // the parallel integration loop never inserts entries (hash-map insertion
    // is not thread-safe; updates to distinct entries are fine). Voxels
    // already covered by a previously allocated disk (skip_center) are skipped
    // with a cheap distance test instead of ~4M redundant hash operations —
    // consecutive keyframes overlap ~96%, so this cuts allocation to the new
    // crescent. The 2-voxel margin keeps the test conservative for disks whose
    // pose drifted slightly (below the re-allocation threshold) since they
    // were allocated.
    void allocateScanRegion(const lgmath::se3::Transformation& pose,
                            const std::optional<ba::VoxelMap::Index>& skip_center) {
        const double res = voxel_map_->res();
        const auto pose_mat = pose.matrix();
        const auto center = voxel_map_->index(pose_mat(0, 3), pose_mat(1, 3));
        const int32_t n = static_cast<int32_t>(std::ceil(max_dist_ / res));
        const double skip_dist = max_dist_ - 2.0 * res;

        // Scan the disk in parallel (pure arithmetic), collecting the indices
        // that need allocation; insert serially afterwards since hash-map
        // insertion is not thread-safe.
        std::vector<std::vector<ba::VoxelMap::Index>> to_insert(omp_get_max_threads());
        #pragma omp parallel
        {
            auto& local = to_insert[omp_get_thread_num()];
            #pragma omp for schedule(static)
            for (int32_t da = -n; da <= n; ++da) {
                for (int32_t db = -n; db <= n; ++db) {
                    if (std::sqrt(std::pow(da * res, 2) + std::pow(db * res, 2)) > max_dist_) {
                        continue;
                    }
                    ba::VoxelMap::Index idx{center.first + da, center.second + db};
                    if (skip_center.has_value()) {
                        double dpa = static_cast<double>(idx.first - skip_center->first) * res;
                        double dpb = static_cast<double>(idx.second - skip_center->second) * res;
                        if (std::sqrt(dpa * dpa + dpb * dpb) <= skip_dist) {
                            continue;  // already allocated by the skip_center disk
                        }
                    }
                    local.push_back(idx);
                }
            }
        }
        for (const auto& local : to_insert) {
            for (const auto& idx : local) {
                voxel_map_->init_voxel(idx);
                voxel_accum_.try_emplace(idx, 0.0, 0.0);
            }
        }
        last_alloc_center_ = center;
    }

    // Same per-voxel computation as the offline DrBASolver::update_map: each
    // voxel is the inverse-variance-weighted mean of the interpolated scan
    // intensities, intensity = sum(I_i / cov_i) / sum(1 / cov_i) — but
    // maintained incrementally. The per-voxel running sums persist across
    // updates, so integrating (sign=+1) or removing (sign=-1) one scan only
    // touches that scan's own footprint while all other scans' contributions
    // stay in the sums. A pose change is applied by removing the scan at its
    // old pose and re-adding it at the new one; removal cancels the exact
    // values that were added because interpolation is a deterministic function
    // of (scan data, pose, voxel). Contributions are limited to max_dist_
    // around the scan pose (the region init_map allocates) rather than the
    // scan's full image coverage.
    void accumulateScan(const ba::Scan& scan, double sign) {
        const double res = voxel_map_->res();
        const auto pose_mat = scan.pose().matrix();
        const auto center = voxel_map_->index(pose_mat(0, 3), pose_mat(1, 3));
        const int32_t n = static_cast<int32_t>(std::ceil(max_dist_ / res));

        #pragma omp parallel for schedule(static)
        for (int32_t da = -n; da <= n; ++da) {
            for (int32_t db = -n; db <= n; ++db) {
                if (std::sqrt(std::pow(da * res, 2) + std::pow(db * res, 2)) > max_dist_) {
                    continue;
                }
                ba::VoxelMap::Index voxel_idx{center.first + da, center.second + db};
                auto it = voxel_accum_.find(voxel_idx);
                if (it == voxel_accum_.end()) {
                    // Outside the allocated map (possible after pose updates)
                    continue;
                }
                double voxel_x = static_cast<double>(voxel_idx.first) * res;
                double voxel_y = static_cast<double>(voxel_idx.second) * res;
                std::optional<ba::Scan::Measurement> interp_meas = scan.interpolate(voxel_x, voxel_y);
                // If scan is outside coverage, no intensity will be provided
                if (!interp_meas.has_value()) {
                    continue;
                }
                double inv_cov = 1.0 / interp_meas->covariance;
                auto& sums = it->second;
                sums.first += sign * interp_meas->intensity * inv_cov;  // sum(I / cov)
                sums.second += sign * inv_cov;                          // sum(1 / cov)

                // Refresh the map value; the last pass over a voxel this cycle
                // leaves the final ratio. Real single-scan weights are O(1),
                // so weights below the threshold are float residue left by
                // removals: clear them so unobserved voxels return to zero.
                double& voxel = voxel_map_->at(voxel_idx);
                if (sums.second > kMinAccumWeight) {
                    voxel = sums.first / sums.second;
                } else {
                    sums = {0.0, 0.0};
                    voxel = 0.0;
                }
            }
        }
    }

    bool ensureLoaded(ba::Scan& scan) {
        if (scan.data_loaded()) {
            return true;
        }
        try {
            scan.load_data();
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "Failed to load data for scan %d: %s, skipping it this update.",
                        scan.id(), e.what());
            return false;
        }
        return true;
    }

    // Keep only the most recent max_loaded_scans_ scans' image data in memory
    void enforceLoadedScanLimit() {
        if (max_loaded_scans_ < 1) {
            return;
        }
        const auto scan_ids = scan_manager_.get_all_scan_ids();
        const int num_scans = static_cast<int>(scan_ids.size());
        for (int i = 0; i < num_scans - max_loaded_scans_; ++i) {
            auto scan = scan_manager_.get_scan(scan_ids[i]);
            if (scan->data_loaded()) {
                scan->unload_data();
            }
        }
    }

    // Pose difference measured as worst-case voxel displacement: translation
    // plus the yaw change's lever arm at the edge of the scan's update region
    double poseDelta(const lgmath::se3::Transformation& a, const lgmath::se3::Transformation& b) const {
        lgmath::se3::Transformation rel = a.inverse() * b;
        double del_x = rel.r_ab_inb()(0);
        double del_y = rel.r_ab_inb()(1);
        double del_theta = rel.vec()(5);
        return std::sqrt(del_x * del_x + del_y * del_y) + std::abs(del_theta) * max_dist_;
    }

    void updateLoop() {
        // Must be set from this thread: a std::thread starts with the global
        // OpenMP defaults (all cores), not the value set on the main thread.
        if (num_threads_ > 0) {
            omp_set_num_threads(num_threads_);
        }

        while (rclcpp::ok()) {
            {
                std::unique_lock<std::mutex> lock(update_cv_mutex_);
                update_cv_.wait_for(lock, std::chrono::milliseconds(200),
                                    [this] { return stop_ || work_pending_; });
                if (stop_) {
                    return;
                }
                work_pending_ = false;
            }
            integrateOnce();
            maybeSaveMap();
        }
    }

    void notifyWorker() {
        {
            std::lock_guard<std::mutex> lock(update_cv_mutex_);
            work_pending_ = true;
        }
        update_cv_.notify_one();
    }

    void integrateOnce() {
        auto t_start = std::chrono::steady_clock::now();
        // Snapshot the latest pogo path (if it changed since the last update)
        std::vector<int64_t> path_times;
        std::vector<std::array<double, 3>> path_poses_xyt;
        bool path_updated = false;
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            path_updated = path_updated_;
            path_updated_ = false;
            if (path_updated) {
                path_times = path_times_;
                path_poses_xyt = path_poses_xyt_;
            }
        }

        // Find existing scans whose pose-graph estimate moved beyond the
        // tolerance. Checked before draining, so newly queued scans (whose
        // poses get set from this same path snapshot below) are never in it.
        // Between loop closures pogo leaves old node poses untouched, so this
        // is empty in steady state.
        struct MovedScan {
            std::shared_ptr<ba::Scan> scan;
            lgmath::se3::Transformation new_pose;
            double delta;
        };
        std::vector<MovedScan> moved;
        size_t moved_total = 0;
        if (path_updated) {
            for (int scan_id : scan_manager_.get_all_scan_ids()) {
                auto scan = scan_manager_.get_scan(scan_id);
                auto path_pose = lookupPathPose(path_times, path_poses_xyt, scan->timestamp());
                if (path_pose.has_value()) {
                    double delta = poseDelta(scan->pose(), *path_pose);
                    if (delta > pose_update_tol_) {
                        moved.push_back({scan, *path_pose, delta});
                    }
                }
            }
            moved_total = moved.size();
            // A loop closure can move every scan at once; re-integrating them
            // all in one cycle stalls scan intake and map saves for minutes.
            // Process the worst offenders now — the rest stay at their old
            // poses and are re-detected (against the freshest path) next cycle.
            if (max_moved_per_update_ > 0 && moved.size() > static_cast<size_t>(max_moved_per_update_)) {
                std::partial_sort(moved.begin(), moved.begin() + max_moved_per_update_, moved.end(),
                                  [](const MovedScan& a, const MovedScan& b) { return a.delta > b.delta; });
                moved.resize(max_moved_per_update_);
            }
        }

        // Drain scans queued since the last update.
        std::vector<PendingScan> new_pending;
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            new_pending.swap(pending_scans_);
        }

        // Write each new keyframe's PNGs and build its LocalMapScan here, on
        // the worker thread, not the subscription callback: this is the only
        // place scan image data is read back off disk too (ensureLoaded,
        // below), so intake never blocks on I/O either direction.
        std::vector<std::pair<std::shared_ptr<ba::LocalMapScan>, std::chrono::steady_clock::time_point>> new_scans;
        new_scans.reserve(new_pending.size());
        double write_sec = 0.0;
        for (const auto& p : new_pending) {
            std::string filename = std::to_string(p.timestamp_us) + ".png";
            fs::path img_path = local_maps_dir_ / filename;
            fs::path cumul_img_path = cumul_dir_ / filename;
            auto t_write0 = std::chrono::steady_clock::now();
            if (!writeImage(*p.img_msg, img_path) || !writeImage(*p.cumul_msg, cumul_img_path)) {
                RCLCPP_WARN(get_logger(), "Failed to write local map images for stamp %ld, skipping scan.",
                            p.timestamp_us);
                continue;
            }
            write_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_write0).count();

            auto scan = std::make_shared<ba::LocalMapScan>(
                p.timestamp_us,
                p.scan_id,
                p.resolution,
                opt_opts_,
                p.scan_pose,
                p.scan_pose,  // no separate groundtruth pose available online
                img_path,
                cumul_img_path);
            new_scans.emplace_back(scan, p.arrival);
        }

        // Refresh poses from the path first so voxel allocation uses the
        // corrected pose.
        for (const auto& [scan, arrival] : new_scans) {
            if (path_updated) {
                auto path_pose = lookupPathPose(path_times, path_poses_xyt, scan->timestamp());
                if (path_pose.has_value()) {
                    scan->set_pose(*path_pose);
                }
            }
            allocateScanRegion(scan->pose(), last_alloc_center_);
            scan_manager_.add_scan(scan);
        }

        if (new_scans.empty() && moved.empty()) {
            return;
        }

        auto t0 = std::chrono::steady_clock::now();
        double alloc_sec = std::chrono::duration<double>(t0 - t_start).count();

        // How long the oldest queued keyframe waited for this timer tick
        double max_queue_wait_sec = 0.0;
        for (const auto& [scan, arrival] : new_scans) {
            max_queue_wait_sec = std::max(max_queue_wait_sec,
                std::chrono::duration<double>(t0 - arrival).count());
        }

        // Integrate new scans first so a large moved backlog (loop closure)
        // never delays fresh observations
        for (const auto& [scan, arrival] : new_scans) {
            if (ensureLoaded(*scan)) {
                accumulateScan(*scan, 1.0);
            }
        }
        auto t_new_done = std::chrono::steady_clock::now();

        // Re-integrate moved scans: remove their contributions at the old
        // pose, then add them back at the new pose. Voxels they don't touch
        // keep their sums (and thus their weighted mean over all observers).
        size_t moved_processed = 0;
        for (const auto& m : moved) {
            // Guarantee min_moved_per_update_ corrections per cycle so pose
            // corrections make steady progress even while keyframes stream in;
            // beyond that, yield to fresh keyframes (an unprocessed correction
            // is re-detected next cycle against an even fresher path, but a
            // queued keyframe just accumulates latency).
            if (moved_processed >= static_cast<size_t>(std::max(min_moved_per_update_, 0))) {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                if (!pending_scans_.empty()) {
                    break;
                }
            }
            auto& scan = m.scan;
            if (!ensureLoaded(*scan)) {
                continue;  // pose left unchanged; retried on the next path update
            }
            accumulateScan(*scan, -1.0);
            // Allocate voxels/accumulator entries around the new pose if the
            // update disk actually shifted (a yaw-only change never does);
            // the scan's old disk is already allocated and can be skipped
            const auto old_pose_mat = scan->pose().matrix();
            const auto old_center = voxel_map_->index(old_pose_mat(0, 3), old_pose_mat(1, 3));
            double shift = (scan->pose().inverse() * m.new_pose).r_ab_inb().head<2>().norm();
            scan->set_pose(m.new_pose);
            if (shift > 0.5 * voxel_map_->res()) {
                allocateScanRegion(m.new_pose, old_center);
            }
            accumulateScan(*scan, 1.0);
            ++moved_processed;
        }
        auto t_moved_done = std::chrono::steady_clock::now();

        enforceLoadedScanLimit();

        // Rebuild the voxel map's stored poses (init_map appends one entry per
        // call, including re-allocations, so replace the list wholesale)
        {
            const auto scan_ids = scan_manager_.get_all_scan_ids();
            std::vector<lgmath::se2::Transformation> poses;
            poses.reserve(scan_ids.size());
            for (int scan_id : scan_ids) {
                poses.push_back(scan_manager_.get_scan(scan_id)->pose2d());
            }
            voxel_map_->set_poses(scan_ids, poses);
        }

        changed_since_save_ = true;

        double new_sec = std::chrono::duration<double>(t_new_done - t0).count();
        double moved_sec = std::chrono::duration<double>(t_moved_done - t_new_done).count();

        RCLCPP_INFO(get_logger(),
                    "[timing] integrate: queue wait %.1f s (max) | write %.2f s | allocate %.2f s | "
                    "integrate %zu new %.2f s | re-integrate %zu/%zu moved %.2f s | %d scans, %zu voxels",
                    max_queue_wait_sec, write_sec, alloc_sec,
                    new_scans.size(), new_sec,
                    moved_processed, moved_total, moved_sec,
                    scan_manager_.num_scans(), voxel_map_->size());
    }

    // Save and publish the map at most once per map_update_period_sec
    void maybeSaveMap() {
        if (!changed_since_save_) {
            return;
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_save_).count() < map_update_period_sec_) {
            return;
        }

        voxel_map_->save_to_file(map_file_path_, scan_manager_);
        double save_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - now).count();
        last_save_ = std::chrono::steady_clock::now();
        changed_since_save_ = false;

        std_msgs::msg::String msg;
        msg.data = map_file_path_;
        map_path_pub_->publish(msg);

        RCLCPP_INFO(get_logger(), "[timing] map saved and published in %.2f s (%zu voxels)",
                    save_sec, voxel_map_->size());
    }

    // Config
    double voxel_map_res_ = 0.1;
    double max_dist_ = 80.0;
    double map_update_period_sec_ = 5.0;
    double max_kf_dist_ = 5.0;   // meters
    double max_kf_rot_ = 30.0;   // degrees
    int max_loaded_scans_ = 100; // scans kept loaded between updates (<1 = all)
    double pose_update_tol_ = 0.01;  // meters of map-edge displacement before a scan is re-integrated
    int max_moved_per_update_ = 20;  // max pose-corrected scans re-integrated per cycle (<1 = unbounded)
    int min_moved_per_update_ = 5;   // corrections guaranteed per cycle before yielding to new keyframes
    int num_threads_ = 0;            // OpenMP threads for map updates (<1 = all cores)
    // Accumulated weights below this are float residue from scan removals
    static constexpr double kMinAccumWeight = 1e-6;
    ba::OptimizationOptions opt_opts_;

    // Keyframe image storage
    fs::path local_maps_dir_;
    fs::path cumul_dir_;

    // Keyframing state (subscription thread only)
    bool have_prev_kf_ = false;
    lgmath::se3::Transformation T_kf_prev_;
    int next_scan_id_ = 0;

    // A keyframe queued by the subscription thread, not yet written to disk
    // or turned into a LocalMapScan (both happen on the mapping thread).
    struct PendingScan {
        sensor_msgs::msg::Image::ConstSharedPtr img_msg;
        sensor_msgs::msg::Image::ConstSharedPtr cumul_msg;
        int64_t timestamp_us;
        int scan_id;
        double resolution;
        lgmath::se3::Transformation scan_pose;
        std::chrono::steady_clock::time_point arrival;
    };

    // Scans queued by the subscription thread, drained by the mapping timer
    std::mutex pending_mutex_;
    std::vector<PendingScan> pending_scans_;

    // Latest /pogo_path, written by the path callback, consumed by the mapping
    // timer (which applies it to the scans) and by new-scan initialization
    std::mutex path_mutex_;
    std::vector<int64_t> path_times_;                     // microseconds, ascending
    std::vector<std::array<double, 3>> path_poses_xyt_;   // (x, y, yaw)
    bool path_updated_ = false;

    // Output
    std::string output_path_;
    std::string map_file_path_;

    std::unique_ptr<ba::VoxelMap> voxel_map_;
    // Per-voxel running sums (sum(I/cov), sum(1/cov)); entries mirror the
    // voxel map's allocation and persist across updates
    ankerl::unordered_dense::map<ba::VoxelMap::Index, std::pair<double, double>> voxel_accum_;
    // Center of the most recently allocated (fully covered) disk, used to
    // skip redundant allocation work for overlapping consecutive keyframes
    std::optional<ba::VoxelMap::Index> last_alloc_center_;
    ba::ScanManager scan_manager_;

    using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, dre::msg::LocalMapInfo>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    // Map update worker thread
    std::thread update_thread_;
    std::condition_variable update_cv_;
    std::mutex update_cv_mutex_;
    bool work_pending_ = false;
    bool stop_ = false;
    std::chrono::steady_clock::time_point last_save_{};
    bool changed_since_save_ = false;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr map_path_pub_;
    rclcpp::CallbackGroup::SharedPtr sub_group_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> cumul_sub_;
    message_filters::Subscriber<dre::msg::LocalMapInfo> info_sub_;
    std::shared_ptr<Synchronizer> sync_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MappingNode>();
    // Two executor threads: scan intake and pogo path intake (the map update
    // runs on the node's own worker thread, not the executor)
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
