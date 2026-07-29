#include <ba/problem/problem.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>

namespace ba {

void Problem::preload_images() {
    if (scan_indices_.empty()) {
        throw std::runtime_error("Scan indices are empty. Cannot preload images.");
    }
    std::cout << "Preloading images for sequence: " << seq_id_ << std::endl;

    FrameProcessingOptions proc_opts;
    bool use_cumul_thresh = false;
    if (type_ == "ba") {
        proc_opts = opts_.ba_opts.frame_processing_opts;
        use_cumul_thresh = opts_.ba_opts.optimization_opts.use_cumul_thresh;
    } else if (type_ == "map") {
        proc_opts = opts_.map_opts.frame_processing_opts;
        use_cumul_thresh = opts_.map_opts.optimization_opts.use_cumul_thresh;
    } else if (type_ == "loc") {
        proc_opts = opts_.loc_opts.frame_processing_opts;
        use_cumul_thresh = opts_.loc_opts.optimization_opts.use_cumul_thresh;
    } else {
        throw std::invalid_argument("Unknown problem type: " + type_);
    }

    // Set up temporary folder for Gaussian-blurred images to be stored
    fs::path temp_dir = opts_.meas_path / seq_id_ / "blurred";
    fs::create_directories(temp_dir);

    // TODO: Add support for more than just local_maps
    if (proc_opts.input_type != "scans" && proc_opts.input_type != "local_maps") {
        throw std::invalid_argument("Input type " + proc_opts.input_type + " not supported yet.");
    }

    // Load in images
    fs::path all_img_dir = opts_.meas_path / seq_id_ / proc_opts.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        if (entry.path().extension() != ".png") {
            continue;
        }
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    // Load in cumulative return images
    std::vector<fs::path> cumul_files;
    if (use_cumul_thresh) {
        fs::path cumul_img_dir = opts_.meas_path / seq_id_ / "cumulated_returns";
        for (const auto& entry : fs::directory_iterator(cumul_img_dir)) {
            if (entry.is_regular_file()) {
                cumul_files.push_back(entry.path());
            }
        }
        std::sort(cumul_files.begin(), cumul_files.end());
    }

    // Set reference timestamp based on first scan time
    int64_t ref_timestamp = std::stoll(files[0].stem().string());
    scan_manager_.set_ref_timestamp(ref_timestamp);

    // Loop through all images
    double eps = proc_opts.min_int_val_tol;    
    double min_percent_nonzero = proc_opts.min_percent_nonzero;

    // Create temporary scan path
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << min_percent_nonzero;
    std::string s_min_percent_nonzero = oss.str(); // "1.50"
    std::replace(s_min_percent_nonzero.begin(), s_min_percent_nonzero.end(), '.', '_'); // "1_50"
    oss.str("");
    oss << std::fixed << std::setprecision(2) << proc_opts.min_int_val_tol;
    std::string s_min_int_val_tol = oss.str(); // "0.05"
    std::replace(s_min_int_val_tol.begin(), s_min_int_val_tol.end(), '.', '_'); // "0_05"
    oss.str("");
    oss << std::fixed << std::setprecision(2) << proc_opts.max_blur_sigma;
    std::string s_max_blur_sigma = oss.str(); // "15.00"
    std::replace(s_max_blur_sigma.begin(), s_max_blur_sigma.end(), '.', '_'); // "15_00"

    fs::path temp_img_dir = temp_dir / proc_opts.input_type;
    fs::path temp_folder;
    if (!proc_opts.adaptive_blur) {
        temp_folder = std::to_string(static_cast<int>(std::round(proc_opts.gauss_blur_sigma)));
    } else {
        temp_folder = s_min_percent_nonzero + "pct_" + s_min_int_val_tol + "minint" + s_max_blur_sigma + "maxsigma";
    }
    temp_img_dir /= temp_folder;
    fs::create_directories(temp_img_dir);

    for (int idx : scan_indices_) {
        const auto& img_path = files[idx];
        std::string img_stem = img_path.stem().string(); // stem is just timestamp

        // Add timestamp to list
        int64_t timestamp = std::stoll(img_stem); // in microseconds
        timestamps_.push_back(timestamp);

        std::string image_name = img_stem;
        image_name += ".png";
        fs::path temp_img_path = temp_img_dir / image_name;

        // Load in image as Eigen matrix
        if (!fs::exists(temp_img_path)) {
            // Only process if the temp image does not already exist
            cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
            img.convertTo(img, CV_32F, 1.0 / 255.0);

            if (img.empty()) {
                throw std::runtime_error("Failed to load image: " + img_path.string());
            }

            // Do smart selection of Gaussian sigma to ensure minimum percentage of non-zero pixels
            cv::Mat temp_img;
            double percent_nonzero = 0.0;
            double sigma = 3.0;
            if (!proc_opts.adaptive_blur) {
                sigma = proc_opts.gauss_blur_sigma;
            }
            while (percent_nonzero < min_percent_nonzero) {
                int ksize = (int(std::ceil(6 * sigma)) | 1);
                cv::GaussianBlur(img, temp_img, cv::Size(ksize, ksize), sigma);

                // Re-normalize after blur
                double min_val, max_val;
                cv::minMaxLoc(temp_img, &min_val, &max_val);
                temp_img = (temp_img - min_val) / (max_val - min_val);

                // Recompute percentage of non-zero pixels
                percent_nonzero = 100.0 * cv::countNonZero(temp_img > eps) / (temp_img.rows * temp_img.cols);

                if (!proc_opts.adaptive_blur || sigma > proc_opts.max_blur_sigma) {
                    // If not adaptive blur, just do one iteration
                    break;
                }

                // Increase sigma for next attempt
                sigma += 2.0;
            }
            img = temp_img;

            // std::cout << "Final sigma: " << sigma - 2.0 << ", percent non-zero: " << percent_nonzero << "%" << std::endl;

            // Clip to [0, 1] (should already be in this range, but just to be safe)
            cv::threshold(img, img, 0.0, 0.0, cv::THRESH_TOZERO);
            cv::threshold(img, img, 1.0, 1.0, cv::THRESH_TRUNC);

            // Convert back to 8-bit for saving
            img.convertTo(img, CV_8U, 255.0);
            // Save blurred image to temp directory for easy loading
            cv::imwrite(temp_img_path, img);
        }

        // Store path
        img_paths_.push_back(temp_img_path);
        if (use_cumul_thresh)
            cumul_paths_.push_back(cumul_files[idx]);

    }
    std::cout << "Preloaded " << img_paths_.size() << " images out of " << files.size() << " total images." << std::endl;
}





}   // namespace ba